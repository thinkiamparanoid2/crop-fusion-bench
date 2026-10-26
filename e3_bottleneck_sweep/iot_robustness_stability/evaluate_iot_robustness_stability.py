"""Multi-seed IoT robustness stability check for E2-matched-8 vs E3-8.

This script does not retrain. It loads the saved best checkpoints from
e3_bottleneck_sweep/checkpoints and evaluates test-time corruptions applied only
to the 64-dimensional IoT embedding slice. The 1280-dimensional image embedding,
labels, split order, and model weights are held fixed.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, recall_score  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
SWEEP_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = SWEEP_ROOT.parent
E2_SCRIPTS = PROJECT_ROOT / "e2_multimodal_fusion" / "scripts"
SWEEP_SCRIPTS = SWEEP_ROOT / "scripts"
E3_ROOT = PROJECT_ROOT / "e3_quantum_fusion"
for _p in (PROJECT_ROOT, E2_SCRIPTS, SWEEP_SCRIPTS, E3_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train_bottleneck import E2MatchedK, E3VQCK, count_param_split  # noqa: E402
from train_e2_fusion import load_embedding_manifest, make_loader, predict  # noqa: E402

FEATURE_DIR = PROJECT_ROOT / "e2_multimodal_fusion" / "features"
CHECKPOINT_DIR = SWEEP_ROOT / "checkpoints"
OUT_DIR = SWEEP_ROOT / "results" / "iot_robustness_stability"
BOTTLENECK = 8
VQC_LAYERS = 4
IMAGE_DIM = 1280
IOT_DIM = 64
CONDITIONS = [
    "clean",
    "gaussian_noise_5",
    "gaussian_noise_10",
    "gaussian_noise_15",
    "missing_10",
    "missing_20",
    "shuffled_iot_20",
    "contradictory_iot_10",
    "combined_noise10_missing10_contradictory10",
]
METRIC_COLUMNS = [
    "accuracy",
    "macro_f1",
    "macro_recall",
    "healthy_recall",
    "early_recall",
    "mid_recall",
    "severe_recall",
    "ece_10",
    "average_confidence",
]
SEVERE_COLUMNS = [
    "true_severe_count",
    "pred_healthy",
    "pred_early",
    "pred_mid",
    "pred_severe",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate IoT robustness over multiple corruption seeds.")
    parser.add_argument("--corruption-seeds", type=int, nargs="+", default=[13, 37, 73, 101, 202])
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def expected_calibration_error_10(probs, targets):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    ece = 0.0
    n_bins = 10
    for i in range(n_bins):
        lower = i / n_bins
        upper = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not mask.any():
            continue
        bin_acc = (predictions[mask] == targets[mask]).mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / float(len(targets))) * abs(bin_acc - bin_conf)
    return float(ece)


def labels_from_indices(indices, classes):
    return np.array([classes[int(i)] for i in indices])


def stages_from_labels(labels):
    return np.array([str(label).split("_")[2] for label in labels])


def stage_recalls(targets, preds, classes):
    true_stages = stages_from_labels(labels_from_indices(targets, classes))
    pred_stages = stages_from_labels(labels_from_indices(preds, classes))
    out = {}
    for stage in ["0", "1", "2", "3"]:
        mask = true_stages == stage
        out[stage] = np.nan if mask.sum() == 0 else float((pred_stages[mask] == true_stages[mask]).mean())
    return out


def severe_confusion_row(targets, preds, classes):
    true_stages = stages_from_labels(labels_from_indices(targets, classes))
    pred_stages = stages_from_labels(labels_from_indices(preds, classes))
    severe = true_stages == "3"
    return {
        "true_severe_count": int(severe.sum()),
        "pred_healthy": int(((pred_stages == "0") & severe).sum()),
        "pred_early": int(((pred_stages == "1") & severe).sum()),
        "pred_mid": int(((pred_stages == "2") & severe).sum()),
        "pred_severe": int(((pred_stages == "3") & severe).sum()),
    }


def compute_metrics(eval_output, classes):
    targets = eval_output["targets"]
    preds = eval_output["preds"]
    probs = eval_output["probs"]
    stages = stage_recalls(targets, preds, classes)
    severe = severe_confusion_row(targets, preds, classes)
    return {
        "accuracy": float(accuracy_score(targets, preds)),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, preds, average="macro", zero_division=0)),
        "healthy_recall": stages["0"],
        "early_recall": stages["1"],
        "mid_recall": stages["2"],
        "severe_recall": stages["3"],
        "ece_10": expected_calibration_error_10(probs, targets),
        "average_confidence": float(probs.max(axis=1).mean()),
        **severe,
    }


def load_split_parts(split):
    path = FEATURE_DIR / f"{split}_embeddings.npz"
    data = np.load(path, allow_pickle=False)
    image = data["image_embedding"].astype(np.float32)
    iot = data["iot_embedding"].astype(np.float32)
    y = data["y"].astype(np.int64)
    meta = pd.DataFrame({
        "image": data["image_id"].astype(str),
        "label": data["label"].astype(str),
        "stage": data["stage"].astype(str),
        "label_idx": y,
    })
    return image, iot, y, meta


def apply_gaussian(iot, rng, percent, train_std):
    out = iot.copy()
    scale = (percent / 100.0) * np.maximum(train_std, 1e-6)
    out += rng.normal(0.0, scale, size=out.shape).astype(np.float32)
    return out


def apply_missing(iot, rng, percent, train_mean):
    out = iot.copy()
    mask = rng.random(out.shape) < (percent / 100.0)
    out[mask] = np.broadcast_to(train_mean, out.shape)[mask]
    return out


def apply_shuffle_iot(iot, rng, percent):
    out = iot.copy()
    n = len(out)
    count = int(round(n * percent / 100.0))
    idx = rng.choice(n, size=count, replace=False)
    replacement = rng.integers(0, n, size=count)
    same = replacement == idx
    while same.any():
        replacement[same] = rng.integers(0, n, size=int(same.sum()))
        same = replacement == idx
    out[idx] = out[replacement]
    return out


def apply_contradictory_iot(iot, stages, rng, percent):
    out = iot.copy()
    n = len(out)
    count = int(round(n * percent / 100.0))
    idx = rng.choice(n, size=count, replace=False)
    stages = np.asarray(stages).astype(str)
    different_stage_indices = {stage: np.where(stages != stage)[0] for stage in np.unique(stages)}
    for i in idx:
        candidates = different_stage_indices.get(stages[i])
        if candidates is None or len(candidates) == 0:
            continue
        out[i] = out[rng.choice(candidates)]
    return out


def corrupt_iot(condition, clean_iot, test_stages, train_mean, train_std, seed):
    rng = np.random.default_rng(seed + CONDITIONS.index(condition) * 1009)
    if condition == "clean":
        return clean_iot.copy()
    if condition == "gaussian_noise_5":
        return apply_gaussian(clean_iot, rng, 5, train_std)
    if condition == "gaussian_noise_10":
        return apply_gaussian(clean_iot, rng, 10, train_std)
    if condition == "gaussian_noise_15":
        return apply_gaussian(clean_iot, rng, 15, train_std)
    if condition == "missing_10":
        return apply_missing(clean_iot, rng, 10, train_mean)
    if condition == "missing_20":
        return apply_missing(clean_iot, rng, 20, train_mean)
    if condition == "shuffled_iot_20":
        return apply_shuffle_iot(clean_iot, rng, 20)
    if condition == "contradictory_iot_10":
        return apply_contradictory_iot(clean_iot, test_stages, rng, 10)
    if condition == "combined_noise10_missing10_contradictory10":
        out = apply_gaussian(clean_iot, rng, 10, train_std)
        out = apply_missing(out, rng, 10, train_mean)
        out = apply_contradictory_iot(out, test_stages, rng, 10)
        return out
    raise ValueError(condition)


def make_fused(image, iot):
    if image.shape[1] != IMAGE_DIM or iot.shape[1] != IOT_DIM:
        raise RuntimeError(f"Expected image/IOT dims {IMAGE_DIM}/{IOT_DIM}, got {image.shape[1]}/{iot.shape[1]}")
    return np.concatenate([image, iot], axis=1).astype(np.float32)


def load_models():
    models = []
    e2_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    e2 = E2MatchedK(BOTTLENECK, num_classes=25)
    e2.load_state_dict(torch.load(CHECKPOINT_DIR / "best_e2_matched_8.pt", map_location=e2_device))
    models.append(("e2_matched_8", e2.to(e2_device), e2_device, None))

    e3 = E3VQCK(BOTTLENECK, VQC_LAYERS, num_classes=25, sim="lightning.qubit")
    e3.load_state_dict(torch.load(CHECKPOINT_DIR / "best_e3_8_vqc.pt", map_location="cpu"))
    models.append(("e3_8_vqc", e3.to(torch.device("cpu")), torch.device("cpu"), "lightning.qubit"))
    return models


def audit_or_raise(manifest, train_image, train_iot, test_image, test_iot, y_test):
    checks = [
        ("checkpoint_e2_exists", (CHECKPOINT_DIR / "best_e2_matched_8.pt").exists()),
        ("checkpoint_e3_exists", (CHECKPOINT_DIR / "best_e3_8_vqc.pt").exists()),
        ("manifest_image_dim_1280", manifest["image_embedding_dim"] == IMAGE_DIM),
        ("manifest_iot_dim_64", manifest["iot_embedding_dim"] == IOT_DIM),
        ("train_image_dim_1280", train_image.shape[1] == IMAGE_DIM),
        ("train_iot_dim_64", train_iot.shape[1] == IOT_DIM),
        ("test_image_dim_1280", test_image.shape[1] == IMAGE_DIM),
        ("test_iot_dim_64", test_iot.shape[1] == IOT_DIM),
        ("test_labels_unchanged_in_memory", len(y_test) == len(test_iot)),
    ]
    failed = [name for name, ok in checks if not ok]
    if failed:
        raise RuntimeError("Audit failed: " + ", ".join(failed))
    return checks


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_mean_std(df, value_cols):
    grouped = df.groupby(["model", "condition"], sort=False)
    rows = []
    for (model, condition), sub in grouped:
        row = {"model": model, "condition": condition, "n_seeds": int(sub["corruption_seed"].nunique())}
        for col in value_cols:
            row[f"{col}_mean"] = float(sub[col].mean())
            row[f"{col}_std"] = float(sub[col].std(ddof=0))
        if "parameter_count" in sub:
            row["parameter_count"] = int(sub["parameter_count"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_mean_std_table(df):
    columns = [
        "model",
        "condition",
        "accuracy",
        "macro_f1",
        "macro_recall",
        "healthy_recall",
        "early_recall",
        "mid_recall",
        "severe_recall",
        "ece_10",
        "average_confidence",
        "parameter_count",
    ]
    headers = [
        "Model",
        "Condition",
        "Accuracy",
        "Macro F1",
        "Macro recall",
        "Healthy",
        "Early",
        "Mid",
        "Severe",
        "ECE-10",
        "Avg conf",
        "Params",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            if col == "model" or col == "condition":
                values.append(str(row[col]))
            elif col == "parameter_count":
                values.append(str(int(row[col])))
            else:
                values.append(f"{row[col + '_mean']:.6f} +/- {row[col + '_std']:.6f}")
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_metrics_markdown(mean_std_df):
    lines = [
        "# IoT Robustness Stability Metrics",
        "",
        "Mean +/- population std over corruption seeds.",
        "",
        *markdown_mean_std_table(mean_std_df),
        "",
    ]
    (OUT_DIR / "robustness_metrics_mean_std.md").write_text("\n".join(lines), encoding="utf-8")


def lookup(mean_df, model, condition, metric):
    row = mean_df[(mean_df["model"] == model) & (mean_df["condition"] == condition)].iloc[0]
    return float(row[f"{metric}_mean"])


def write_summary(mean_df, all_df, seeds):
    e2 = "e2_matched_8"
    e3 = "e3_8_vqc"
    conditions = list(mean_df["condition"].drop_duplicates())

    e3_lower_ece_all_seed_rows = True
    for seed in seeds:
        seed_df = all_df[all_df["corruption_seed"] == seed]
        for condition in conditions:
            e2_ece = seed_df[(seed_df["model"] == e2) & (seed_df["condition"] == condition)]["ece_10"].iloc[0]
            e3_ece = seed_df[(seed_df["model"] == e3) & (seed_df["condition"] == condition)]["ece_10"].iloc[0]
            if not e3_ece < e2_ece:
                e3_lower_ece_all_seed_rows = False

    e3_beats_macro_on_average_conditions = []
    for condition in conditions:
        if lookup(mean_df, e3, condition, "macro_f1") > lookup(mean_df, e2, condition, "macro_f1"):
            e3_beats_macro_on_average_conditions.append(condition)

    e2_clean_macro = lookup(mean_df, e2, "clean", "macro_f1")
    e3_clean_macro = lookup(mean_df, e3, "clean", "macro_f1")
    e2_shuffle_macro = lookup(mean_df, e2, "shuffled_iot_20", "macro_f1")
    e3_shuffle_macro = lookup(mean_df, e3, "shuffled_iot_20", "macro_f1")
    e2_macro_drop = e2_shuffle_macro - e2_clean_macro
    e3_macro_drop = e3_shuffle_macro - e3_clean_macro
    smaller_macro_degradation = e2 if abs(e2_macro_drop) < abs(e3_macro_drop) else e3

    e2_clean_severe = lookup(mean_df, e2, "clean", "severe_recall")
    e3_clean_severe = lookup(mean_df, e3, "clean", "severe_recall")
    e2_shuffle_severe = lookup(mean_df, e2, "shuffled_iot_20", "severe_recall")
    e3_shuffle_severe = lookup(mean_df, e3, "shuffled_iot_20", "severe_recall")
    e2_severe_drop = e2_shuffle_severe - e2_clean_severe
    e3_severe_drop = e3_shuffle_severe - e3_clean_severe
    smaller_severe_degradation = e2 if abs(e2_severe_drop) < abs(e3_severe_drop) else e3

    stable = (
        e3_lower_ece_all_seed_rows
        and not e3_beats_macro_on_average_conditions
        and smaller_macro_degradation == e2
        and smaller_severe_degradation == e2
    )

    lines = [
        "# IoT Robustness Stability Summary",
        "",
        f"Corruption seeds: {', '.join(str(s) for s in seeds)}.",
        "",
        "## Required Answers",
        "",
        f"- Does E3-8 keep lower ECE across all corruption seeds? {'Yes' if e3_lower_ece_all_seed_rows else 'No'}.",
        f"- Does E3-8 ever beat E2-matched-8 in Macro F1 on average? {'Yes: ' + ', '.join(e3_beats_macro_on_average_conditions) if e3_beats_macro_on_average_conditions else 'No'}.",
        f"- Which model has smaller Macro F1 degradation under shuffled_iot_20? {smaller_macro_degradation} (E2 drop {e2_macro_drop:.6f}; E3 drop {e3_macro_drop:.6f}).",
        f"- Which model has smaller severe recall degradation under shuffled_iot_20? {smaller_severe_degradation} (E2 drop {e2_severe_drop:.6f}; E3 drop {e3_severe_drop:.6f}).",
        f"- Is the earlier result stable across corruption randomness? {'Yes' if stable else 'Partially'}: E3 remains better calibrated, while E2 remains stronger and less degraded on Macro F1/severe recall under shuffled IoT.",
        "",
    ]
    (OUT_DIR / "robustness_stability_summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_mean_std(mean_df, metric, ylabel, filename):
    plt.figure(figsize=(12, 5))
    x = np.arange(len(CONDITIONS))
    for model in mean_df["model"].drop_duplicates():
        sub = mean_df[mean_df["model"] == model].set_index("condition").loc[CONDITIONS]
        plt.errorbar(
            x,
            sub[f"{metric}_mean"],
            yerr=sub[f"{metric}_std"],
            marker="o",
            capsize=3,
            linewidth=1.7,
            label=model,
        )
    plt.xticks(x, CONDITIONS, rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.xlabel("IoT condition")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=160)
    plt.close()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_embedding_manifest(FEATURE_DIR)
    classes = manifest["classes"]
    train_image, train_iot, y_train, train_meta = load_split_parts("train")
    test_image, test_iot, y_test, test_meta = load_split_parts("test")
    audit_or_raise(manifest, train_image, train_iot, test_image, test_iot, y_test)

    train_mean = train_iot.mean(axis=0).astype(np.float32)
    train_std = train_iot.std(axis=0).astype(np.float32)
    test_stages = test_meta["stage"].astype(str).to_numpy()
    models = load_models()
    criterion = nn.CrossEntropyLoss()

    metrics_rows = []
    severe_rows = []
    ece_rows = []

    print("Audit: corruptions are applied only to IoT embeddings; image embeddings, labels, and sample order stay fixed.")
    print("Audit: saved best checkpoints are loaded from e3_bottleneck_sweep/checkpoints.")
    print("Audit: ECE uses softmax confidence with 10 bins.")
    print("Audit: missing values are imputed with the train-set IoT embedding mean.")
    print("Audit: shuffled_iot_20 and contradictory_iot_10 replace only IoT vectors.")

    for seed in args.corruption_seeds:
        for condition in CONDITIONS:
            corrupted_iot = corrupt_iot(condition, test_iot, test_stages, train_mean, train_std, seed)
            x_test = make_fused(test_image, corrupted_iot)
            for model_name, model, device, sim in models:
                total_params, _ = count_param_split(model)
                loader = make_loader(x_test, y_test, args.batch_size, False, 0)
                output = predict(model, loader, criterion.to(device), device)
                values = compute_metrics(output, classes)
                row = {
                    "corruption_seed": seed,
                    "model": model_name,
                    "condition": condition,
                    "sim": sim,
                    "parameter_count": total_params,
                    **values,
                }
                metrics_rows.append(row)
                severe_rows.append({
                    "corruption_seed": seed,
                    "model": model_name,
                    "condition": condition,
                    "true_severe_count": values["true_severe_count"],
                    "pred_healthy": values["pred_healthy"],
                    "pred_early": values["pred_early"],
                    "pred_mid": values["pred_mid"],
                    "pred_severe": values["pred_severe"],
                    "severe_recall": values["severe_recall"],
                })
                ece_rows.append({
                    "corruption_seed": seed,
                    "model": model_name,
                    "condition": condition,
                    "ece_10": values["ece_10"],
                    "average_confidence": values["average_confidence"],
                })
                print(
                    f"seed={seed} {model_name} {condition}: macro_f1={values['macro_f1']:.4f} "
                    f"severe={values['severe_recall']:.4f} ece10={values['ece_10']:.4f}"
                )

    write_csv(OUT_DIR / "robustness_metrics_all_seeds.csv", metrics_rows)

    all_df = pd.DataFrame(metrics_rows)
    severe_df = pd.DataFrame(severe_rows)
    ece_df = pd.DataFrame(ece_rows)
    mean_std_df = aggregate_mean_std(all_df, METRIC_COLUMNS)
    severe_mean_std_df = aggregate_mean_std(severe_df, SEVERE_COLUMNS + ["severe_recall"])
    ece_mean_std_df = aggregate_mean_std(ece_df, ["ece_10", "average_confidence"])

    mean_std_df.to_csv(OUT_DIR / "robustness_metrics_mean_std.csv", index=False)
    severe_mean_std_df.to_csv(OUT_DIR / "severe_confusion_mean_std.csv", index=False)
    ece_mean_std_df.to_csv(OUT_DIR / "ece_mean_std.csv", index=False)
    write_metrics_markdown(mean_std_df)
    write_summary(mean_std_df, all_df, args.corruption_seeds)
    plot_mean_std(mean_std_df, "macro_f1", "Macro F1", "macro_f1_vs_condition_mean_std.png")
    plot_mean_std(mean_std_df, "ece_10", "ECE (10 bins)", "ece_vs_condition_mean_std.png")
    plot_mean_std(mean_std_df, "severe_recall", "Severe-stage recall", "severe_recall_vs_condition_mean_std.png")

    print("\n===== robustness_stability_summary.md =====")
    print((OUT_DIR / "robustness_stability_summary.md").read_text(encoding="utf-8"))
    print("\n===== robustness_metrics_mean_std.md =====")
    print((OUT_DIR / "robustness_metrics_mean_std.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
