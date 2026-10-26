"""Evaluate E2-matched-8 vs E3-8 robustness to IoT-embedding corruptions.

No retraining is performed. The script loads the saved best checkpoints from
e3_bottleneck_sweep/checkpoints and corrupts only the 64-dimensional IoT
embedding slice of the cached fused test inputs.
"""

import csv
import json
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

SWEEP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SWEEP_ROOT.parent
E2_SCRIPTS = PROJECT_ROOT / "e2_multimodal_fusion" / "scripts"
for _p in (PROJECT_ROOT, E2_SCRIPTS, SWEEP_ROOT / "scripts", PROJECT_ROOT / "e3_quantum_fusion"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from train_bottleneck import E2MatchedK, E3VQCK, count_param_split  # noqa: E402
from train_e2_fusion import load_embedding_manifest, load_split, make_loader, predict  # noqa: E402

FEATURE_DIR = PROJECT_ROOT / "e2_multimodal_fusion" / "features"
CHECKPOINT_DIR = SWEEP_ROOT / "checkpoints"
OUT_DIR = SWEEP_ROOT / "results" / "iot_robustness"
BOTTLENECK = 8
VQC_LAYERS = 4
SEED = 42
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
        out[stage] = None if mask.sum() == 0 else float((pred_stages[mask] == true_stages[mask]).mean())
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
    by_stage = {stage: np.where(stages != stage)[0] for stage in ["0", "1", "2", "3"]}
    for i in idx:
        candidates = by_stage.get(stages[i])
        if candidates is None or len(candidates) == 0:
            continue
        out[i] = out[rng.choice(candidates)]
    return out


def corrupt_iot(condition, clean_iot, test_stages, train_mean, train_std):
    rng = np.random.default_rng(SEED + CONDITIONS.index(condition) * 1009)
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
    return np.concatenate([image, iot], axis=1).astype(np.float32)


def load_models():
    models = []

    e2 = E2MatchedK(BOTTLENECK, num_classes=25)
    e2.load_state_dict(torch.load(CHECKPOINT_DIR / "best_e2_matched_8.pt", map_location="cuda" if torch.cuda.is_available() else "cpu"))
    e2_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models.append(("e2_matched_8", e2.to(e2_device), e2_device, None))

    e3 = E3VQCK(BOTTLENECK, VQC_LAYERS, num_classes=25, sim="lightning.qubit")
    e3.load_state_dict(torch.load(CHECKPOINT_DIR / "best_e3_8_vqc.pt", map_location="cpu"))
    models.append(("e3_8_vqc", e3.to(torch.device("cpu")), torch.device("cpu"), "lightning.qubit"))
    return models


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, columns, headers):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for row in rows:
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def write_markdown_outputs(metrics_rows, severe_rows, ece_rows):
    metrics_cols = [
        "model", "condition", "accuracy", "macro_f1", "macro_recall",
        "healthy_recall", "early_recall", "mid_recall", "severe_recall",
        "ece_10", "average_confidence", "parameter_count",
    ]
    metrics_headers = [
        "Model", "Condition", "Accuracy", "Macro F1", "Macro recall",
        "Healthy", "Early", "Mid", "Severe", "ECE-10", "Avg conf", "Params",
    ]
    (OUT_DIR / "robustness_metrics.md").write_text(
        "\n".join(["# IoT Robustness Metrics", ""] + markdown_table(metrics_rows, metrics_cols, metrics_headers) + [""]),
        encoding="utf-8",
    )

    severe_cols = [
        "model", "condition", "true_severe_count", "pred_healthy", "pred_early",
        "pred_mid", "pred_severe", "severe_recall",
    ]
    severe_headers = [
        "Model", "Condition", "True severe", "Pred healthy", "Pred early",
        "Pred mid", "Pred severe", "Severe recall",
    ]
    (OUT_DIR / "severe_confusion_by_condition.md").write_text(
        "\n".join(["# Severe Confusion by IoT Condition", ""] + markdown_table(severe_rows, severe_cols, severe_headers) + [""]),
        encoding="utf-8",
    )

    ece_cols = ["model", "condition", "ece_10", "average_confidence"]
    ece_headers = ["Model", "Condition", "ECE-10", "Average confidence"]
    (OUT_DIR / "ece_by_condition.csv").write_text(
        pd.DataFrame(ece_rows).to_csv(index=False),
        encoding="utf-8",
    )


def condition_drop_summary(metrics_rows):
    by_model = {}
    for row in metrics_rows:
        by_model.setdefault(row["model"], {})[row["condition"]] = row
    lines = [
        "# IoT Robustness Summary",
        "",
        "Only the 64-dimensional IoT embedding slice was corrupted. Image embeddings, labels, splits, and weights were unchanged.",
        "",
        "## Clean Baselines",
        "",
        "| Model | Accuracy | Macro F1 | Early recall | Mid recall | Severe recall | ECE-10 | Avg confidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, rows in by_model.items():
        clean = rows["clean"]
        lines.append(
            f"| {model} | {clean['accuracy']:.6f} | {clean['macro_f1']:.6f} | {clean['early_recall']:.6f} | {clean['mid_recall']:.6f} | {clean['severe_recall']:.6f} | {clean['ece_10']:.6f} | {clean['average_confidence']:.6f} |"
        )

    lines.extend(["", "## Worst Drops vs Clean", ""])
    for model, rows in by_model.items():
        clean = rows["clean"]
        worst_macro = min(rows.values(), key=lambda r: r["macro_f1"] - clean["macro_f1"])
        worst_severe = min(rows.values(), key=lambda r: r["severe_recall"] - clean["severe_recall"])
        worst_ece = max(rows.values(), key=lambda r: r["ece_10"] - clean["ece_10"])
        lines.extend([
            f"### {model}",
            "",
            f"- Worst Macro F1 condition: `{worst_macro['condition']}` ({worst_macro['macro_f1']:.6f}, drop {worst_macro['macro_f1'] - clean['macro_f1']:.6f}).",
            f"- Worst severe-recall condition: `{worst_severe['condition']}` ({worst_severe['severe_recall']:.6f}, drop {worst_severe['severe_recall'] - clean['severe_recall']:.6f}).",
            f"- Worst ECE condition: `{worst_ece['condition']}` ({worst_ece['ece_10']:.6f}, change {worst_ece['ece_10'] - clean['ece_10']:.6f}).",
            "",
        ])

    lines.extend([
        "## Interpretation Prompts",
        "",
        "- If both models degrade similarly under IoT corruption, the 8D recovery is mostly robust to noisy sensor embeddings.",
        "- If E3-8 keeps lower ECE under corruption, that supports a calibration-oriented quantum result.",
        "- If severe recall collapses mainly under shuffled/contradictory IoT, then severe recognition depends on sensor-image consistency.",
        "",
    ])
    (OUT_DIR / "robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")


def plot_metric(metrics_rows, metric, ylabel, filename):
    df = pd.DataFrame(metrics_rows)
    plt.figure(figsize=(12, 5))
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        plt.plot(sub["condition"], sub[metric], marker="o", label=model)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(ylabel)
    plt.xlabel("IoT condition")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=160)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_embedding_manifest(FEATURE_DIR)
    classes = manifest["classes"]
    train_image, train_iot, y_train, train_meta = load_split_parts("train")
    test_image, test_iot, y_test, test_meta = load_split_parts("test")
    train_mean = train_iot.mean(axis=0).astype(np.float32)
    train_std = train_iot.std(axis=0).astype(np.float32)
    test_stages = test_meta["stage"].astype(str).to_numpy()
    models = load_models()

    metrics_rows = []
    severe_rows = []
    ece_rows = []
    criterion = nn.CrossEntropyLoss()

    for condition in CONDITIONS:
        corrupted_iot = corrupt_iot(condition, test_iot, test_stages, train_mean, train_std)
        x_test = make_fused(test_image, corrupted_iot)
        for model_name, model, device, sim in models:
            total_params, _ = count_param_split(model)
            loader = make_loader(x_test, y_test, 512, False, 0)
            output = predict(model, loader, criterion.to(device), device)
            values = compute_metrics(output, classes)
            row = {
                "model": model_name,
                "condition": condition,
                "sim": sim,
                "parameter_count": total_params,
                **values,
            }
            metrics_rows.append(row)
            severe_rows.append({
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
                "model": model_name,
                "condition": condition,
                "ece_10": values["ece_10"],
                "average_confidence": values["average_confidence"],
            })
            print(
                f"{model_name} {condition}: macro_f1={values['macro_f1']:.4f} "
                f"early={values['early_recall']:.4f} severe={values['severe_recall']:.4f} "
                f"ece10={values['ece_10']:.4f}"
            )

    write_csv(OUT_DIR / "robustness_metrics.csv", metrics_rows)
    write_csv(OUT_DIR / "severe_confusion_by_condition.csv", severe_rows)
    write_markdown_outputs(metrics_rows, severe_rows, ece_rows)
    condition_drop_summary(metrics_rows)
    plot_metric(metrics_rows, "macro_f1", "Macro F1", "macro_f1_vs_condition.png")
    plot_metric(metrics_rows, "ece_10", "ECE (10 bins)", "ece_vs_condition.png")
    plot_metric(metrics_rows, "severe_recall", "Severe-stage recall", "severe_recall_vs_condition.png")

    print("\n===== robustness_metrics.md =====")
    print((OUT_DIR / "robustness_metrics.md").read_text(encoding="utf-8"))
    print("\n===== robustness_summary.md =====")
    print((OUT_DIR / "robustness_summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

