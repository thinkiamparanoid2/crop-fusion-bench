"""Enrich saved corruption-study metrics with fields and bootstrap CIs.

This script is offline post-processing only. It reads saved metrics JSON files
and prediction CSVs, then writes enriched metrics JSONs in place. It does not
run inference, train models, or touch checkpoints.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


E2_MODELS = {"e2_canonical", "e2_8", "e2_128_25"}
REGENERATED_CONDITIONS = {"clean_regen", "image_corrupted_only", "iot_corrupted_only", "both_corrupted"}

CHECKPOINTS = {
    "e0": "e0_image_only/canonical/best_e0_bs768_mobilenetv3large.pth",
    "e1": "e1_iot_only/canonical/best_e1_bs512_mlp128x64.pt",
    "e2_canonical": "e2_multimodal_fusion/canonical/best_e2_bs512_fusionmlp.pt",
    "e2_8": "e3_bottleneck_sweep/checkpoints/best_e2_matched_8.pt",
    "e2_128_25": "e2_canonical_128_25/checkpoints/best_e2_canonical_128_25.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Add bootstrap CIs to corruption-study metrics")
    parser.add_argument("--results-dir", default="corruption_study/results")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260609)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def prediction_path(results_dir, run_name):
    return results_dir / "predictions" / f"{run_name}__predictions.csv"


def load_predictions(results_dir, run_name):
    path = prediction_path(results_dir, run_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction CSV for {run_name}: {path}")
    df = pd.read_csv(path, dtype={"sample_id": str})
    return df[["sample_id", "y_true", "y_pred"]].copy()


def present_class_macro_f1(y_true, y_pred):
    labels = np.unique(y_true)
    f1_values = []
    for label in labels:
        true_pos = np.sum((y_true == label) & (y_pred == label))
        false_pos = np.sum((y_true != label) & (y_pred == label))
        false_neg = np.sum((y_true == label) & (y_pred != label))
        denom = (2 * true_pos) + false_pos + false_neg
        f1_values.append(0.0 if denom == 0 else (2 * true_pos) / denom)
    return float(np.mean(f1_values)) if f1_values else 0.0


def point_metrics(y_true, y_pred):
    precision, _recall, _f1, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    _precision, _recall, weighted_f1, _support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": present_class_macro_f1(y_true, y_pred),
        "macro_precision": float(precision),
        "weighted_f1": float(weighted_f1),
    }


def bootstrap_absolute(y_true, y_pred, n_bootstrap, rng):
    n = len(y_true)
    acc = np.empty(n_bootstrap, dtype=np.float64)
    f1 = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        acc[i] = np.mean(yt == yp)
        f1[i] = present_class_macro_f1(yt, yp)
    return {
        "accuracy_ci_low": float(np.percentile(acc, 2.5)),
        "accuracy_ci_high": float(np.percentile(acc, 97.5)),
        "macro_f1_ci_low": float(np.percentile(f1, 2.5)),
        "macro_f1_ci_high": float(np.percentile(f1, 97.5)),
    }


def bootstrap_drop(clean_df, run_df, n_bootstrap, rng):
    merged = run_df.merge(
        clean_df,
        on="sample_id",
        suffixes=("_run", "_clean"),
        validate="one_to_one",
    )
    if not (merged["y_true_run"].to_numpy() == merged["y_true_clean"].to_numpy()).all():
        raise RuntimeError("Clean and corrupted y_true values differ after sample_id alignment")

    y_true = merged["y_true_run"].to_numpy(dtype=np.int64)
    y_pred_run = merged["y_pred_run"].to_numpy(dtype=np.int64)
    y_pred_clean = merged["y_pred_clean"].to_numpy(dtype=np.int64)

    n = len(merged)
    acc_drop = np.empty(n_bootstrap, dtype=np.float64)
    f1_drop = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        clean_pred = y_pred_clean[idx]
        run_pred = y_pred_run[idx]
        acc_drop[i] = np.mean(yt == clean_pred) - np.mean(yt == run_pred)
        f1_drop[i] = present_class_macro_f1(yt, clean_pred) - present_class_macro_f1(yt, run_pred)

    return {
        "accuracy_drop_ci_low": float(np.percentile(acc_drop, 2.5)),
        "accuracy_drop_ci_high": float(np.percentile(acc_drop, 97.5)),
        "macro_f1_drop_ci_low": float(np.percentile(f1_drop, 2.5)),
        "macro_f1_drop_ci_high": float(np.percentile(f1_drop, 97.5)),
    }


def clean_reference_run(model, condition):
    if model in E2_MODELS and condition in REGENERATED_CONDITIONS:
        return f"{model}__clean_regen__sev0", "clean_regen"
    return f"{model}__clean__sev0", "clean"


def run_name_from_metrics_path(path):
    return path.name.replace("__metrics.json", "")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    metrics_paths = sorted(results_dir.glob("*__metrics.json"))
    if not metrics_paths:
        raise FileNotFoundError(f"No metrics JSON files found in {results_dir}")

    prediction_cache = {}
    rng = np.random.default_rng(args.seed)
    updated = 0

    for metrics_path in metrics_paths:
        metrics = load_json(metrics_path)
        run_name = run_name_from_metrics_path(metrics_path)
        model = metrics["model"]
        condition = metrics["condition"]

        run_pred = prediction_cache.get(run_name)
        if run_pred is None:
            run_pred = load_predictions(results_dir, run_name)
            prediction_cache[run_name] = run_pred

        y_true = run_pred["y_true"].to_numpy(dtype=np.int64)
        y_pred = run_pred["y_pred"].to_numpy(dtype=np.int64)
        computed = point_metrics(y_true, y_pred)
        metrics.update(computed)

        reference_run, reference_condition = clean_reference_run(model, condition)
        clean_pred = prediction_cache.get(reference_run)
        if clean_pred is None:
            clean_pred = load_predictions(results_dir, reference_run)
            prediction_cache[reference_run] = clean_pred

        clean_metrics_path = results_dir / f"{reference_run}__metrics.json"
        clean_metrics = load_json(clean_metrics_path)

        metrics["clean_reference_condition"] = reference_condition
        metrics["clean_accuracy"] = float(clean_metrics["accuracy"])
        metrics["clean_macro_f1"] = float(clean_metrics["macro_f1"])
        metrics["accuracy_drop"] = metrics["clean_accuracy"] - float(metrics["accuracy"])
        metrics["macro_f1_drop"] = metrics["clean_macro_f1"] - float(metrics["macro_f1"])
        metrics["relative_macro_f1_drop_percent"] = (
            100.0 * metrics["macro_f1_drop"] / metrics["clean_macro_f1"]
            if metrics["clean_macro_f1"]
            else None
        )

        metrics.update(bootstrap_absolute(y_true, y_pred, args.n_bootstrap, rng))
        metrics.update(bootstrap_drop(clean_pred, run_pred, args.n_bootstrap, rng))

        metrics["training_seed"] = metrics.get("training_seed")
        metrics["checkpoint_path"] = metrics.get("checkpoint_path") or CHECKPOINTS.get(model)
        metrics["config_path"] = metrics.get("config_path")
        metrics["seed"] = metrics.get("seed", 42)
        metrics["timestamp"] = metrics.get("timestamp") or datetime.fromtimestamp(metrics_path.stat().st_mtime).isoformat()
        metrics["bootstrap_n"] = args.n_bootstrap
        metrics["bootstrap_ci"] = 0.95
        metrics["bootstrap_seed"] = args.seed
        metrics["bootstrap_macro_f1_handling"] = "present-class macro-F1; labels absent from a resample are skipped"
        metrics["bootstrap_drop_method"] = "paired by sample_id using the same bootstrap indices for clean and condition predictions"
        metrics["note"] = metrics.get("note", "Inference only; no training.")

        save_json(metrics_path, metrics)
        updated += 1

    settings = {
        "n_bootstrap": args.n_bootstrap,
        "ci": 0.95,
        "seed": args.seed,
        "absolute_ci_method": "resample saved condition predictions with replacement",
        "drop_ci_method": "paired bootstrap aligned by sample_id",
        "macro_f1_handling": "present-class macro-F1; labels absent from a resample are skipped",
        "clean_reference_rule": {
            "e0_e1": "severity-0 clean",
            "e2_regenerated_conditions": "severity-0 clean_regen",
        },
    }
    save_json(results_dir / "bootstrap_settings.json", settings)
    print(f"updated {updated} metrics JSON files")
    print(f"wrote {results_dir / 'bootstrap_settings.json'}")


if __name__ == "__main__":
    main()
