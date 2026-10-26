"""Generate final MVP corruption-study report artifacts.

This script reads saved metrics and prediction CSVs only. It does not run
inference, train models, or modify checkpoints.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


EXPECTED_CLEAN = {
    "e0": {
        "accuracy": 0.9821397607074736,
        "macro_f1": 0.9638961355829906,
    },
    "e1": {
        "accuracy": 0.9436448760187273,
        "macro_f1": 0.8833699043245444,
    },
    "e2_canonical": {
        "accuracy": 0.9873417721518988,
        "macro_f1": 0.9679255699042882,
    },
    "e2_8": {
        "accuracy": 0.9875151725333796,
        "macro_f1": 0.9733555279688368,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate corruption-study report artifacts")
    parser.add_argument("--results-dir", default="corruption_study/results")
    parser.add_argument("--tolerance", type=float, default=0.005)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metrics_for_run(results_dir, run_name):
    path = results_dir / f"{run_name}__metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return load_json(path)


def prediction_runs(results_dir):
    pred_dir = results_dir / "predictions"
    for path in sorted(pred_dir.glob("*__predictions.csv")):
        yield path.name.replace("__predictions.csv", ""), path


def generate_per_class_recall(results_dir):
    rows = []
    for run_name, pred_path in prediction_runs(results_dir):
        metrics_path = results_dir / f"{run_name}__metrics.json"
        if not metrics_path.exists():
            continue

        meta = load_json(metrics_path)
        pred = pd.read_csv(pred_path)
        labels = sorted(pred["label_true"].astype(str).unique())
        precision, recall, f1, support = precision_recall_fscore_support(
            pred["label_true"].astype(str),
            pred["label_pred"].astype(str),
            labels=labels,
            zero_division=0,
        )
        for idx, label in enumerate(labels):
            rows.append({
                "run_name": run_name,
                "model": meta.get("model"),
                "condition": meta.get("condition"),
                "corruption_type": meta.get("corruption_type"),
                "image_corruption_type": meta.get("image_corruption_type"),
                "iot_corruption_type": meta.get("iot_corruption_type"),
                "severity": meta.get("severity"),
                "corruption_seed": meta.get("corruption_seed"),
                "label": label,
                "support": int(support[idx]),
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
            })

    out = pd.DataFrame(rows)
    out_path = results_dir / "per_class_recall_by_condition.csv"
    out.to_csv(out_path, index=False)
    return out_path, len(out)


def status_from_deltas(acc_delta, f1_delta, tolerance):
    return "PASS" if acc_delta <= tolerance and f1_delta <= tolerance else "FAIL"


def generate_severity0_sanity(results_dir, tolerance):
    rows = []
    for model, expected in EXPECTED_CLEAN.items():
        metrics = metrics_for_run(results_dir, f"{model}__clean__sev0")
        acc_delta = abs(float(metrics["accuracy"]) - expected["accuracy"])
        f1_delta = abs(float(metrics["macro_f1"]) - expected["macro_f1"])
        rows.append({
            "model": model,
            "accuracy": float(metrics["accuracy"]),
            "expected_accuracy": expected["accuracy"],
            "accuracy_delta": acc_delta,
            "macro_f1": float(metrics["macro_f1"]),
            "expected_macro_f1": expected["macro_f1"],
            "macro_f1_delta": f1_delta,
            "status": status_from_deltas(acc_delta, f1_delta, tolerance),
        })

    df = pd.DataFrame(rows)
    out_path = results_dir / "severity0_sanity_check.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Severity-0 Sanity Check\n\n")
        f.write("Severity 0 evaluates the original clean test inputs and should reproduce the audited clean metrics.\n\n")
        f.write(f"Tolerance: `{tolerance}` absolute delta for accuracy and Macro-F1.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    return out_path, df


def generate_passthrough_sanity(results_dir, tolerance):
    clean = metrics_for_run(results_dir, "e0__clean__sev0")
    passthrough = metrics_for_run(results_dir, "e0__image_passthrough__sev0")
    acc_delta = abs(float(passthrough["accuracy"]) - float(clean["accuracy"]))
    f1_delta = abs(float(passthrough["macro_f1"]) - float(clean["macro_f1"]))
    df = pd.DataFrame([{
        "model": "e0",
        "clean_accuracy": float(clean["accuracy"]),
        "passthrough_accuracy": float(passthrough["accuracy"]),
        "accuracy_delta": acc_delta,
        "clean_macro_f1": float(clean["macro_f1"]),
        "passthrough_macro_f1": float(passthrough["macro_f1"]),
        "macro_f1_delta": f1_delta,
        "status": status_from_deltas(acc_delta, f1_delta, tolerance),
    }])

    out_path = results_dir / "passthrough_sanity_check.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Passthrough Sanity Check\n\n")
        f.write("Passthrough materializes resized PNGs without corruption and verifies the image path wiring.\n\n")
        f.write(f"Tolerance: `{tolerance}` absolute delta for accuracy and Macro-F1.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    return out_path, df


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(results_dir)

    per_class_path, per_class_rows = generate_per_class_recall(results_dir)
    severity_path, severity_df = generate_severity0_sanity(results_dir, args.tolerance)
    passthrough_path, passthrough_df = generate_passthrough_sanity(results_dir, args.tolerance)

    print(f"wrote {per_class_path} rows={per_class_rows}")
    print(f"wrote {severity_path} status={','.join(severity_df['status'].tolist())}")
    print(f"wrote {passthrough_path} status={','.join(passthrough_df['status'].tolist())}")


if __name__ == "__main__":
    main()
