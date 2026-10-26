"""Analyze where true Stage 3 samples are predicted."""

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SWEEP_ROOT = ROOT / "e3_bottleneck_sweep"
RESULTS_DIR = SWEEP_ROOT / "results"
OUT_CSV = RESULTS_DIR / "severe_stage_confusion.csv"
OUT_MD = RESULTS_DIR / "severe_stage_confusion.md"

RUN_SOURCES = [
    ("e2_matched_4", ROOT / "e3_quantum_fusion" / "results" / "e2_matched_4_test_predictions.csv"),
    ("e3_4_vqc", ROOT / "e3_quantum_fusion" / "results" / "e3_4_vqc_test_predictions.csv"),
    ("e2_matched_8", RESULTS_DIR / "e2_matched_8_test_predictions.csv"),
    ("e3_8_vqc", RESULTS_DIR / "e3_8_vqc_test_predictions.csv"),
]

STAGE_NAMES = {
    "0": "healthy",
    "1": "early",
    "2": "mid",
    "3": "severe",
}


def analyze(path):
    df = pd.read_csv(path, dtype=str)
    severe = df[df["stage"].astype(str) == "3"].copy()
    counts = {stage: int((severe["pred_stage"].astype(str) == stage).sum()) for stage in ["0", "1", "2", "3"]}
    total = len(severe)
    return total, counts


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for run, path in RUN_SOURCES:
        if not path.exists():
            continue
        total, counts = analyze(path)
        row = {
            "run": run,
            "true_stage": "3",
            "true_stage_name": "severe",
            "total_true_severe": total,
            "pred_healthy": counts["0"],
            "pred_early": counts["1"],
            "pred_mid": counts["2"],
            "pred_severe": counts["3"],
            "severe_recall": 0.0 if total == 0 else counts["3"] / total,
        }
        rows.append(row)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "run", "true_stage", "true_stage_name", "total_true_severe",
            "pred_healthy", "pred_early", "pred_mid", "pred_severe", "severe_recall",
        ])
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Severe-Stage Confusion Analysis",
        "",
        "Rows are true Stage 3 severe samples. Columns show predicted stage counts.",
        "",
        "| Run | True severe count | Pred healthy | Pred early | Pred mid | Pred severe | Severe recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['total_true_severe']} | {row['pred_healthy']} | {row['pred_early']} | {row['pred_mid']} | {row['pred_severe']} | {row['severe_recall']:.6f} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()

