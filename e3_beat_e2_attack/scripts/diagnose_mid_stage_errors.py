"""Diagnose where an E3 attack run loses to E2-matched-8 on saved predictions."""

import argparse

from pathlib import Path

import pandas as pd

ATTACK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ATTACK_ROOT.parent
SOURCE_RESULTS = PROJECT_ROOT / "e3_bottleneck_sweep" / "results"
OUT_DIR = ATTACK_ROOT / "results" / "diagnostics"
DOCS_ATTACK_DIR = PROJECT_ROOT / "docs" / "by_stage" / "06_e3_attack"


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose E2-vs-E3 prediction differences")
    parser.add_argument(
        "--e2-predictions",
        type=Path,
        default=SOURCE_RESULTS / "e2_matched_8_test_predictions.csv",
        help="Path to the E2-matched prediction CSV.",
    )
    parser.add_argument(
        "--e3-predictions",
        type=Path,
        default=SOURCE_RESULTS / "e3_8_vqc_test_predictions.csv",
        help="Path to the E3 prediction CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Directory for diagnostic CSV outputs.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DOCS_ATTACK_DIR / "MID_STAGE_ERROR_DIAGNOSTIC.md",
        help="Markdown report path.",
    )
    return parser.parse_args()


def stage_name(stage):
    return {"0": "healthy", "1": "early", "2": "mid", "3": "severe"}.get(str(stage), str(stage))


def display_path(path):
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    DOCS_ATTACK_DIR.mkdir(parents=True, exist_ok=True)
    e2 = pd.read_csv(args.e2_predictions)
    e3 = pd.read_csv(args.e3_predictions)

    key_cols = ["image", "label", "stage"]
    merged = e2[key_cols + ["pred_label", "pred_stage", "correct", "confidence"]].merge(
        e3[key_cols + ["pred_label", "pred_stage", "correct", "confidence"]],
        on=key_cols,
        suffixes=("_e2", "_e3"),
        validate="one_to_one",
    )
    merged["stage"] = merged["stage"].astype(str)
    merged["e2_correct_e3_wrong"] = merged["correct_e2"] & ~merged["correct_e3"]
    merged["e3_wrong"] = ~merged["correct_e3"]

    stage_summary = []
    for stage, sub in merged.groupby("stage", sort=True):
        stage_summary.append({
            "stage": stage,
            "stage_name": stage_name(stage),
            "n": len(sub),
            "e2_correct": int(sub["correct_e2"].sum()),
            "e3_correct": int(sub["correct_e3"].sum()),
            "e2_correct_e3_wrong": int(sub["e2_correct_e3_wrong"].sum()),
            "e2_accuracy": float(sub["correct_e2"].mean()),
            "e3_accuracy": float(sub["correct_e3"].mean()),
        })
    stage_summary_df = pd.DataFrame(stage_summary)

    mid_lost = merged[(merged["stage"] == "2") & merged["e2_correct_e3_wrong"]].copy()
    mid_lost.to_csv(args.out_dir / "mid_stage_e2_correct_e3_wrong.csv", index=False)
    stage_summary_df.to_csv(args.out_dir / "stage_error_summary.csv", index=False)

    e3_wrong_pairs = (
        merged[merged["e3_wrong"]]
        .groupby(["stage", "label", "pred_label_e3", "pred_stage_e3"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "stage", "label"], ascending=[False, True, True])
    )
    e3_wrong_pairs.to_csv(args.out_dir / "e3_wrong_confusion_pairs.csv", index=False)

    mid_stage_direction = (
        merged[(merged["stage"] == "2") & merged["e3_wrong"]]
        .groupby(["pred_stage_e3"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    mid_stage_direction["pred_stage_name"] = mid_stage_direction["pred_stage_e3"].map(stage_name)
    mid_stage_direction.to_csv(args.out_dir / "e3_mid_stage_error_direction.csv", index=False)

    lines = [
        "# E3 Mid-Stage Error Diagnostic",
        "",
        "Source predictions:",
        "",
        f"- `{display_path(args.e2_predictions)}`",
        f"- `{display_path(args.e3_predictions)}`",
        "",
        "## Stage Summary",
        "",
        "| Stage | Name | N | E2 correct | E3 correct | E2 correct / E3 wrong | E2 acc | E3 acc |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in stage_summary_df.iterrows():
        lines.append(
            f"| {row['stage']} | {row['stage_name']} | {row['n']} | {row['e2_correct']} | {row['e3_correct']} | "
            f"{row['e2_correct_e3_wrong']} | {row['e2_accuracy']:.6f} | {row['e3_accuracy']:.6f} |"
        )

    lines.extend([
        "",
        "## E3 Mid-Stage Error Direction",
        "",
        "| E3 predicted stage | Name | Count |",
        "|---|---|---:|",
    ])
    for _, row in mid_stage_direction.iterrows():
        lines.append(f"| {row['pred_stage_e3']} | {row['pred_stage_name']} | {row['count']} |")

    lines.extend([
        "",
        "## Top E3 Wrong Confusion Pairs",
        "",
        "| True stage | True label | E3 pred label | E3 pred stage | Count |",
        "|---|---|---|---|---:|",
    ])
    for _, row in e3_wrong_pairs.head(20).iterrows():
        lines.append(
            f"| {row['stage']} | {row['label']} | {row['pred_label_e3']} | {row['pred_stage_e3']} | {row['count']} |"
        )

    lines.extend([
        "",
        "## Files",
        "",
        "- `stage_error_summary.csv`",
        "- `mid_stage_e2_correct_e3_wrong.csv`",
        "- `e3_wrong_confusion_pairs.csv`",
        "- `e3_mid_stage_error_direction.csv`",
        "",
    ])
    report = args.report_path
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
