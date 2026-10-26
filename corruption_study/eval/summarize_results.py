"""Summarize corruption-study metrics JSON files.

This reads saved metrics only. It does not run inference or training.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


E2_MODELS = {"e2_canonical", "e2_8", "e2_128_25"}
REGENERATED_CONDITIONS = {"clean_regen", "image_corrupted_only", "iot_corrupted_only", "both_corrupted"}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize corruption study metrics")
    parser.add_argument("--results-dir", default="corruption_study/results")
    return parser.parse_args()


def load_metrics(results_dir):
    rows = []
    for path in sorted(Path(results_dir).glob("*__metrics.json")):
        with open(path, "r", encoding="utf-8") as f:
            row = json.load(f)
        row["metrics_file"] = str(path)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No metrics JSON files found in {results_dir}")
    return pd.DataFrame(rows)


def add_clean_references(df):
    clean = df[df["condition"] == "clean"].drop_duplicates("model").set_index("model")
    clean_regen = df[df["condition"] == "clean_regen"].drop_duplicates("model").set_index("model")
    clean_acc = clean["accuracy"].to_dict()
    clean_f1 = clean["macro_f1"].to_dict()
    clean_regen_acc = clean_regen["accuracy"].to_dict()
    clean_regen_f1 = clean_regen["macro_f1"].to_dict()

    def reference(row, metric):
        model = row["model"]
        condition = row["condition"]
        use_regen = model in E2_MODELS and condition in REGENERATED_CONDITIONS and model in clean_regen_acc
        if metric == "accuracy":
            return clean_regen_acc[model] if use_regen else clean_acc.get(model)
        return clean_regen_f1[model] if use_regen else clean_f1.get(model)

    df["clean_reference_condition"] = df.apply(
        lambda row: "clean_regen"
        if row["model"] in E2_MODELS and row["condition"] in REGENERATED_CONDITIONS and row["model"] in clean_regen_acc
        else "clean",
        axis=1,
    )
    df["clean_accuracy"] = df.apply(lambda row: reference(row, "accuracy"), axis=1)
    df["clean_macro_f1"] = df.apply(lambda row: reference(row, "macro_f1"), axis=1)
    df["accuracy_drop"] = df["clean_accuracy"] - df["accuracy"]
    df["macro_f1_drop"] = df["clean_macro_f1"] - df["macro_f1"]
    df["relative_macro_f1_drop_percent"] = 100.0 * df["macro_f1_drop"] / df["clean_macro_f1"]
    return df


def make_robustness_summary(df):
    corrupted = df[df["severity"].astype(int) > 0].copy()
    group_cols = [
        "model",
        "condition",
        "image_corruption_type",
        "iot_corruption_type",
        "corruption_type",
        "severity",
    ]
    available_group_cols = [c for c in group_cols if c in corrupted.columns]
    summary = (
        corrupted
        .groupby(available_group_cols, dropna=False)
        .agg(
            n_runs=("macro_f1", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            accuracy_ci_low_mean=("accuracy_ci_low", "mean") if "accuracy_ci_low" in corrupted.columns else ("accuracy", "mean"),
            accuracy_ci_high_mean=("accuracy_ci_high", "mean") if "accuracy_ci_high" in corrupted.columns else ("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_ci_low_mean=("macro_f1_ci_low", "mean") if "macro_f1_ci_low" in corrupted.columns else ("macro_f1", "mean"),
            macro_f1_ci_high_mean=("macro_f1_ci_high", "mean") if "macro_f1_ci_high" in corrupted.columns else ("macro_f1", "mean"),
            accuracy_drop_mean=("accuracy_drop", "mean"),
            accuracy_drop_std=("accuracy_drop", "std"),
            accuracy_drop_ci_low_mean=("accuracy_drop_ci_low", "mean") if "accuracy_drop_ci_low" in corrupted.columns else ("accuracy_drop", "mean"),
            accuracy_drop_ci_high_mean=("accuracy_drop_ci_high", "mean") if "accuracy_drop_ci_high" in corrupted.columns else ("accuracy_drop", "mean"),
            macro_f1_drop_mean=("macro_f1_drop", "mean"),
            macro_f1_drop_std=("macro_f1_drop", "std"),
            macro_f1_drop_ci_low_mean=("macro_f1_drop_ci_low", "mean") if "macro_f1_drop_ci_low" in corrupted.columns else ("macro_f1_drop", "mean"),
            macro_f1_drop_ci_high_mean=("macro_f1_drop_ci_high", "mean") if "macro_f1_drop_ci_high" in corrupted.columns else ("macro_f1_drop", "mean"),
        )
        .reset_index()
    )
    return summary


def write_markdown(summary, path):
    cols = [
        "model",
        "condition",
        "corruption_type",
        "severity",
        "n_runs",
        "macro_f1_mean",
        "macro_f1_drop_mean",
        "accuracy_mean",
        "accuracy_drop_mean",
    ]
    cols = [c for c in cols if c in summary.columns]
    out = summary[cols].copy()
    for col in out.columns:
        if out[col].dtype.kind in "fc":
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Robustness Summary\n\n")
        f.write("Macro-F1 drop is computed from each model's own severity-0 clean Macro-F1.\n\n")
        f.write(out.to_markdown(index=False))
        f.write("\n")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    df = add_clean_references(load_metrics(results_dir))
    all_metrics_path = results_dir / "all_metrics.csv"
    summary_path = results_dir / "robustness_summary.csv"
    md_path = results_dir / "robustness_summary.md"
    df.to_csv(all_metrics_path, index=False)
    summary = make_robustness_summary(df)
    summary.to_csv(summary_path, index=False)
    write_markdown(summary, md_path)
    print(f"wrote {all_metrics_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
