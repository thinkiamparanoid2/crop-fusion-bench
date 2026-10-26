"""Generate corruption robustness plots from summary CSVs.

Reads saved metrics only. Does not run inference or training.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CORE_IMAGE_MODELS = ["e0", "e2_canonical", "e2_8"]
CORE_IOT_MODELS = ["e1", "e2_canonical", "e2_8"]
PAIRED_MODELS = ["e2_canonical", "e2_8", "e2_128_25"]
MODEL_LABELS = {
    "e0": "E0 image-only",
    "e1": "E1 IoT-only",
    "e2_canonical": "E2 canonical",
    "e2_8": "E2-8",
    "e2_128_25": "E2 128-to-25 (exploratory)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate corruption robustness plots")
    parser.add_argument("--results-dir", default="corruption_study/results")
    return parser.parse_args()


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def mean_by_seed(df):
    group_cols = ["model", "condition", "corruption_type", "severity"]
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            macro_f1=("macro_f1", "mean"),
            macro_f1_drop=("macro_f1_drop", "mean"),
            accuracy=("accuracy", "mean"),
            accuracy_drop=("accuracy_drop", "mean"),
            n_runs=("macro_f1", "count"),
        )
        .reset_index()
    )


def clean_reference_rows(df, models):
    rows = []
    for model in models:
        condition = "clean_regen" if model.startswith("e2_") else "clean"
        ref = df[(df["model"] == model) & (df["condition"] == condition)]
        if ref.empty:
            continue
        row = ref.iloc[0].copy()
        row["severity"] = 0
        row["corruption_type"] = "clean"
        row["macro_f1_drop"] = 0.0
        row["accuracy_drop"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_curves(df, models, condition, corruption_type, out_dir, prefix):
    subset = df[
        (df["condition"] == condition)
        & (df["corruption_type"] == corruption_type)
        & (df["model"].isin(models))
    ].copy()
    clean = clean_reference_rows(df, models)
    if clean.empty and subset.empty:
        return []

    plot_df = pd.concat([clean, subset], ignore_index=True)
    plot_df = mean_by_seed(plot_df)
    written = []

    for metric, ylabel, suffix in [
        ("macro_f1", "Macro-F1", "macro_f1"),
        ("macro_f1_drop", "Macro-F1 drop", "macro_f1_drop"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model in models:
            model_df = plot_df[plot_df["model"] == model].sort_values("severity")
            if model_df.empty:
                continue
            ax.plot(
                model_df["severity"],
                model_df[metric],
                marker="o",
                linewidth=2,
                label=MODEL_LABELS.get(model, model),
            )
        ax.set_title(f"{corruption_type} - {ylabel}")
        ax.set_xlabel("Severity")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 2, 3, 5])
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_path = out_dir / f"{prefix}__{corruption_type}__{suffix}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        written.append(out_path)
    return written


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    df = pd.read_csv(results_dir / "all_metrics.csv")
    out_dir = results_dir / "plots"
    ensure_dir(out_dir)

    written = []
    image_corrs = sorted(
        df[df["condition"] == "image_corrupted_only"]["corruption_type"].dropna().unique()
    )
    for corr in image_corrs:
        written.extend(plot_curves(df, CORE_IMAGE_MODELS, "image_corrupted_only", corr, out_dir, "image"))

    iot_corrs = sorted(
        df[df["condition"] == "iot_corrupted_only"]["corruption_type"].dropna().unique()
    )
    for corr in iot_corrs:
        written.extend(plot_curves(df, CORE_IOT_MODELS, "iot_corrupted_only", corr, out_dir, "iot"))

    both_corrs = sorted(
        df[df["condition"] == "both_corrupted"]["corruption_type"].dropna().unique()
    )
    for corr in both_corrs:
        written.extend(plot_curves(df, PAIRED_MODELS, "both_corrupted", corr, out_dir, "both"))

    manifest = pd.DataFrame({"plot_path": [str(path) for path in written]})
    manifest_path = out_dir / "plot_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"wrote {len(written)} plots")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
