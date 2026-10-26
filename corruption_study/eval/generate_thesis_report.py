"""Generate thesis-style corruption robustness report artifacts.

Reads enriched metrics CSVs only. Does not run inference or training.
"""

import argparse
from pathlib import Path

import pandas as pd


CORE_MODELS = ["e0", "e1", "e2_canonical", "e2_8"]
E2_MODELS = ["e2_canonical", "e2_8"]
E2_VARIANTS = ["e2_canonical", "e2_8", "e2_128_25"]
EXPLORATORY = {"e2_128_25"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate thesis-style robustness report")
    parser.add_argument("--results-dir", default="corruption_study/results")
    parser.add_argument("--output", default="corruption_study/results/THESIS_STYLE_ROBUSTNESS_REPORT.md")
    return parser.parse_args()


def fmt(value):
    if pd.isna(value):
        return ""
    return f"{float(value):.6f}"


def model_label(model):
    return f"{model} (exploratory)" if model in EXPLORATORY else model


def write_table(f, df, cols, sort_cols=None):
    out = df.copy()
    if sort_cols:
        out = out.sort_values(sort_cols)
    out = out[cols]
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(fmt)
    f.write(out.to_markdown(index=False))
    f.write("\n\n")


def build_curve_data(df):
    clean_rows = []
    for model in sorted(df["model"].dropna().unique()):
        ref = df[(df["model"] == model) & (df["condition"] == "clean")]
        if ref.empty:
            continue
        if model.startswith("e2_"):
            regen = df[(df["model"] == model) & (df["condition"] == "clean_regen")]
            if not regen.empty:
                ref = regen
        row = ref.iloc[0].copy()
        row["severity"] = 0
        row["corruption_type"] = "clean"
        row["image_corruption_type"] = None
        row["iot_corruption_type"] = None
        clean_rows.append(row)
    curve = pd.concat([pd.DataFrame(clean_rows), df[df["severity"].astype(int) > 0]], ignore_index=True)
    keep = [
        "model",
        "condition",
        "corruption_type",
        "image_corruption_type",
        "iot_corruption_type",
        "severity",
        "corruption_seed",
        "macro_f1",
        "macro_f1_ci_low",
        "macro_f1_ci_high",
        "macro_f1_drop",
        "macro_f1_drop_ci_low",
        "macro_f1_drop_ci_high",
        "accuracy",
        "accuracy_drop",
        "clean_reference_condition",
    ]
    return curve[[c for c in keep if c in curve.columns]]


def aggregate(df):
    group_cols = ["model", "condition", "corruption_type", "severity"]
    return (
        df[df["severity"].astype(int) > 0]
        .groupby(group_cols, dropna=False)
        .agg(
            n_runs=("macro_f1", "count"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_drop_mean=("macro_f1_drop", "mean"),
            macro_f1_ci_low_mean=("macro_f1_ci_low", "mean"),
            macro_f1_ci_high_mean=("macro_f1_ci_high", "mean"),
            macro_f1_drop_ci_low_mean=("macro_f1_drop_ci_low", "mean"),
            macro_f1_drop_ci_high_mean=("macro_f1_drop_ci_high", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_drop_mean=("accuracy_drop", "mean"),
            accuracy_ci_low_mean=("accuracy_ci_low", "mean"),
            accuracy_ci_high_mean=("accuracy_ci_high", "mean"),
            accuracy_drop_ci_low_mean=("accuracy_drop_ci_low", "mean"),
            accuracy_drop_ci_high_mean=("accuracy_drop_ci_high", "mean"),
        )
        .reset_index()
    )


def corruption_mean_drop(df, model, condition):
    subset = df[(df["model"] == model) & (df["condition"] == condition) & (df["severity"].astype(int) > 0)]
    return (
        subset.groupby("corruption_type", dropna=False)
        .agg(
            n_runs=("macro_f1_drop", "count"),
            mean_macro_f1_drop=("macro_f1_drop", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            mean_accuracy=("accuracy", "mean"),
            mean_accuracy_drop=("accuracy_drop", "mean"),
            mean_macro_precision=("macro_precision", "mean"),
            mean_weighted_f1=("weighted_f1", "mean"),
            mean_macro_f1_drop_ci_low=("macro_f1_drop_ci_low", "mean"),
            mean_macro_f1_drop_ci_high=("macro_f1_drop_ci_high", "mean"),
            mean_accuracy_drop_ci_low=("accuracy_drop_ci_low", "mean"),
            mean_accuracy_drop_ci_high=("accuracy_drop_ci_high", "mean"),
        )
        .reset_index()
        .sort_values("mean_macro_f1_drop", ascending=False)
    )


def both_corrupted_table(summary):
    return summary[summary["condition"] == "both_corrupted"].copy()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    df = pd.read_csv(results_dir / "all_metrics.csv")
    summary = aggregate(df)

    curve = build_curve_data(df)
    curve_path = results_dir / "degradation_curve_data.csv"
    drop_path = results_dir / "drop_summary_with_ci.csv"
    curve.to_csv(curve_path, index=False)
    summary.to_csv(drop_path, index=False)

    image_core = pd.concat(
        [
            corruption_mean_drop(df, "e0", "image_corrupted_only").assign(comparison_model="e0"),
            corruption_mean_drop(df, "e2_canonical", "image_corrupted_only").assign(comparison_model="e2_canonical"),
            corruption_mean_drop(df, "e2_8", "image_corrupted_only").assign(comparison_model="e2_8"),
        ],
        ignore_index=True,
    )
    iot_core = pd.concat(
        [
            corruption_mean_drop(df, "e1", "iot_corrupted_only").assign(comparison_model="e1"),
            corruption_mean_drop(df, "e2_canonical", "iot_corrupted_only").assign(comparison_model="e2_canonical"),
            corruption_mean_drop(df, "e2_8", "iot_corrupted_only").assign(comparison_model="e2_8"),
        ],
        ignore_index=True,
    )

    both = both_corrupted_table(summary)
    both["model"] = both["model"].map(model_label)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write("# Thesis-Style Corruption Robustness Report\n\n")
        f.write("This report is generated from saved metrics and predictions only. No model was retrained or re-evaluated while generating this report.\n\n")
        f.write("Primary framing:\n\n")
        f.write("- E0 vs E2-canonical/E2-8 under image corruption.\n")
        f.write("- E1 vs E2-canonical/E2-8 under IoT feature-space corruption.\n")
        f.write("- E2-canonical vs E2-8 under representative paired stress conditions.\n")
        f.write("- `e2_128_25` is included only as an exploratory classical variant.\n\n")

        f.write("## Bootstrap Settings\n\n")
        f.write("Confidence intervals use 1000 bootstrap resamples, 95% percentile intervals, and fixed seed `20260609`. ")
        f.write("Drop intervals use paired bootstrap aligned by `sample_id`. Macro-F1 is computed over classes present in each resample.\n\n")

        f.write("## Image Corruption Degradation\n\n")
        f.write("Mean metrics across severities 2, 3, and 5; stochastic corruptions average across seeds. ")
        f.write("Macro-F1 drop remains the primary robustness metric; accuracy drop, weighted F1, and macro precision are supporting metrics.\n\n")
        write_table(
            f,
            image_core,
            [
                "comparison_model",
                "corruption_type",
                "n_runs",
                "mean_macro_f1",
                "mean_macro_f1_drop",
                "mean_macro_f1_drop_ci_low",
                "mean_macro_f1_drop_ci_high",
                "mean_accuracy",
                "mean_accuracy_drop",
                "mean_accuracy_drop_ci_low",
                "mean_accuracy_drop_ci_high",
                "mean_weighted_f1",
                "mean_macro_precision",
            ],
            ["corruption_type", "comparison_model"],
        )

        f.write("## IoT Corruption Degradation\n\n")
        f.write("Mean metrics across severities 2, 3, and 5; stochastic corruptions average across seeds.\n\n")
        write_table(
            f,
            iot_core,
            [
                "comparison_model",
                "corruption_type",
                "n_runs",
                "mean_macro_f1",
                "mean_macro_f1_drop",
                "mean_macro_f1_drop_ci_low",
                "mean_macro_f1_drop_ci_high",
                "mean_accuracy",
                "mean_accuracy_drop",
                "mean_accuracy_drop_ci_low",
                "mean_accuracy_drop_ci_high",
                "mean_weighted_f1",
                "mean_macro_precision",
            ],
            ["corruption_type", "comparison_model"],
        )

        f.write("## Representative Paired Stress Conditions\n\n")
        f.write("Paired conditions use matched severity index only; this does not imply calibrated-equivalent damage across modalities.\n\n")
        f.write("Two representative paired stress conditions were run:\n\n")
        f.write("- Canonical paired stress: `gaussian_noise + sensor_noise`.\n")
        f.write("- Data-driven worst-case distinct paired stress: `defocus_blur + drift`.\n\n")
        f.write("The worst-case distinct pair was selected from existing single-modality results: ")
        f.write("E0's largest mean image-corruption Macro-F1 drop was `defocus_blur` (0.903679), ")
        f.write("and E1's largest mean IoT-corruption Macro-F1 drop was `drift` (0.821596). ")
        f.write("Because this pair is distinct from the canonical pair, no fallback to a second-largest corruption was needed.\n\n")
        write_table(
            f,
            both,
            [
                "model",
                "corruption_type",
                "severity",
                "n_runs",
                "macro_f1_mean",
                "macro_f1_drop_mean",
                "macro_f1_drop_ci_low_mean",
                "macro_f1_drop_ci_high_mean",
                "accuracy_mean",
                "accuracy_drop_mean",
                "accuracy_drop_ci_low_mean",
                "accuracy_drop_ci_high_mean",
            ],
            ["corruption_type", "severity", "model"],
        )

        diff_path = results_dir / "difference_of_drops_summary.csv"
        if diff_path.exists():
            diff = pd.read_csv(diff_path)
            diff["multimodal_model"] = diff["multimodal_model"].map(model_label)
            f.write("## Direct Difference-of-Drops Bootstrap\n\n")
            f.write("Positive difference-of-drops values mean the multimodal model degraded less than the matching unimodal baseline.\n\n")
            write_table(
                f,
                diff,
                [
                    "comparison",
                    "corruption_type",
                    "unimodal_model",
                    "multimodal_model",
                    "n_runs",
                    "unimodal_macro_f1_drop_mean",
                    "multimodal_macro_f1_drop_mean",
                    "difference_of_drops_mean",
                    "difference_of_drops_ci_low_mean",
                    "difference_of_drops_ci_high_mean",
                ],
                ["comparison", "corruption_type", "multimodal_model"],
            )

        f.write("## Output Tables\n\n")
        f.write(f"- Degradation curve data: `{curve_path}`\n")
        f.write(f"- Drop summary with CIs: `{drop_path}`\n")
        f.write("- Full enriched run table: `corruption_study/results/all_metrics.csv`\n")
        f.write("- Per-class metrics: `corruption_study/results/per_class_recall_by_condition.csv`\n\n")

        f.write("## Interpretation Guardrail\n\n")
        f.write("The main robustness evidence is Macro-F1 degradation from the correct clean reference, not clean accuracy alone. ")
        f.write("For regenerated-embedding E2 variants, clean reference is `clean_regen`; for E0/E1 it is severity-0 `clean`.\n")

    print(f"wrote {curve_path}")
    print(f"wrote {drop_path}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
