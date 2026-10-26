"""Bootstrap direct difference-of-drops comparisons.

This computes CIs for "unimodal drop - multimodal drop" using saved prediction
CSVs only. Positive values mean the multimodal model degraded less.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


E2_MODELS = ["e2_canonical", "e2_8", "e2_128_25"]


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap difference-of-drops")
    parser.add_argument("--results-dir", default="corruption_study/results")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260610)
    return parser.parse_args()


def prediction_path(results_dir, run_name):
    return results_dir / "predictions" / f"{run_name}__predictions.csv"


def load_prediction(results_dir, run_name):
    path = prediction_path(results_dir, run_name)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={"sample_id": str})[["sample_id", "y_true", "y_pred"]].copy()


def present_class_macro_f1(y_true, y_pred):
    labels = np.unique(y_true)
    values = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        denom = (2 * tp) + fp + fn
        values.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(values)) if values else 0.0


def merge_predictions(frames):
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="sample_id", validate="one_to_one")
    y_cols = [col for col in merged.columns if col.startswith("y_true")]
    first = merged[y_cols[0]].to_numpy()
    for col in y_cols[1:]:
        if not np.array_equal(first, merged[col].to_numpy()):
            raise RuntimeError(f"y_true mismatch after sample_id alignment: {col}")
    return merged


def run_name(row):
    parts = [row["model"], row["condition"]]
    if pd.notna(row.get("image_corruption_type")) and row.get("image_corruption_type") != "":
        parts.append(str(row["image_corruption_type"]))
    if pd.notna(row.get("iot_corruption_type")) and row.get("iot_corruption_type") != "":
        parts.append(str(row["iot_corruption_type"]))
    parts.append(f"sev{int(row['severity'])}")
    if pd.notna(row.get("corruption_seed")) and row.get("corruption_seed") != "":
        parts.append(f"cseed{int(float(row['corruption_seed']))}")
    return "__".join(parts)


def bootstrap_comparison(results_dir, uni_clean, uni_run, mm_clean, mm_run, n_bootstrap, rng):
    uni_clean_df = load_prediction(results_dir, uni_clean).rename(columns={"y_true": "y_true_uni_clean", "y_pred": "pred_uni_clean"})
    uni_run_df = load_prediction(results_dir, uni_run).rename(columns={"y_true": "y_true_uni_run", "y_pred": "pred_uni_run"})
    mm_clean_df = load_prediction(results_dir, mm_clean).rename(columns={"y_true": "y_true_mm_clean", "y_pred": "pred_mm_clean"})
    mm_run_df = load_prediction(results_dir, mm_run).rename(columns={"y_true": "y_true_mm_run", "y_pred": "pred_mm_run"})

    merged = merge_predictions([uni_clean_df, uni_run_df, mm_clean_df, mm_run_df])
    y = merged["y_true_uni_clean"].to_numpy(dtype=np.int64)
    pred_uni_clean = merged["pred_uni_clean"].to_numpy(dtype=np.int64)
    pred_uni_run = merged["pred_uni_run"].to_numpy(dtype=np.int64)
    pred_mm_clean = merged["pred_mm_clean"].to_numpy(dtype=np.int64)
    pred_mm_run = merged["pred_mm_run"].to_numpy(dtype=np.int64)

    uni_drop_point = present_class_macro_f1(y, pred_uni_clean) - present_class_macro_f1(y, pred_uni_run)
    mm_drop_point = present_class_macro_f1(y, pred_mm_clean) - present_class_macro_f1(y, pred_mm_run)
    diff_point = uni_drop_point - mm_drop_point

    n = len(merged)
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y[idx]
        uni_drop = present_class_macro_f1(yt, pred_uni_clean[idx]) - present_class_macro_f1(yt, pred_uni_run[idx])
        mm_drop = present_class_macro_f1(yt, pred_mm_clean[idx]) - present_class_macro_f1(yt, pred_mm_run[idx])
        diffs[i] = uni_drop - mm_drop

    return {
        "unimodal_macro_f1_drop": uni_drop_point,
        "multimodal_macro_f1_drop": mm_drop_point,
        "difference_of_drops": diff_point,
        "difference_of_drops_ci_low": float(np.percentile(diffs, 2.5)),
        "difference_of_drops_ci_high": float(np.percentile(diffs, 97.5)),
    }


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    metrics = pd.read_csv(results_dir / "all_metrics.csv")
    rng = np.random.default_rng(args.seed)
    rows = []

    image_runs = metrics[
        (metrics["model"] == "e0")
        & (metrics["condition"] == "image_corrupted_only")
        & (metrics["severity"].astype(int) > 0)
    ].copy()
    for _, uni in image_runs.iterrows():
        for model in E2_MODELS:
            mm_match = metrics[
                (metrics["model"] == model)
                & (metrics["condition"] == "image_corrupted_only")
                & (metrics["corruption_type"] == uni["corruption_type"])
                & (metrics["severity"].astype(int) == int(uni["severity"]))
                & (metrics["corruption_seed"].fillna("").astype(str) == ("" if pd.isna(uni["corruption_seed"]) else str(uni["corruption_seed"])))
            ]
            if mm_match.empty:
                continue
            mm = mm_match.iloc[0]
            result = bootstrap_comparison(
                results_dir,
                "e0__clean__sev0",
                run_name(uni),
                f"{model}__clean_regen__sev0",
                run_name(mm),
                args.n_bootstrap,
                rng,
            )
            result.update({
                "comparison": "image",
                "corruption_type": uni["corruption_type"],
                "severity": int(uni["severity"]),
                "corruption_seed": uni["corruption_seed"],
                "unimodal_model": "e0",
                "multimodal_model": model,
                "exploratory": model == "e2_128_25",
            })
            rows.append(result)

    iot_runs = metrics[
        (metrics["model"] == "e1")
        & (metrics["condition"] == "iot_corrupted_only")
        & (metrics["severity"].astype(int) > 0)
    ].copy()
    for _, uni in iot_runs.iterrows():
        for model in E2_MODELS:
            mm_match = metrics[
                (metrics["model"] == model)
                & (metrics["condition"] == "iot_corrupted_only")
                & (metrics["corruption_type"] == uni["corruption_type"])
                & (metrics["severity"].astype(int) == int(uni["severity"]))
                & (metrics["corruption_seed"].fillna("").astype(str) == ("" if pd.isna(uni["corruption_seed"]) else str(uni["corruption_seed"])))
            ]
            if mm_match.empty:
                continue
            mm = mm_match.iloc[0]
            result = bootstrap_comparison(
                results_dir,
                "e1__clean__sev0",
                run_name(uni),
                f"{model}__clean_regen__sev0",
                run_name(mm),
                args.n_bootstrap,
                rng,
            )
            result.update({
                "comparison": "iot",
                "corruption_type": uni["corruption_type"],
                "severity": int(uni["severity"]),
                "corruption_seed": uni["corruption_seed"],
                "unimodal_model": "e1",
                "multimodal_model": model,
                "exploratory": model == "e2_128_25",
            })
            rows.append(result)

    out = pd.DataFrame(rows)
    out_path = results_dir / "difference_of_drops_bootstrap.csv"
    out.to_csv(out_path, index=False)

    summary = (
        out.groupby(["comparison", "corruption_type", "unimodal_model", "multimodal_model", "exploratory"], dropna=False)
        .agg(
            n_runs=("difference_of_drops", "count"),
            difference_of_drops_mean=("difference_of_drops", "mean"),
            difference_of_drops_ci_low_mean=("difference_of_drops_ci_low", "mean"),
            difference_of_drops_ci_high_mean=("difference_of_drops_ci_high", "mean"),
            unimodal_macro_f1_drop_mean=("unimodal_macro_f1_drop", "mean"),
            multimodal_macro_f1_drop_mean=("multimodal_macro_f1_drop", "mean"),
        )
        .reset_index()
    )
    summary_path = results_dir / "difference_of_drops_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {out_path} rows={len(out)}")
    print(f"wrote {summary_path} rows={len(summary)}")


if __name__ == "__main__":
    main()
