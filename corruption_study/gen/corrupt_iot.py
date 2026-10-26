"""Materialize feature-space corrupted TEST IoT feature CSVs.

This script corrupts extracted continuous E1 feature columns. It does not train models.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


META_COLS = ["image", "label", "crop", "disease", "stage", "n_rows"]
DEFAULT_CORRUPTIONS = ["sensor_noise", "dropout", "drift"]
STOCHASTIC = {"sensor_noise", "dropout"}
SCHEDULES = {
    "sensor_noise": [0.1, 0.2, 0.3, 0.5, 1.0],
    "dropout": [0.05, 0.10, 0.20, 0.30, 0.40],
    "drift": [0.25, 0.5, 1.0, 1.5, 2.0],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate corrupted TEST IoT feature CSVs")
    parser.add_argument("--train-features", required=True)
    parser.add_argument("--test-features", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--corruptions", nargs="+", default=DEFAULT_CORRUPTIONS)
    parser.add_argument("--severities", nargs="+", type=int, default=[2, 3, 5])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def align_features(features, manifest, name):
    selected = features.set_index("image").loc[manifest["image"].astype(str).tolist()].reset_index()
    expected = manifest["label"].astype(str).tolist()
    actual = selected["label"].astype(str).tolist()
    if actual != expected:
        raise RuntimeError(f"{name} labels do not match manifest order")
    return selected


def feature_columns(df):
    return [
        col for col in df.columns
        if col not in META_COLS and pd.api.types.is_numeric_dtype(df[col])
    ]


def corrupt(df, cols, means, stds, corruption, severity, seed):
    rng = np.random.default_rng(seed)
    out = df.copy()
    values = out[cols].to_numpy(dtype=np.float64)
    scale = stds.to_numpy(dtype=np.float64)
    scale = np.where(scale <= 0, 1.0, scale)

    if corruption == "sensor_noise":
        factor = SCHEDULES[corruption][severity - 1]
        values = values + rng.normal(0.0, factor * scale, size=values.shape)
    elif corruption == "dropout":
        frac = SCHEDULES[corruption][severity - 1]
        mask = rng.random(values.shape) < frac
        mean_values = means.to_numpy(dtype=np.float64)
        values[mask] = np.broadcast_to(mean_values, values.shape)[mask]
    elif corruption == "drift":
        factor = SCHEDULES[corruption][severity - 1]
        signs = rng.choice(np.array([-1.0, 1.0]), size=(len(cols),))
        values = values + factor * scale * signs
    else:
        raise ValueError(f"Unsupported corruption: {corruption}")

    out.loc[:, cols] = values
    return out


def output_dir(root, corruption, severity, seed):
    base = Path(root) / corruption / f"sev_{severity}"
    if corruption in STOCHASTIC:
        base = base / f"cseed_{seed}"
    return base


def main():
    args = parse_args()
    train_features = pd.read_csv(args.train_features, dtype={"image": str, "label": str})
    test_features = pd.read_csv(args.test_features, dtype={"image": str, "label": str})
    train_manifest = pd.read_csv(args.train_manifest, dtype={"image": str, "label": str})
    test_manifest = pd.read_csv(args.test_manifest, dtype={"image": str, "label": str})

    train_df = align_features(train_features, train_manifest, "train")
    test_df = align_features(test_features, test_manifest, "test")
    cols = feature_columns(train_df)
    means = train_df[cols].mean(axis=0)
    stds = train_df[cols].std(axis=0).fillna(0.0)

    results_dir = Path("corruption_study") / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "corrupted_iot_columns.json", "w", encoding="utf-8") as f:
        json.dump({"corrupted_columns": cols, "left_untouched": [c for c in train_df.columns if c not in cols]}, f, indent=2)

    for corruption in args.corruptions:
        if corruption not in SCHEDULES:
            raise ValueError(f"Unsupported corruption: {corruption}")
        seeds = args.seeds if corruption in STOCHASTIC else [args.seeds[0]]
        for severity in args.severities:
            if severity < 1 or severity > 5:
                raise ValueError("Severity must be in 1..5")
            for seed in seeds:
                out_dir = output_dir(args.output_root, corruption, severity, seed)
                out_path = out_dir / "test_iot_features.csv"
                if out_path.exists() and not args.overwrite:
                    print(f"skip existing {out_path}")
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                corrupted = corrupt(test_df, cols, means, stds, corruption, severity, seed)
                corrupted.to_csv(out_path, index=False)
                print(f"{corruption} sev_{severity} seed={seed}: wrote {out_path}")


if __name__ == "__main__":
    main()
