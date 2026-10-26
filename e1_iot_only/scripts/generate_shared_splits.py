import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


E1_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = E1_ROOT.parent

BASE_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = E1_ROOT / "generated_splits"
SEEDS = [13, 37, 73]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--val-size", type=float, default=0.2)
    return parser.parse_args()


def save_label_counts(df, output_path):
    counts = df["label"].value_counts().sort_index()
    counts.to_csv(output_path, header=["count"])


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = pd.read_csv(base_dir / "train.csv")
    test_manifest = pd.read_csv(base_dir / "test.csv")

    required_cols = ["image", "label"]
    train_manifest = train_manifest[required_cols].copy()
    test_manifest = test_manifest[required_cols].copy()

    test_manifest.to_csv(output_dir / "test.csv", index=False)
    save_label_counts(test_manifest, output_dir / "test_label_counts.csv")

    summary_rows = []

    for seed in SEEDS:
        train_split, val_split = train_test_split(
            train_manifest,
            test_size=args.val_size,
            random_state=seed,
            stratify=train_manifest["label"],
        )

        train_split = train_split.sort_values("image").reset_index(drop=True)
        val_split = val_split.sort_values("image").reset_index(drop=True)

        train_path = output_dir / ("seed_%d_train.csv" % seed)
        val_path = output_dir / ("seed_%d_val.csv" % seed)

        train_split.to_csv(train_path, index=False)
        val_split.to_csv(val_path, index=False)

        save_label_counts(train_split, output_dir / ("seed_%d_train_label_counts.csv" % seed))
        save_label_counts(val_split, output_dir / ("seed_%d_val_label_counts.csv" % seed))

        overlap = set(train_split["image"]).intersection(set(val_split["image"]))
        if overlap:
            raise RuntimeError("Seed %d has train/val overlap" % seed)

        summary_rows.append(
            {
                "seed": seed,
                "train_rows": len(train_split),
                "val_rows": len(val_split),
                "test_rows": len(test_manifest),
                "train_unique_labels": train_split["label"].nunique(),
                "val_unique_labels": val_split["label"].nunique(),
                "test_unique_labels": test_manifest["label"].nunique(),
            }
        )

        print(
            "seed %d: train=%d val=%d test=%d"
            % (seed, len(train_split), len(val_split), len(test_manifest))
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "split_summary.csv", index=False)
    print("saved splits to %s" % output_dir)


if __name__ == "__main__":
    main()
