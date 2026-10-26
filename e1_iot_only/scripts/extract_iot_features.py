import argparse
from pathlib import Path

import pandas as pd


E1_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = E1_ROOT.parent

BASE_DIR = PROJECT_ROOT / "Data"
OUTPUT_DIR = E1_ROOT / "features"

NUMERIC_COLS = [
    "int_temp_1_avg",
    "int_temp_1_max",
    "int_temp_1_min",
    "int_hum_1_avg",
    "int_hum_1_max",
    "int_hum_1_min",
    "int_dew_point_avg",
    "int_dew_point_max",
    "int_dew_point_min",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(BASE_DIR))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def clean_numeric_data(df):
    data = df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
    data = data.ffill().bfill()
    data = data.fillna(data.median()).fillna(0)
    return data


def make_feature_row(image_id, label, csv_path):
    df = pd.read_csv(csv_path)
    data = clean_numeric_data(df)

    crop, disease, stage = label.split("_")

    row = {
        "image": image_id,
        "label": label,
        "crop": crop,
        "disease": disease,
        "stage": int(stage),
        "n_rows": len(df),
    }

    for col in NUMERIC_COLS:
        values = data[col]
        row[col + "_mean"] = values.mean()
        row[col + "_std"] = values.std(ddof=0)
        row[col + "_min"] = values.min()
        row[col + "_max"] = values.max()
        row[col + "_median"] = values.median()
        row[col + "_last"] = values.iloc[-1] if len(values) else 0

    return row


def extract_split(base_dir, output_dir, split, limit):
    manifest_path = base_dir / (split + ".csv")
    manifest = pd.read_csv(manifest_path)

    if limit is not None:
        manifest = manifest.head(limit)

    rows = []
    total = len(manifest)
    print("[%s] extracting %d samples" % (split, total))

    for i, item in manifest.iterrows():
        image_id = str(item["image"])
        label = str(item["label"])
        csv_path = base_dir / split / image_id / (image_id + "_final.csv")

        rows.append(make_feature_row(image_id, label, csv_path))

        done = len(rows)
        if done == 1 or done % 5000 == 0 or done == total:
            print("[%s] %d/%d" % (split, done, total))

    features = pd.DataFrame(rows)

    if len(features) != total:
        raise RuntimeError("%s row count mismatch" % split)

    if features.isna().sum().sum() > 0:
        raise RuntimeError("%s features contain NaN values" % split)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (split + "_iot_features.csv")
    features.to_csv(output_path, index=False)

    print("[%s] saved %s with shape %s" % (split, output_path, features.shape))


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)

    extract_split(base_dir, output_dir, "train", args.limit)
    extract_split(base_dir, output_dir, "test", args.limit)


if __name__ == "__main__":
    main()
