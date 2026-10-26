"""Create a stratified train/validation split and optionally copy files.

Usage:
  python split_train_val.py --labels Data/train.csv --data-dir Data --out-dir e0_image_only/legacy/splits --val-size 0.1 --seed 42 --copy

The script will write `train.csv` and `val.csv` in `--out-dir` and (if `--copy`) copy
matching folders/files from the configured train directory into `--out-dir`/train and /val.
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import DATA_ROOT, TRAIN_CSV

LEGACY_SPLIT_DIR = REPO_ROOT / "legacy" / "splits"

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from sklearn.model_selection import StratifiedShuffleSplit
except Exception:
    StratifiedShuffleSplit = None

LABEL_CANDIDATES = ["label", "labels", "class", "target", "category"]
ID_CANDIDATES = ["id", "image_id", "image", "filename", "file"]


def read_csv(path):
    if pd:
        return pd.read_csv(path)
    with open(path, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        rows = list(r)
    return rows


def write_csv(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if pd and hasattr(df, 'to_csv'):
        df.to_csv(path, index=False)
        return
    if not df:
        with open(path, 'w', newline='', encoding='utf-8') as f:
            f.write('')
        return
    cols = list(df[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(df)


def detect_columns(df):
    cols = df.columns.tolist() if pd and hasattr(df, 'columns') else (list(df[0].keys()) if df else [])
    label = next((c for c in LABEL_CANDIDATES if c in cols), None)
    idx = next((c for c in ID_CANDIDATES if c in cols), None)
    if label is None:
        label = cols[-1] if cols else None
    if idx is None:
        idx = cols[0] if cols else None
    return idx, label


def stratified_split(df, id_col, label_col, val_size, seed):
    if pd and StratifiedShuffleSplit is not None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        X = df
        y = df[label_col]
        for train_idx, val_idx in sss.split(X, y):
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            return train_df, val_df

    if pd:
        groups = df.groupby(label_col)
        train_parts = []
        val_parts = []
        for name, g in groups:
            items = g.sample(frac=1, random_state=seed).reset_index(drop=True)
            n_val = int(round(len(items) * val_size))
            if n_val == 0 and len(items) > 1 and val_size > 0:
                n_val = 1
            val_parts.append(items.iloc[:n_val])
            train_parts.append(items.iloc[n_val:])
        train_df = pd.concat(train_parts, ignore_index=True)
        val_df = pd.concat(val_parts, ignore_index=True)
        return train_df, val_df

    rng = random.Random(seed)
    by_label = {}
    for r in df:
        key = r[label_col]
        by_label.setdefault(key, []).append(r)
    train_list = []
    val_list = []
    for key, items in by_label.items():
        rng.shuffle(items)
        n_val = int(round(len(items) * val_size))
        if n_val == 0 and len(items) > 1 and val_size > 0:
            n_val = 1
        val_list.extend(items[:n_val])
        train_list.extend(items[n_val:])
    return train_list, val_list


def copy_item(src_base, dst_base, item_name):
    src_base = Path(src_base)
    dst_base = Path(dst_base)
    src = src_base / str(item_name)
    if src.exists() and src.is_dir():
        dst = dst_base / str(item_name)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return True
    for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff'):
        f = src_base / (str(item_name) + ext)
        if f.exists():
            dst_base.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst_base / f.name)
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default=str(DATA_ROOT))
    parser.add_argument('--labels', default=str(TRAIN_CSV))
    parser.add_argument('--out-dir', default=str(LEGACY_SPLIT_DIR))
    parser.add_argument('--val-size', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--copy', action='store_true')
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise SystemExit(f'Labels file not found: {labels_path}')

    df = read_csv(labels_path)
    id_col, label_col = detect_columns(df)
    if id_col is None or label_col is None:
        raise SystemExit('Could not detect id or label columns in labels CSV')

    train_df, val_df = stratified_split(df, id_col, label_col, args.val_size, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(train_df, out_dir / 'train.csv')
    write_csv(val_df, out_dir / 'val.csv')

    print(f'Wrote train: {(out_dir / "train.csv")} rows={len(train_df)}')
    print(f'Wrote val: {(out_dir / "val.csv")} rows={len(val_df)}')

    if args.copy:
        src = Path(args.data_dir) / 'train'
        if not src.exists():
            print(f'Source train folder not found: {src}. Skipping copy.')
            return
        out_train = out_dir / 'train'
        out_val = out_dir / 'val'
        out_train.mkdir(parents=True, exist_ok=True)
        out_val.mkdir(parents=True, exist_ok=True)
        if pd and hasattr(train_df, 'columns'):
            train_ids = [str(x) for x in train_df[id_col].tolist()]
            val_ids = [str(x) for x in val_df[id_col].tolist()]
        else:
            train_ids = [str(r[id_col]) for r in train_df]
            val_ids = [str(r[id_col]) for r in val_df]

        copied = 0
        missing = 0
        for t in train_ids:
            if copy_item(src, out_train, t):
                copied += 1
            else:
                missing += 1
        for v in val_ids:
            if copy_item(src, out_val, v):
                copied += 1
            else:
                missing += 1
        print(f'Copied {copied} items, {missing} missing')


if __name__ == '__main__':
    main()
