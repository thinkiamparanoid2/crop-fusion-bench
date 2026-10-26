"""Check class distributions in split CSVs and print differences.
Usage: run this with the same Python used for the project virtualenv.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEGACY_SPLIT_DIR = REPO_ROOT / "legacy" / "splits"

LABEL_CANDIDATES = ["label", "labels", "class", "target", "category"]


def detect_label_field(fieldnames):
    for c in LABEL_CANDIDATES:
        if c in fieldnames:
            return c
    # fallback to last column
    return fieldnames[-1]


def read_counts(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    with open(p, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []
        if not fieldnames:
            return None, Counter()
        label_field = detect_label_field(fieldnames)
        counts = Counter()
        total = 0
        for row in r:
            key = row.get(label_field)
            counts[key] += 1
            total += 1
    return label_field, counts


def print_comparison(train_counts, val_counts):
    all_labels = sorted(set(train_counts.keys()) | set(val_counts.keys()))
    t_total = sum(train_counts.values())
    v_total = sum(val_counts.values())
    print(f"Train total: {t_total}, Val total: {v_total}\n")
    print(f"{'Label':30} {'Train':12} {'Train%':8} {'Val':12} {'Val%':8} {'AbsDiff':8} {'RelDiff%':8}")
    for lbl in all_labels:
        t = train_counts.get(lbl, 0)
        v = val_counts.get(lbl, 0)
        tp = (t / t_total * 100) if t_total else 0
        vp = (v / v_total * 100) if v_total else 0
        absdiff = t - v
        reldiff = ((tp - vp) / vp * 100) if vp else float('inf') if tp else 0
        reldiff_str = f"{reldiff:.2f}" if vp else ("inf" if tp else "0.00")
        print(f"{str(lbl):30} {t:12d} {tp:8.3f}% {v:12d} {vp:8.3f}% {absdiff:8d} {reldiff_str:8}")


def main():
    train_csv = LEGACY_SPLIT_DIR / 'train.csv'
    val_csv = LEGACY_SPLIT_DIR / 'val.csv'
    lf1, tcounts = read_counts(train_csv)
    lf2, vcounts = read_counts(val_csv)
    if lf1 != lf2:
        print(f"Detected label fields differ: train={lf1} val={lf2}")
    print(f"Using label field: {lf1}\n")
    print_comparison(tcounts, vcounts)


if __name__ == '__main__':
    main()
