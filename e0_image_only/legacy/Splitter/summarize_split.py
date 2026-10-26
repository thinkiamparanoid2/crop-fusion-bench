"""Summarize class-distribution differences between configured split/train.csv and split/val.csv."""
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LEGACY_SPLIT_DIR = REPO_ROOT / "legacy" / "splits"

def read_counts(path):
    p = Path(path)
    with open(p, newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []
        if not fieldnames:
            return None, Counter()
        # detect label field
        label_field = 'label' if 'label' in fieldnames else fieldnames[-1]
        c = Counter()
        for row in r:
            c[row.get(label_field)] += 1
    return label_field, c

def main():
    tfield, tc = read_counts(LEGACY_SPLIT_DIR / 'train.csv')
    vfield, vc = read_counts(LEGACY_SPLIT_DIR / 'val.csv')
    if tfield != vfield:
        print('Label fields differ:', tfield, vfield)
    labels = sorted(set(tc.keys()) | set(vc.keys()))
    ttotal = sum(tc.values())
    vtotal = sum(vc.values())
    diffs = []
    for lbl in labels:
        t = tc.get(lbl,0); v = vc.get(lbl,0)
        tp = (t/ttotal*100) if ttotal else 0
        vp = (v/vtotal*100) if vtotal else 0
        absdiff = abs(t - v)
        reldiff = abs(tp - vp)
        diffs.append((reldiff, absdiff, lbl, t, v, tp, vp))
    diffs.sort(reverse=True)
    max_reldiff, max_absdiff, lbl, t, v, tp, vp = diffs[0]
    print(f"Train total: {ttotal}, Val total: {vtotal}, classes: {len(labels)}")
    print(f"Max relative percent difference: {max_reldiff:.3f}% for label '{lbl}' (train {t}, val {v})")
    print(f"Max absolute difference: {max_absdiff} for label '{lbl}'")
    print('\nTop 10 labels by relative percent difference:')
    print(f"{'Label':30} Train Val Train% Val% Rel% AbsDiff")
    for reldiff, absdiff, lbl, t, v, tp, vp in diffs[:10]:
        print(f"{lbl:30} {t:5d} {v:5d} {tp:7.3f} {vp:7.3f} {reldiff:7.3f} {absdiff:6d}")

if __name__ == '__main__':
    main()
