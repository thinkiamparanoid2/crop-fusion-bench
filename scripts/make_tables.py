"""Print the README's results tables from saved metrics. No model is loaded or run.

Usage: python scripts/make_tables.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [  # (label, inputs, metrics file)
    ("E0 MobileNetV3-Large", "image", "e0_image_only/canonical/e0_bs768_mobilenetv3large_metrics.json"),
    ("E1 MLP", "sensor", "e1_iot_only/canonical/e1_bs512_mlp128x64_metrics.json"),
    ("E2 fusion MLP (canonical)", "image + sensor", "e2_multimodal_fusion/canonical/e2_bs512_fusionmlp_metrics.json"),
    ("E2-matched-8 (classical head)", "image + sensor", "e3_bottleneck_sweep/results/e2_matched_8_metrics.json"),
    ("E3-8 VQC (8-qubit head)", "image + sensor", "e3_bottleneck_sweep/results/e3_8_vqc_metrics.json"),
]

print("| Model | Inputs | Test macro-F1 | Test accuracy | Test ECE |")
print("|---|---|---|---|---|")
for label, inputs, path in RUNS:
    t = json.loads((ROOT / path).read_text())["test"]
    print(f"| {label} | {inputs} | {t['macro_f1']:.3f} | {t['accuracy']:.3f} | {t['ece']:.3f} |")

print()
print("| Corruption | Unimodal model | Unimodal macro-F1 drop | E2-8 drop | Difference (95% CI) |")
print("|---|---|---|---|---|")
with open(ROOT / "corruption_study/results/difference_of_drops_summary.csv", newline="") as f:
    for r in csv.DictReader(f):
        if r["multimodal_model"] != "e2_8":
            continue
        ci = f"{float(r['difference_of_drops_mean']):.3f} ({float(r['difference_of_drops_ci_low_mean']):.3f} to " \
             f"{float(r['difference_of_drops_ci_high_mean']):.3f})"
        print(f"| {r['comparison']}: {r['corruption_type']} | {r['unimodal_model'].upper()} "
              f"| {float(r['unimodal_macro_f1_drop_mean']):.3f} | {float(r['multimodal_macro_f1_drop_mean']):.3f} | {ci} |")
