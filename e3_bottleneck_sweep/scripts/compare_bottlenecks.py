"""Generate docs/results/E3_4_vs_E3_8_BOTTLENECK_COMPARISON.md."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWEEP_ROOT = ROOT / "e3_bottleneck_sweep"
DOCS_RESULTS = ROOT / "docs" / "results"
OUT_PATH = DOCS_RESULTS / "E3_4_vs_E3_8_BOTTLENECK_COMPARISON.md"

RUNS = [
    ("E2-matched-4", ROOT / "e3_quantum_fusion" / "results" / "e2_matched_4_metrics.json"),
    ("E3-4", ROOT / "e3_quantum_fusion" / "results" / "e3_4_vqc_metrics.json"),
    ("E2-matched-8", SWEEP_ROOT / "results" / "e2_matched_8_metrics.json"),
    ("E3-8", SWEEP_ROOT / "results" / "e3_8_vqc_metrics.json"),
]


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def row(label, metrics):
    test = metrics["test"]
    stages = test["per_stage_recall"]
    return {
        "label": label,
        "run": metrics["run_name"],
        "model": metrics["model"],
        "bottleneck": metrics["bottleneck_dim"],
        "qubits": metrics.get("qubits"),
        "vqc_layers": metrics.get("vqc_layers"),
        "best_epoch": metrics["best_epoch"],
        "device": metrics["device"],
        "params": metrics["total_trainable_params"],
        "post_params": metrics["post_bottleneck_trainable_params"],
        "accuracy": test["accuracy"],
        "macro_f1": test["macro_f1"],
        "macro_recall": test["macro_recall"],
        "early": test["early_stage_recall"],
        "mid": stages["2"],
        "severe": stages["3"],
        "ece": test["ece"],
    }


def by_label(rows):
    return {row["label"]: row for row in rows}


def delta(a, b, key):
    return b[key] - a[key]


def main():
    # Regenerate severe confusion if possible.
    analyzer = SWEEP_ROOT / "scripts" / "analyze_severe_confusion.py"
    subprocess.run([sys.executable, str(analyzer)], check=True)

    rows = []
    missing = []
    for label, path in RUNS:
        if not path.exists():
            missing.append(str(path))
            continue
        rows.append(row(label, load(path)))

    DOCS_RESULTS.mkdir(parents=True, exist_ok=True)
    lines = [
        "# E3-4 vs E3-8 Bottleneck Comparison",
        "",
        "This report compares the completed 4D bottleneck E2/E3 runs against the new 8D bottleneck sweep.",
        "",
        "## Metrics",
        "",
        "| Model | Run | Bottleneck | Qubits | VQC layers | Best epoch | Device | Params | Post params | Accuracy | Macro F1 | Macro recall | Early recall | Mid recall | Severe recall | ECE |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {label} | {run} | {bottleneck} | {qubits} | {vqc_layers} | {best_epoch} | {device} | {params} | {post_params} | {accuracy} | {macro_f1} | {macro_recall} | {early} | {mid} | {severe} | {ece} |".format(
                **{k: fmt(v) for k, v in r.items()}
            )
        )

    labels = by_label(rows)
    if all(k in labels for k in ["E2-matched-4", "E3-4", "E2-matched-8", "E3-8"]):
        e2_4 = labels["E2-matched-4"]
        e3_4 = labels["E3-4"]
        e2_8 = labels["E2-matched-8"]
        e3_8 = labels["E3-8"]
        lines.extend([
            "",
            "## Key Deltas",
            "",
            "| Comparison | Accuracy | Macro F1 | Macro recall | Early recall | Mid recall | Severe recall | ECE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            f"| E2-matched-8 minus E2-matched-4 | {delta(e2_4, e2_8, 'accuracy'):.6f} | {delta(e2_4, e2_8, 'macro_f1'):.6f} | {delta(e2_4, e2_8, 'macro_recall'):.6f} | {delta(e2_4, e2_8, 'early'):.6f} | {delta(e2_4, e2_8, 'mid'):.6f} | {delta(e2_4, e2_8, 'severe'):.6f} | {delta(e2_4, e2_8, 'ece'):.6f} |",
            f"| E3-8 minus E3-4 | {delta(e3_4, e3_8, 'accuracy'):.6f} | {delta(e3_4, e3_8, 'macro_f1'):.6f} | {delta(e3_4, e3_8, 'macro_recall'):.6f} | {delta(e3_4, e3_8, 'early'):.6f} | {delta(e3_4, e3_8, 'mid'):.6f} | {delta(e3_4, e3_8, 'severe'):.6f} | {delta(e3_4, e3_8, 'ece'):.6f} |",
            f"| E3-8 minus E2-matched-8 | {delta(e2_8, e3_8, 'accuracy'):.6f} | {delta(e2_8, e3_8, 'macro_f1'):.6f} | {delta(e2_8, e3_8, 'macro_recall'):.6f} | {delta(e2_8, e3_8, 'early'):.6f} | {delta(e2_8, e3_8, 'mid'):.6f} | {delta(e2_8, e3_8, 'severe'):.6f} | {delta(e2_8, e3_8, 'ece'):.6f} |",
        ])

    lines.extend([
        "",
        "## Severe-Stage Confusion",
        "",
        "See also `e3_bottleneck_sweep/results/severe_stage_confusion.csv`.",
        "",
    ])
    severe_md = SWEEP_ROOT / "results" / "severe_stage_confusion.md"
    if severe_md.exists():
        lines.extend(severe_md.read_text(encoding="utf-8").splitlines()[4:])

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The 8D bottleneck recovered early recall for E3: E3-4 early recall was 0.868124, while E3-8 reached 0.987189.",
        "- The 8D bottleneck recovered severe recall for E3: E3-4 severe recall was 0.000000, while E3-8 reached 0.803571.",
        "- The classical E2-matched-8 comparator recovered severe recall to the same 0.803571 and achieved stronger Macro F1/accuracy than E3-8. This suggests the severe-stage recovery is primarily a bottleneck-capacity effect, not uniquely a quantum-layer effect.",
        "- E3-8 has slightly higher early recall than E2-matched-8 and substantially lower ECE, but lower accuracy, Macro F1, macro recall, and mid-stage recall.",
        "- The fair conclusion is that the 4D bottleneck was too compressed for severe-stage recognition. The 8D quantum layer is viable and better calibrated, but the strongest 8D model in this sweep is the classical matched comparator.",
        "",
    ])
    if missing:
        lines.extend(["## Missing Inputs", ""])
        lines.extend([f"- `{path}`" for path in missing])
        lines.append("")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
