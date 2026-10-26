"""Write a compact Markdown comparison for e3_early_rescue runs."""

import json
from pathlib import Path

RESCUE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RESCUE_ROOT.parent
RESULTS_DIR = RESCUE_ROOT / "results"
OUT_PATH = RESULTS_DIR / "E3_EARLY_RESCUE_COMPARISON.md"

BASELINE_PATHS = [
    PROJECT_ROOT / "e3_quantum_fusion" / "results" / "e2_matched_4_metrics.json",
    PROJECT_ROOT / "e3_quantum_fusion" / "results" / "e3_4_vqc_metrics.json",
]


def load_metrics():
    rows = []
    for path in BASELINE_PATHS:
        if not path.exists():
            continue
        rows.append(read_metrics(path, group="baseline_ce"))
    for path in sorted(RESULTS_DIR.glob("*_metrics.json")):
        rows.append(read_metrics(path, group="early_rescue"))
    return rows


def read_metrics(path, group):
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    test = m["test"]
    val = m["val"]
    return {
        "group": group,
        "run": m["run_name"],
        "model": m["model"],
        "loss": m["loss"],
        "gamma": m.get("focal_gamma"),
        "checkpoint": m.get("checkpoint_metric", m.get("scheduler_monitor")),
        "best_epoch": m["best_epoch"],
        "device": m["device"],
        "sim": m.get("sim"),
        "val_early": val["early_stage_recall"],
        "val_macro_f1": val["macro_f1"],
        "test_acc": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "test_macro_recall": test["macro_recall"],
        "test_early": test["early_stage_recall"],
        "test_stage0": test["per_stage_recall"]["0"],
        "test_stage1": test["per_stage_recall"]["1"],
        "test_stage2": test["per_stage_recall"]["2"],
        "test_stage3": test["per_stage_recall"]["3"],
        "test_ece": test["ece"],
    }


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_metrics()
    lines = [
        "# E3 Early-Detection Rescue Comparison",
        "",
        "This report includes the original CE baselines plus rescue runs generated from `e3_early_rescue/results/*_metrics.json`.",
        "",
        "## Runs",
        "",
        "| Group | Run | Model | Loss | Gamma | Checkpoint | Best epoch | Device | Test acc | Test Macro F1 | Test early recall | Test severe recall | Test ECE |",
        "|---|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {group} | {run} | {model} | {loss} | {gamma} | {checkpoint} | {best_epoch} | {device} | {test_acc} | {test_macro_f1} | {test_early} | {test_stage3} | {test_ece} |".format(
                **{k: fmt(v) for k, v in row.items()}
            )
        )

    lines.extend([
        "",
        "## Per-Stage Test Recall",
        "",
        "| Run | Stage 0 healthy | Stage 1 early | Stage 2 mid | Stage 3 severe |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['run']} | {fmt(row['test_stage0'])} | {fmt(row['test_stage1'])} | {fmt(row['test_stage2'])} | {fmt(row['test_stage3'])} |"
        )

    lines.extend([
        "",
        "## Success Criterion",
        "",
        "The rescue target is to keep E3-4 from losing about 10 points of early-stage recall versus the matched classical comparator.",
        "",
        "## Interpretation",
        "",
        "- Plain focal loss did not rescue early recall for E3-4; it lowered early recall and Macro F1.",
        "- Weighted focal loss did rescue E3-4 early recall numerically, but at a major cost to accuracy and healthy-stage recall.",
        "- None of the rescue runs recovered severe-stage recall; Stage 3 remains 0.0 for all focal variants.",
        "- The weighted focal result should be treated as a tradeoff experiment, not a clean improved E3 model.",
        "",
    ])

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
