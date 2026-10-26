"""Summarize E3 attack-run metric JSON files."""

import json
from pathlib import Path

import pandas as pd

ATTACK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ATTACK_ROOT.parent
RESULTS_DIR = ATTACK_ROOT / "results"
DOCS_ATTACK_DIR = PROJECT_ROOT / "docs" / "by_stage" / "06_e3_attack"
OUT_PATH = DOCS_ATTACK_DIR / "ATTACK_RUN_COMPARISON.md"
E2_MACRO_F1_BAR = 0.973356


def main():
    rows = []
    for path in sorted(RESULTS_DIR.glob("*_metrics.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if str(data["run_name"]).startswith(("smoke_", "bench_")):
            continue
        test = data["test"]
        rows.append({
            "run_name": data["run_name"],
            "checkpoint_kind": data["checkpoint_kind"],
            "seed": data["seed"],
            "model": data.get("model", ""),
            "residual": data.get("residual", False),
            "post_params": data.get("post_bottleneck_trainable_params", ""),
            "alpha_bounded": data.get("alpha_bounded", None),
            "alpha_max": data.get("alpha_max", None),
            "alpha_final": data.get("alpha_final", None),
            "accuracy": test["accuracy"],
            "macro_f1": test["macro_f1"],
            "macro_recall": test["macro_recall"],
            "early_recall": test["early_stage_recall"],
            "mid_recall": test["per_stage_recall"]["2"],
            "severe_recall": test["per_stage_recall"]["3"],
            "ece": test["ece"],
            "beats_e2_macro_f1_bar": test["macro_f1"] > E2_MACRO_F1_BAR,
            "file": path.name,
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_ATTACK_DIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        OUT_PATH.write_text("# E3 Attack Run Comparison\n\nNo attack-run metrics found yet.\n", encoding="utf-8")
        print(OUT_PATH.read_text(encoding="utf-8"))
        return

    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    df.to_csv(RESULTS_DIR / "attack_run_comparison.csv", index=False)

    lines = [
        "# E3 Attack Run Comparison",
        "",
        f"E2-matched-8 test Macro F1 bar: {E2_MACRO_F1_BAR:.6f}.",
        "",
        "| Run | Model | Checkpoint | Seed | Post params | Alpha mode | Alpha | Accuracy | Macro F1 | Macro recall | Early | Mid | Severe | ECE | Beats E2 bar |",
        "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in df.iterrows():
        alpha = "" if row["alpha_final"] is None else f"{row['alpha_final']:.6f}"
        if not row["residual"]:
            alpha_mode = ""
        elif row["alpha_bounded"] is True:
            alpha_mode = f"bounded <= {row['alpha_max']:.3g}"
        else:
            alpha_mode = "free"
        model_name = f"{row['model']}{' residual' if row['residual'] else ''}"
        lines.append(
            f"| {row['run_name']} | {model_name} | {row['checkpoint_kind']} | {row['seed']} | "
            f"{row['post_params']} | {alpha_mode} | {alpha} | "
            f"{row['accuracy']:.6f} | "
            f"{row['macro_f1']:.6f} | {row['macro_recall']:.6f} | {row['early_recall']:.6f} | "
            f"{row['mid_recall']:.6f} | {row['severe_recall']:.6f} | {row['ece']:.6f} | "
            f"{'yes' if row['beats_e2_macro_f1_bar'] else 'no'} |"
        )
    lines.append("")
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
