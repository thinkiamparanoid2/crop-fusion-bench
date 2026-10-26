# E3 Quantum Fusion to Bottleneck Sweep Changes

Date: 2026-06-04

This document records what changed between the first E3 quantum fusion experiment
and the bottleneck-8 sweep.

## Starting Point: E3 Quantum Fusion

Folder:

```text
e3_quantum_fusion/
```

Completed runs:

```text
e2_matched_4
e3_4_vqc
```

Goal:

```text
Compare a 4-dimensional matched classical decision sub-network against a
4-qubit VQC under the same frozen E2 embedding interface.
```

Shared input:

```text
1280-dim frozen E0 image embedding
64-dim frozen E1 IoT embedding
1344-dim fused input
```

Architecture:

```text
Shared projection:
Linear(1344,128) -> BatchNorm -> ReLU -> Dropout(0.1) -> Linear(128,4)

E2-matched-4 middle:
tanh -> Linear(4,4,bias=False) -> tanh

E3-4 middle:
tanh*pi -> 4-qubit VQC, 2 layers

Shared readout:
Linear(4,25)
```

Parameter matching:

```text
E2-matched-4 middle: 4 * 4 = 16 params
E3-4 VQC: 2 layers * 4 qubits * 2 rotations = 16 params
Total trainable params: 173,073 each
Post-bottleneck params: 16 each
```

Main report:

```text
docs/by_stage/04_e3_quantum_fusion/E2_MATCHED_E3_4_COMPARISON.md
```

## Problem Found

E3-4 improved Macro F1 and ECE over E2-matched-4, but it failed the severe-stage
test and lost early recall:

| Model | Accuracy | Macro F1 | Early recall | Mid recall | Severe recall | ECE |
|---|---:|---:|---:|---:|---:|---:|
| E2-matched-4 | 0.924224 | 0.661438 | 0.971364 | 0.743182 | 0.250000 | 0.336519 |
| E3-4 | 0.896133 | 0.719661 | 0.868124 | 0.947727 | 0.000000 | 0.263169 |

Severe-stage row:

```text
E2-matched-4: 56 severe -> healthy 14, early 28, mid 0,  severe 14
E3-4:         56 severe -> healthy 0,  early 4,  mid 52, severe 0
```

Interpretation:

```text
The 4D bottleneck may be too compressed for severe-stage recognition.
E3-4 appears to route true severe samples mostly into mid-stage predictions.
```

## New Stage: Bottleneck Sweep

New folder:

```text
e3_bottleneck_sweep/
```

Reason for a new folder:

```text
Keep the original E3-4 experiment frozen and isolate the bottleneck-capacity
ablation in its own scripts, checkpoints, and results.
```

New completed runs:

```text
e2_matched_8
e3_8_vqc
```

Main report:

```text
docs/by_stage/05_e3_rescue_bottleneck_robustness/E3_4_vs_E3_8_BOTTLENECK_COMPARISON.md
```

Severe-stage analysis:

```text
e3_bottleneck_sweep/results/severe_stage_confusion.csv
e3_bottleneck_sweep/results/severe_stage_confusion.md
```

## What Changed Architecturally

The frozen E0/E1 embedding interface did not change:

```text
same 1280-dim image embedding
same 64-dim IoT embedding
same 1344-dim fused input
same train/val/test split
same class order
same metrics
same weighted CE loss
```

The bottleneck changed:

```text
from 4 dimensions
to 8 dimensions
```

E2-matched changed from:

```text
Linear(128,4) -> tanh -> Linear(4,4,bias=False) -> tanh -> Linear(4,25)
```

to:

```text
Linear(128,8) -> tanh -> Linear(8,8,bias=False) -> tanh -> Linear(8,25)
```

E3 changed from:

```text
Linear(128,4) -> tanh*pi -> 4-qubit VQC, 2 layers -> Linear(4,25)
```

to:

```text
Linear(128,8) -> tanh*pi -> 8-qubit VQC, 4 layers -> Linear(8,25)
```

## Parameter Matching Change

For bottleneck 4:

```text
E2 middle: 4 * 4 = 16 params
E3 VQC: 2 layers * 4 qubits * 2 rotations = 16 params
```

For bottleneck 8:

```text
E2 middle: 8 * 8 = 64 params
E3 VQC: 4 layers * 8 qubits * 2 rotations = 64 params
```

The 8D sweep intentionally uses 4 VQC layers so the quantum decision sub-network
matches the classical `Linear(8,8,bias=False)` comparator.

Parameter counts:

| Model | Bottleneck | Decision params | Total trainable params |
|---|---:|---:|---:|
| E2-matched-4 | 4 | 16 | 173,073 |
| E3-4 | 4 | 16 | 173,073 |
| E2-matched-8 | 8 | 64 | 173,737 |
| E3-8 | 8 | 64 | 173,737 |

## New Scripts

Added:

```text
e3_bottleneck_sweep/scripts/train_bottleneck.py
e3_bottleneck_sweep/scripts/smoke_test.py
e3_bottleneck_sweep/scripts/analyze_severe_confusion.py
e3_bottleneck_sweep/scripts/compare_bottlenecks.py
e3_bottleneck_sweep/RUN_COMMANDS.txt
```

`train_bottleneck.py` generalizes the E3 training logic to support:

```text
--model e2_matched
--model e3_vqc
--bottleneck 8
--vqc-layers 4
```

`analyze_severe_confusion.py` prints where true Stage 3 samples are predicted:

```text
healthy
early
mid
severe
```

## Result Change

| Model | Accuracy | Macro F1 | Macro recall | Early recall | Mid recall | Severe recall | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2-matched-4 | 0.924224 | 0.661438 | 0.691956 | 0.971364 | 0.743182 | 0.250000 | 0.336519 |
| E3-4 | 0.896133 | 0.719661 | 0.741060 | 0.868124 | 0.947727 | 0.000000 | 0.263169 |
| E2-matched-8 | 0.987515 | 0.973356 | 0.971295 | 0.981161 | 0.936364 | 0.803571 | 0.248338 |
| E3-8 | 0.973643 | 0.943716 | 0.944777 | 0.987189 | 0.802273 | 0.803571 | 0.118487 |

Key deltas:

```text
E3-8 vs E3-4:
accuracy       +0.077510
Macro F1       +0.224055
macro recall   +0.203717
early recall   +0.119066
mid recall     -0.145455
severe recall  +0.803571
ECE            -0.144682

E3-8 vs E2-matched-8:
accuracy       -0.013872
Macro F1       -0.029639
macro recall   -0.026517
early recall   +0.006029
mid recall     -0.134091
severe recall   0.000000
ECE            -0.129850
```

## Severe-Stage Confusion Change

| Run | True severe count | Pred healthy | Pred early | Pred mid | Pred severe | Severe recall |
|---|---:|---:|---:|---:|---:|---:|
| e2_matched_4 | 56 | 14 | 28 | 0 | 14 | 0.250000 |
| e3_4_vqc | 56 | 0 | 4 | 52 | 0 | 0.000000 |
| e2_matched_8 | 56 | 0 | 5 | 6 | 45 | 0.803571 |
| e3_8_vqc | 56 | 0 | 9 | 2 | 45 | 0.803571 |

The 8D bottleneck changes the severe-stage failure mode. E3-4 predicted almost
all severe samples as mid-stage. E3-8 correctly predicts 45 of 56 severe samples.

## Interpretation

The bottleneck sweep answers the severe-stage question:

```text
The 4D bottleneck was too compressed for severe-stage recognition.
```

Both E2-matched-8 and E3-8 recover severe recall to the same value:

```text
0.803571
```

Therefore, severe-stage recovery is best interpreted as a bottleneck-capacity
effect, not as evidence that the quantum layer uniquely solves severe-stage
recognition.

E3-8 still has two meaningful positives:

```text
slightly higher early recall than E2-matched-8
substantially better calibration / lower ECE
```

But the strongest 8D model overall is classical:

```text
E2-matched-8 has higher accuracy, Macro F1, macro recall, and mid-stage recall.
```

## Thesis-Safe Summary

Suggested wording:

```text
Increasing the fusion bottleneck from four to eight dimensions substantially
improved both the classical matched comparator and the hybrid VQC model. The
severe-stage failure observed in E3-4 disappeared in E3-8, with severe recall
rising from 0.000 to 0.804. However, the parameter-matched classical 8D comparator
achieved the same severe-stage recall and stronger overall Macro F1, suggesting
that the recovery is primarily due to increased bottleneck capacity rather than
a uniquely quantum effect. The 8-qubit VQC remained competitive and showed better
calibration, but it did not outperform the matched classical comparator on the
primary Macro F1 metric.
```
