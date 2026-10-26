# E2-vs-E3 Parameter-Matching Specification

Date: 2026-06-03
Status: Current E3 implementation specification (E3 scaffold exists; training is not implemented yet)

This document fixes, before E3 is implemented, the exact interface and the
trainable-parameter accounting rule for the controlled E2-vs-E3 comparison.
The comparison must isolate the effect of replacing a classical decision head
with a variational quantum circuit (VQC); everything else is held fixed.

## 1. Shared frozen interface

Both E2 and E3 consume the same frozen embeddings produced by
[scripts/extract_e2_embeddings.py](scripts/extract_e2_embeddings.py):

```text
E0 image encoder (frozen): 1280-dim embedding   (canonical e0_bs768_mobilenetv3large)
E1 IoT encoder   (frozen):   64-dim embedding   (canonical e1_bs512_mlp128x64)
Fused input:               1344-dim  (concatenation, image first then IoT)
Output label space:          25 sorted classes
Split:                     fixed Data/train.csv (46,713) / val.csv (5,193) / test.csv (5,767)
```

The frozen encoder parameters are identical for E2 and E3 and are NOT trainable.
They do not count toward the parameter budget.

## 2. The three models

To avoid conflating the strongest classical baseline with the fair quantum
comparator, three distinct models are defined and reported under distinct names.

### E2-current (existing strongest classical baseline)

```text
Linear(1344, 128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.1) -> Linear(128, 4) -> ReLU -> Linear(4, 25)
```

This is the canonical `e2_bs512_fusionmlp` run. It uses a
128-unit hidden layer before the 4-dim bottleneck. It is the classical
performance ceiling and the projection interface E3-4 will reuse.

### E2-matched (fair classical comparator for E3-4)

```text
Linear(1344, 128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.1) -> Linear(128, 4) -> tanh -> Linear(4, 4, bias=False) -> tanh -> Linear(4, 25)
```

Same 1344->128->4 projection and same readout as E3-4. The only difference vs
E3-4 is the middle 4->4 transform: a classical `Linear(4, 4, bias=False)` (16
parameters) instead of the VQC. This keeps the strong E2 bottleneck projection
and isolates the quantum substitution to the decision sub-network.

### E3-4 (first controlled quantum model)

```text
Linear(1344, 128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.1) -> Linear(128, 4) -> (tanh * pi angle scaling) -> 4-qubit VQC -> Linear(4, 25)
```

The VQC replaces the classical 4->4 decision sub-network.

## 3. Trainable-parameter accounting rule

Every E2/E3 result JSON must report two counts:

```text
total_trainable_params          # whole trainable head (excludes frozen encoders)
post_bottleneck_trainable_params # decision sub-network operating on the 4-dim bottleneck
```

Rules:

- Frozen E0/E1 encoder parameters never count.
- The shared `Linear(1344, 128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.1) -> Linear(128, 4)`
  projection and shared `Linear(4, 25)` readout are reported but are identical
  across E2-matched and E3-4, so they cancel in the comparison.
- The +/-10% trainable-parameter tolerance (master plan sec 12.4) is evaluated on
  BOTH the total head and, primarily, the post-bottleneck decision sub-network,
  because the projection dominates the total and is not where the quantum
  substitution happens.

## 4. Exact parameter counts (verified)

| Model | Shared projection (1344->128->4) | Decision sub-network (4->4) | Readout (4->25) | Total trainable |
|---|---:|---:|---:|---:|
| E2-current | 172,932 | n/a | 125 | 173,057 |
| E2-matched | 172,932 | 16 (Linear 4->4, no bias) | 125 | 173,073 |
| E3-4 | 172,932 | 16 (VQC, see sec 5) | 125 | 173,073 |

Match check (E2-matched vs E3-4):

```text
total:                173,073 vs 173,073 -> 0.0% difference (within +/-10%)
post-bottleneck head:      16 vs 16      -> 0.0% difference (within +/-10%)
```

E2-current (173,057) is nearly identical in total size but lacks the explicit
matched 4->4 decision sub-network. It remains the existing classical ceiling;
the controlled quantum-vs-classical claim is E2-matched vs E3-4.

## 5. Locked first-pass VQC configuration

```text
qubits:        4
encoding:      angle encoding (inputs scaled by tanh * pi into rotation angles)
ansatz depth:  2 layers
trainable gates: RX and RY per qubit per layer
entanglement:  CNOT ring
measurement:   Pauli-Z expectation per qubit (4 outputs)
readout:       classical Linear(4, 25)
trainable VQC params: 2 layers * 4 qubits * 2 rotations = 16
simulator:     default.qubit (CPU) for the first pass
```

Training schedule (master plan sec 12.3): freeze-then-fine-tune. Stage 1 trains
the projection + VQC + readout with the encoders frozen; optional Stage 2 lightly
fine-tunes end-to-end at a low learning rate.

Do not increase qubits in the first E3 implementation.

## 6. Qubit-sweep policy (later ablation, not first pass)

Each VQC width must have its own parameter-matched classical comparator:

```text
E3-4 vs E2-matched-4   (bottleneck 4, VQC 16 params)
E3-6 vs E2-matched-6   (bottleneck 6)
E3-8 vs E2-matched-8   (bottleneck 8)
```

E2-matched-k uses
`Linear(1344,128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.1) -> Linear(128,k) -> tanh -> Linear(k,k,bias=False) -> tanh -> Linear(k,25)`.
E3-k uses the same projection and readout but replaces `Linear(k,k,bias=False)`
with a k-qubit / 2-layer VQC. Report the per-width match in each result. For
k=4 the decision sub-network is exactly matched at 16 vs 16 params; for wider
k, report total and post-bottleneck counts explicitly because k*k and 4k differ.

## 7. Required metrics for every E2/E3 comparison

Scored with the shared `common/thesis_eval.py` logic so E2 and E3 are byte-identical:

```text
accuracy
macro_f1
macro_recall
early_stage_recall
per_stage_recall
ece
confusion_matrix
total_trainable_params
post_bottleneck_trainable_params
```

## 8. Notes and open issues

- The 4-dim bottleneck can reduce classical accuracy versus E2-current because it
  compresses 1344 fused features into 4 values. This is acceptable: E3-4 is
  constrained by the same bottleneck, so E2-matched is the fair reference.
- A 4-qubit `default.qubit` simulator is slow at full-dataset scale; budget time
  accordingly or precompute on the frozen embeddings (no image forward passes are
  needed because embeddings are cached).
- If E3-4 does not beat E2-matched on Macro F1, the thesis still reports
  early-stage recall, calibration (ECE), robustness, parameter efficiency, and
  training stability per the master plan interpretation map.
