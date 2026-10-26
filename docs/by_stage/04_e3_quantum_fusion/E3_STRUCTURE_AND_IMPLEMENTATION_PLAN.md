# E3 Structure and Implementation Plan

## Summary

E3 is the controlled hybrid quantum version of E2. It is not a full replacement of the E0/E1/E2 pipeline. It must reuse the same frozen E0 image embedding, frozen E1 IoT embedding, split, class order, and evaluation metrics so the E2-vs-E3 comparison tests the fusion head only.

Confirmed E2 interface:

```text
E0 image embedding: 1280 dims
E1 IoT embedding:     64 dims
Fused input:        1344 dims
Current E2 head:    1344 -> 128 -> 4 -> 25
```

The 4-dimensional bottleneck is intentional because the first E3 VQC uses 4 qubits. Each bottleneck value becomes one quantum input.

## Model Roles

Use three distinct model names in reports to avoid confusing the strongest classical baseline with the fair quantum comparator:

```text
E2-current:
1344 -> 128 -> 4 -> 25
Existing classical fusion baseline.

E2-matched:
1344 -> 128 -> 4 -> Linear(4,4,bias=False) -> readout -> 25
Fair classical comparator for E3 with the same shared projection and readout.

E3-4:
1344 -> 128 -> 4 -> 4-qubit VQC -> readout -> 25
First controlled quantum model.
```

E0 and E1 remain frozen for the first E3 pass:

```text
same train/val/test split
same class order
same 25-class output label space
same 1280+64 embedding files
same weighted cross entropy where applicable
same metric definitions where possible
```

## First E3 Design

First-pass E3 architecture:

```text
image embedding 1280 + IoT embedding 64
        |
        v
1344 fused vector
        |
        v
Linear(1344,128) + BatchNorm + ReLU + Dropout
        |
        v
Linear(128,4)
        |
        v
E2-matched: Linear(4,4,bias=False)
E3-4: 4-qubit VQC
        |
        v
classical readout -> 25 logits
```

The fair E2-matched comparator uses the same projection and readout:

```text
1344 fused vector
        |
        v
Linear(1344,128) + BatchNorm + ReLU + Dropout
        |
        v
Linear(128,4)
        |
        v
Linear(4,4,bias=False)
        |
        v
classical readout -> 25 logits
```

The E3-4 model replaces only the 4->4 middle:

```text
Linear(128,4)
        |
        v
4-qubit VQC
        |
        v
classical readout -> 25 logits
```

Quantum configuration:

```text
qubits: 4
encoding: angle encoding
ansatz depth: 2 layers
trainable gates: RX/RY
entanglement: CNOT ring
measurement: Pauli-Z
readout: classical linear layer to 25 logits
```

Do not increase qubits in the first E3 implementation. Increasing qubits is a later ablation, not the first controlled comparison.

## Parameter-Matching Rule

Report two parameter counts for E2 and E3:

```text
total_trainable_params
post_bottleneck_trainable_params
```

Frozen E0 and E1 encoder parameters do not count toward E2-vs-E3 trainable parameter comparison.

The shared projection into the bottleneck should be reported separately from the post-bottleneck head because `Linear(1344,128)` dominates the total parameter count, while the controlled classical/quantum substitution is the 16-parameter 4->4 middle.

The thesis comparison should emphasize the matched decision head:

```text
E2 matched middle: 4 -> Linear(4,4,bias=False) -> 4
E3 quantum middle: 4 -> VQC -> 4
```

This avoids claiming that a parameter difference in the large classical projection is actually a quantum effect.

## Qubit Sweep Policy

The first implementation target is:

```text
E3-4
```

Later qubit-sweep ablations:

```text
E3-4: 1344 -> 128 -> 4 -> 4-qubit VQC
E3-6: 1344 -> 128 -> 6 -> 6-qubit VQC
E3-8: 1344 -> 128 -> 8 -> 8-qubit VQC
```

Every qubit size must have a matched classical comparator:

```text
E2-matched-4
E2-matched-6
E2-matched-8
```

This keeps the comparison fair as bottleneck capacity changes.

## Performance Expectation

The bottleneck can hurt classical performance because it compresses 1344 fused features into only 4 values. That is acceptable for the controlled experiment because E3 is also constrained by 4 qubits.

Interpretation:

```text
E2-current / E2-matched = classical reference
E3-4 = fair quantum test under the same bottleneck constraint
E3-6 / E3-8 = later capacity-vs-stability analysis
```

If E3 does not beat E2 on Macro F1, the thesis can still report useful outcomes through early-stage recall, calibration, robustness, parameter efficiency, and training stability.

## Required Metrics

Every E2/E3 comparison must include:

```text
accuracy
macro_f1
macro_recall
early_stage_recall
per_stage_recall
ECE
confusion_matrix
total_trainable_params
post_bottleneck_trainable_params
```

## Test Plan

Verify embedding interface:

```text
image_embedding dim = 1280
iot_embedding dim = 64
fused input dim = 1344
```

Verify model output:

```text
E3 forward pass returns logits shaped [batch_size, 25]
E3-4 VQC receives exactly [batch_size, 4]
```

Verify result artifacts include:

```text
architecture
qubits
bottleneck_dim
total_trainable_params
post_bottleneck_trainable_params
test metrics
validation metrics
confusion matrix
predictions CSV
```

## Acceptance Criteria

E3 implementation is acceptable only when:

```text
E3 trains without shape errors
E3 uses the same split, classes, and features as E2
E3 results JSON includes architecture and parameter accounting
E2-vs-E3 comparison table can be generated without manual recalculation
```

## Assumptions

First implementation target is documentation/specification before code.

First E3 implementation uses 4 qubits.

Qubit increases are deferred to the later E12-style sweep.

Current E2 bottleneck design is retained because it already matches the planned 4-qubit E3 interface.
