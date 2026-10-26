# E3 Bottleneck Sweep

Purpose: test whether the 4D bottleneck is too compressed by training a matched
8D classical comparator and an 8-qubit VQC model.

This folder is isolated from:

```text
e3_quantum_fusion/
e3_early_rescue/
```

It reuses the same frozen E2 embedding cache:

```text
e2_multimodal_fusion/features/{train,val,test}_embeddings.npz
```

## Models

```text
E2-matched-8:
Linear(1344,128) -> BN -> ReLU -> Dropout(0.1) -> Linear(128,8)
-> tanh -> Linear(8,8,bias=False) -> tanh -> Linear(8,25)

E3-8:
Linear(1344,128) -> BN -> ReLU -> Dropout(0.1) -> Linear(128,8)
-> tanh*pi -> 8-qubit VQC -> Linear(8,25)
```

Parameter matching:

```text
E2-matched-8 middle: Linear(8,8,bias=False) = 64 params
E3-8 VQC: 4 layers * 8 qubits * 2 rotations = 64 params
```

## Commands

See:

```text
e3_bottleneck_sweep/RUN_COMMANDS.txt
```

