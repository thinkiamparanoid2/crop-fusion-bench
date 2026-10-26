# E3 Quantum Fusion

Hybrid quantum-classical fusion head for the controlled E2-vs-E3 comparison.
The first controlled E2-matched vs E3-4 run is complete.

Current comparison report:

```text
docs/by_stage/04_e3_quantum_fusion/E2_MATCHED_E3_4_COMPARISON.md
```

Run commands:

```text
e3_quantum_fusion/RUN_COMMANDS.txt
```

## Design (locked)

References:
- `e2_multimodal_fusion/PARAM_MATCH_SPEC.md` (parameter matching, Spec B)
- `docs/by_stage/04_e3_quantum_fusion/E3_STRUCTURE_AND_IMPLEMENTATION_PLAN.md`
- `docs/by_stage/03_baselines_e0_e1_e2/CANONICAL_RUNS.md`

Both comparators share the Spec B projection and readout, differing only in the
4->4 decision sub-network:

```text
shared projection : Linear(1344,128) -> BN -> ReLU -> Dropout(0.1) -> Linear(128,4)
E2-matched middle : tanh -> Linear(4,4,bias=False) -> tanh        (16 params)
E3-4 middle       : (tanh*pi) -> 4-qubit VQC                       (16 params)
shared readout    : Linear(4,25)
```

Verified trainable params: E2-matched = E3-4 = 173,073 (post-bottleneck head = 16).

## VQC (locked first pass)

```text
4 qubits, angle encoding, 2 layers (RX/RY per qubit), CNOT ring, Pauli-Z readout
16 trainable params, lightning.qubit simulator for the completed run
```

`lightning.gpu` is not installed in the current Windows environment, so E3-4 ran
on CPU. E2-matched ran on CUDA.

Do not change the qubit count for the first pass; the qubit sweep (E3-6, E3-8) is
a later ablation, each with its own E2-matched-k comparator.

## Inputs (reuse E2 caches, do not recompute)

```text
e2_multimodal_fusion/features/{train,val,test}_embeddings.npz
keys: image_embedding [N,1280], iot_embedding [N,64], y [N], image_id, label, stage
canonical class order + fixed split (train 46,713 / val 5,193 / test 5,767)
```

`train_e2_fusion.load_split(feature_dir, split)` already returns `(x[N,1344], y[N], meta)`
and can be reused directly.

## Layout

```text
quantum_layer.py       VQC TorchLayer
models.py              FusionProjection, E2MatchedHead, E3HybridHead
scripts/train_e3.py    training + scoring via common/thesis_eval
scripts/smoke_test.py  shape + param-count checks on real embeddings
results/  checkpoints/
```

## Completed Runs

| Model | Run | Device | Best epoch | Test Macro F1 | Test accuracy | Early recall | ECE |
|---|---|---|---:|---:|---:|---:|---:|
| E2-matched | `e2_matched_4` | cuda | 44 | 0.661438 | 0.924224 | 0.971364 | 0.336519 |
| E3-4 | `e3_4_vqc` | cpu / `lightning.qubit` | 53 | 0.719661 | 0.896133 | 0.868124 | 0.263169 |

## Scoring

Use `common/thesis_eval.py` (identical to E0/E1/E2). Each result JSON must include
`architecture`, `qubits`, `bottleneck_dim`, `total_trainable_params`,
`post_bottleneck_trainable_params`, plus val/test metrics and the confusion matrix.
