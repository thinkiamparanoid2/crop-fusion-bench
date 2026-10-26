# Next Session E3 Handoff

Date: 2026-06-03

## Current Status Note (2026-06-06)

This handoff is historical. E3 implementation and the later 8D attack branch have
advanced beyond it. Current active direction:

```text
docs/by_stage/06_e3_attack/E3_BEAT_E2_ATTACK_PLAN.md
```

Current state:

```text
E2-matched-8 test Macro F1 bar: 0.973356
Best E3 attack so far: e3_8_warm_s42 test Macro F1 0.956587
Active next run: e3_8_warm_pair_s42 targeting 3_a9_2 -> 3_a9_1
```

## Current State

Pre-E3 consolidation is effectively complete. E0/E1/E2 canonical runs are locked,
the E2 embedding pipeline reads canonical E0/E1 by default, shared metrics live in
`common/`, and an `e3_quantum_fusion/` scaffold exists.

Canonical runs:

```text
E0: e0_bs768_mobilenetv3large   -> e0_image_only/canonical/
E1: e1_bs512_mlp128x64          -> e1_iot_only/canonical/
E2: e2_bs512_fusionmlp          -> e2_multimodal_fusion/canonical/
```

E2 cached embeddings exist:

```text
e2_multimodal_fusion/features/train_embeddings.npz
e2_multimodal_fusion/features/val_embeddings.npz
e2_multimodal_fusion/features/test_embeddings.npz
```

PennyLane is installed and pinned:

```text
pennylane==0.42.3
pennylane-lightning==0.42.0
```

Smoke verification passed for both `default.qubit` and `lightning.qubit`.

## Important Decisions

The E3 projection choice was discussed but not fully settled by the team. Two
options exist:

```text
Spec A: 1344 -> 4 -> VQC/readout
Spec B: 1344 -> 128 -> 4 -> VQC/readout
```

Docs were reconciled toward **Spec B** because it preserves E2-current's strong
projection and isolates the quantum/classical substitution to the 4->4 middle:

```text
Shared projection:
Linear(1344,128) -> BatchNorm -> ReLU -> Dropout(0.1) -> Linear(128,4)

E2-matched middle:
Linear(4,4,bias=False)     # 16 params

E3-4 middle:
4-qubit VQC, 2 layers RX/RY # 16 params

Shared readout:
Linear(4,25)
```

Spec B parameter accounting:

```text
shared projection params: 172,932
matched middle params:        16 vs 16
readout params:              125
total trainable params:  173,073 vs 173,073
```

If the team chooses Spec A later, update `PARAM_MATCH_SPEC.md`,
`docs/by_stage/00_context_audit/REPO_AUDIT.md`, `docs/by_stage/04_e3_quantum_fusion/E3_STRUCTURE_AND_IMPLEMENTATION_PLAN.md`, and
the `e3_quantum_fusion/` scaffold before writing training code.

## Current E3 Scaffold Status

Folder:

```text
e3_quantum_fusion/
```

Files:

```text
README.md
models.py
quantum_layer.py
scripts/smoke_test.py
scripts/train_e3.py
results/.gitkeep
checkpoints/.gitkeep
```

Status:

```text
folder scaffold: yes
Spec B skeleton: yes
VQC implementation: no
smoke test implementation: no
training implementation: no
```

Current stubs intentionally raise `NotImplementedError` in:

```text
FusionProjection.forward
E2MatchedHead.forward
E3HybridHead.forward
VQCLayer.__init__
VQCLayer.forward
scripts/smoke_test.py
scripts/train_e3.py
```

## Do Not Use Directly

The folder below is a historical staging/prototype area only:

```text
e2_step_notes/Pre P3 Done Code/
```

Problems with that staging code:

```text
uses Spec A (1344 -> 4), not current Spec B
uses stale evaluate_model_metrics API
uses ECE n_bins=10 instead of canonical n_bins=15
has wrong default path roots if run in place
notes say prior runs trained on Gaussian noise / disconnected data
QuantumLayer may return CPU tensors into CUDA readout
```

It can be used only as a conceptual reference for the VQC shape, not copied
wholesale.

## Canonical Source Policy

Training outputs remain separate. Downstream E2/E3 reads canonical snapshots.

Meaning:

```text
training scripts write regular experiment outputs
canonical/ folders are locked downstream inputs
retraining does not affect E2/E3 until manually promoted into canonical/
```

`scripts/thesis.ps1` now sets:

```text
VIRTUAL_ENV
PATH with .venv-e1/Scripts first
TORCH_HOME
XDG_CACHE_HOME
MPLCONFIGDIR=G:\tmp\lgcrop_matplotlib
```

If the user wants a visible `(.venv-e1)` prompt marker, run:

```powershell
. .\.venv-e1\Scripts\Activate.ps1
```

## Next Recommended Work

Implement only the smoke path first. Do not start full training yet.

Step order:

```text
1. Implement FusionProjection.forward.
2. Implement E2MatchedHead.forward.
3. Implement VQCLayer with PennyLane TorchLayer.
4. Implement E3HybridHead.forward.
5. Implement scripts/smoke_test.py.
6. Run smoke_test on 8 rows from val_embeddings.npz.
7. Verify:
   - input dim = 1344
   - logits shape = [8,25]
   - E2-matched total/post params = 173,073 / 16
   - E3-4 total/post params = 173,073 / 16
   - VQC receives [B,4]
```

Only after the smoke test passes should the training loop be implemented.

Training loop should reuse:

```text
e2_multimodal_fusion/scripts/train_e2_fusion.py helpers where safe
common/thesis_eval.py for metrics
e2_multimodal_fusion/features/*.npz
```

Every E2-matched/E3 result JSON must include:

```text
architecture
qubits
bottleneck_dim
total_trainable_params
post_bottleneck_trainable_params
val metrics
test metrics
confusion matrix path
prediction CSV paths
```

## Verification Already Done

Verified locally:

```text
PennyLane imports
default.qubit smoke circuit
lightning.qubit device construction
E2 cached embeddings exist
E2 manifest JSON parses
E2 metrics JSON parses
canonical E2 metrics JSON parses
```

Known cleanup already handled:

```text
E2 extraction reads canonical E0/E1
E2 manifest/metrics use repo-relative canonical paths
canonical bundles are unignored in .gitignore
Step 14 note is unignored
docs/REPO_AUDIT.md is current
```
