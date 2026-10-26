# Pre-E3 Consolidation Plan

## Current Status Note (2026-06-06)

This plan is completed and historical. The repo has advanced through E3-4,
E3-8, robustness checks, and the first warm-start attack run. The current active
direction is:

```text
docs/by_stage/06_e3_attack/E3_BEAT_E2_ATTACK_PLAN.md
```

The safe fallback/framing docs are under:

```text
docs/by_stage/07_final_claim_revision/
docs/by_stage/08_reports_future_work/
```

## Purpose

Before implementing E3, the repo needs a locked and reproducible E0/E1/E2 foundation. E3 should not start until the classical baselines, shared evaluation logic, result docs, and E2-vs-E3 comparison rules are stable.

The goal is to make E3 a clean controlled experiment:

```text
E0: image-only baseline
E1: IoT-only baseline
E2: classical image + IoT fusion baseline
E3: same fusion setup, with VQC quantum head
```

## Current Confirmed E2 Interface

E3 must reuse the E2 fused embedding interface:

```text
E0 image embedding: 1280 dims
E1 IoT embedding:     64 dims
Fused input:        1344 dims
Current E2 head:    1344 -> 128 -> 4 -> 25
```

Current E2 already includes a 4-dimensional bottleneck:

```text
Linear(1344,128)
BatchNorm1d(128)
ReLU
Dropout(0.1)
Linear(128,4)
ReLU
Linear(4,25)
```

This bottleneck is important because first-pass E3 will use a 4-qubit VQC.

## Step 1: Lock Canonical Runs

Declare one canonical run per completed experiment and use those names everywhere in docs, commands, and future comparisons.

Canonical targets:

```text
E0: e0_bs768_mobilenetv3large
E1: e1_bs512_mlp128x64
E2: e2_bs512_fusionmlp
```

Required outputs:

```text
canonical checkpoint path
canonical metrics JSON
canonical predictions CSV
canonical confusion matrix
```

Do not compare E3 against older or inconsistent E2 outputs.

## Step 2: Write E2-vs-E3 Parameter-Matching Spec

Create and maintain:

```text
e2_multimodal_fusion/PARAM_MATCH_SPEC.md
```

The spec must define:

```text
shared frozen encoders
shared 1344-dimensional fused input
shared bottleneck dimension
same 25-class label space
trainable parameter accounting rule
first-pass 4-qubit VQC config
later 4/6/8 qubit sweep policy
```

Important rule:

```text
E3-4 comes first.
E3-6 and E3-8 are later ablations, not first implementation targets.
```

## Step 3: Separate Strong Baseline vs Fair Comparator

Keep the naming clear:

```text
E2-current:
Existing classical fusion result.

E2-matched:
Classical comparator with the same bottleneck dimension as E3.

E3-4:
First 4-qubit hybrid model.
```

The 4-dimensional bottleneck can reduce classical performance because it compresses 1344 features into 4 values. That is acceptable for the fair E2-vs-E3 comparison because the VQC is constrained by the same bottleneck.

## Step 4: Standardize Metrics and Evaluation

E0, E1, E2, and E3 must be scored with the same metric definitions.

Required metrics:

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

Before E3 implementation, extract or standardize duplicated evaluation logic where practical:

```text
expected_calibration_error
stage_recall
multiclass_roc_auc
metrics_dict
per_class_metrics
save_confusion_matrix
save_predictions
```

This prevents E3 from being evaluated with slightly different logic than E2.

## Step 5: Produce Unified E0/E1/E2 Results Doc

Create a consolidated baseline result document before E3:

```text
docs/by_stage/03_baselines_e0_e1_e2/E0_E1_E2_RESULTS_COMPARISON.md
```

It should include:

```text
test accuracy
macro F1
macro recall
early-stage recall
per-stage recall
ECE
trainable parameter counts
stage-level confusion summary
short interpretation of E2 vs unimodal baselines
```

The narrative should clearly establish what E3 must beat or match.

## Step 6: Reproducibility and Commands

Update command documentation before E3 so the full pipeline can be regenerated.

Required command coverage:

```text
data preparation
E1 IoT feature extraction
E0 training/evaluation
E1 training/evaluation
E2 embedding extraction
E2 fusion training/evaluation
```

Use canonical run names in:

```text
README.md
RUN_COMMANDS.txt
FOR_THESIS_CONTEXT.md
```

Avoid hardcoded absolute paths in newly generated docs where possible. Prefer repo-relative paths for future E3 artifacts.

Training scripts may still write regular experiment outputs under the historical
run names. Downstream E2/E3 work must read from the locked `canonical/` folders
unless a retrain is explicitly promoted into canonical.

## Step 7: Documentation Drift Cleanup

Before E3 implementation, align existing docs with the current repo state:

```text
README.md:
Add E2 status, E2 commands, and next step as Pre-E3/E3.

FOR_THESIS_CONTEXT.md:
Update E2 completion state and point next action to E3 planning/implementation.

RUN_COMMANDS.txt:
Add canonical E2 extract/train commands.

docs/by_stage/01_thesis_planning:
Keep master plan and roadmap consistent with the final E3 direction.
```

The docs should not imply that E3 code already exists before it does.

## Step 8: Pre-E3 Acceptance Criteria

E3 implementation should start only after:

```text
canonical E0/E1/E2 runs are named
E2-vs-E3 parameter spec exists
E2 fused input interface is documented
metrics list is locked
E0/E1/E2 comparison doc exists
commands can reproduce E2
docs point to E3 as the next implementation step
```

## Out of Scope Until After Pre-E3

Do not implement these during Pre-E3 consolidation:

```text
PennyLane or quantum circuit code
E3 training script
E3 results generation
6-qubit or 8-qubit models
hardware execution
AgriFusionNet validation
multi-seed significance testing
```

These come after the baseline and comparison foundation is stable.

## First E3 Direction After This Plan

Once Pre-E3 is complete, implement:

```text
E3-4:
1344 fused input
-> 128 projection
-> 4-dimensional bottleneck
-> 4-qubit VQC
-> classical readout
-> 25 logits
```

Later, run:

```text
E3-6 / E2-matched-6
E3-8 / E2-matched-8
```

Those are capacity-vs-stability ablations, not the first E3 implementation.

## Remaining Dependency Gate Before E3

E3 requires a quantum simulator dependency before implementation:

```text
PennyLane
```

At the time of this plan, PennyLane is not part of the pinned baseline stack.
Add and verify it before writing E3 code, then keep the first implementation on
`default.qubit` unless the parameter-matching spec is updated.
