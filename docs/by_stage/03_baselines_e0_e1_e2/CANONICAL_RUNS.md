# Canonical Runs (single source of truth)

Date locked: 2026-06-03

This file declares the one canonical run per completed experiment. All thesis
docs, comparison tables, commands, and the future E3 comparison must reference
these runs and no others. Do not compare E3 against older or inconsistent E0/E1/E2
outputs.

## Canonical runs

Canonical runs are named `e#_batchsize_architecture` and each lives as a frozen
snapshot bundle in a per-experiment `canonical/` folder.

| Experiment | Canonical run | Checkpoint (repo-relative) | Metrics JSON | Trainable params |
|---|---|---|---|---:|
| E0 image-only | `e0_bs768_mobilenetv3large` | `e0_image_only/canonical/best_e0_bs768_mobilenetv3large.pth` | `e0_image_only/canonical/e0_bs768_mobilenetv3large_metrics.json` | 2,214,985 |
| E1 IoT-only | `e1_bs512_mlp128x64` | `e1_iot_only/canonical/best_e1_bs512_mlp128x64.pt` | `e1_iot_only/canonical/e1_bs512_mlp128x64_metrics.json` | 17,305 |
| E2 classical fusion | `e2_bs512_fusionmlp` | `e2_multimodal_fusion/canonical/best_e2_bs512_fusionmlp.pt` | `e2_multimodal_fusion/canonical/e2_bs512_fusionmlp_metrics.json` | 173,057 |

All three use the same fixed split and 25-class sorted label space:

```text
Data/train.csv: 46,713 rows
Data/val.csv:    5,193 rows
Data/test.csv:   5,767 rows
```

E2 freezes the canonical E0 image encoder and the canonical E1 IoT encoder; the
copied E2 metrics JSON references the two canonical checkpoints above.

## Naming and layout

Each `canonical/` folder holds the full run bundle (checkpoint, metrics, history,
val/test predictions, test confusion matrix; for E0 also config, class mapping,
data audit, and OOD corruptions) plus a short `README.md`. The bundles are frozen
**copies** made on 2026-06-03; the original artifacts remain in place and the
training scripts still emit regular experiment outputs. Downstream E2/E3 scripts
read canonical snapshots by default. If you retrain, refresh the snapshot
manually before using the new run downstream.

Original run name -> canonical name and folder:

```text
e0_manual_bs768_w0_20260602            -> e0_bs768_mobilenetv3large   (e0_image_only/canonical/)
e1_h128_64_nodrop_wd_bs512_sched       -> e1_bs512_mlp128x64          (e1_iot_only/canonical/)
e2_frozen_e0_manual_e1_h128_64_fusion  -> e2_bs512_fusionmlp          (e2_multimodal_fusion/canonical/)
```

## Canonical test metrics

| Experiment | Accuracy | Macro F1 | Macro Recall | Early Recall | Severe Recall | ECE |
|---|---:|---:|---:|---:|---:|---:|
| E0 image-only | 0.982140 | 0.963896 | 0.962667 | 0.980407 | 0.803571 | 0.007805 |
| E1 IoT-only | 0.943645 | 0.883370 | 0.882642 | 0.939714 | 0.482143 | 0.016543 |
| E2 classical fusion | 0.987342 | 0.967926 | 0.970493 | 0.986436 | 0.821429 | 0.006775 |

Canonical validation Macro F1: E0 0.973325, E1 0.887618, E2 0.970913.

Full per-stage recall, confusion matrices, and narrative live in
[E0_E1_E2_RESULTS_COMPARISON.md](E0_E1_E2_RESULTS_COMPARISON.md).

## Non-canonical E0 runs (do not cite for the thesis comparison)

The E0 model directory contains several older checkpoints. Only the bs768 run
above is canonical. In particular:

- `e0_mobilenetv3_large_gpu_bs256_unfreeze2_amp` (batch size 256) is an earlier
  run with slightly different metrics (test acc 0.983180, Macro F1 0.961853).
- It is the run cited by the historical [E0_E1_RESULTS_COMPARISON.md](E0_E1_RESULTS_COMPARISON.md).

OOD robustness has been re-run on the canonical E0 run; see
[E0_OOD_CORRUPTION_RESULTS.md](E0_OOD_CORRUPTION_RESULTS.md). That OOD bundle is
also copied into `e0_image_only/canonical/`.
