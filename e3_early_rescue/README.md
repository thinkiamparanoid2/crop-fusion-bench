# E3 Early-Detection Rescue

Purpose: isolate loss-function experiments that try to recover E3-4 early-stage
recall without modifying the completed baseline E3 comparison.

Baseline issue from `docs/by_stage/04_e3_quantum_fusion/E2_MATCHED_E3_4_COMPARISON.md`:

```text
E2-matched-4 CE test early recall: 0.971364
E3-4 CE test early recall:        0.868124
drop:                            -0.103240
```

Goal:

```text
E3-4 should no longer lose about 10 points of early-stage recall.
```

## What This Folder Runs

The script reuses:

```text
e2_multimodal_fusion/features/{train,val,test}_embeddings.npz
e3_quantum_fusion/models.py
common/thesis_eval.py
```

It writes new artifacts only under:

```text
e3_early_rescue/results/
e3_early_rescue/checkpoints/
```

## Loss Variants

```text
weighted_ce       standard weighted cross entropy, included as a control
focal             unweighted focal loss
weighted_focal    focal loss with train-set class weights
```

For fair comparison, run the same loss variant for both:

```text
e2_matched
e3_4
```

## Recommended Manual Runs

From repo root:

```powershell
& ".\.venv-e1\Scripts\python.exe" .\e3_early_rescue\scripts\train_rescue.py `
  --model e3_4 `
  --loss focal `
  --run-name e3_4_focal_g2 `
  --sim lightning.qubit `
  --gamma 2.0 `
  --checkpoint-metric val_early_stage_recall `
  --batch-size 512 `
  --epochs 60 `
  --patience 8
```

```powershell
& ".\.venv-e1\Scripts\python.exe" .\e3_early_rescue\scripts\train_rescue.py `
  --model e2_matched `
  --loss focal `
  --run-name e2_matched_focal_g2 `
  --gamma 2.0 `
  --checkpoint-metric val_early_stage_recall `
  --batch-size 512 `
  --epochs 60 `
  --patience 8
```

If focal loss does not recover early recall, run weighted focal:

```powershell
& ".\.venv-e1\Scripts\python.exe" .\e3_early_rescue\scripts\train_rescue.py `
  --model e3_4 `
  --loss weighted_focal `
  --run-name e3_4_weighted_focal_g2 `
  --sim lightning.qubit `
  --gamma 2.0 `
  --checkpoint-metric val_early_stage_recall `
  --batch-size 512 `
  --epochs 60 `
  --patience 8
```

```powershell
& ".\.venv-e1\Scripts\python.exe" .\e3_early_rescue\scripts\train_rescue.py `
  --model e2_matched `
  --loss weighted_focal `
  --run-name e2_matched_weighted_focal_g2 `
  --gamma 2.0 `
  --checkpoint-metric val_early_stage_recall `
  --batch-size 512 `
  --epochs 60 `
  --patience 8
```

## Compare Runs

After two or more rescue runs finish:

```powershell
& ".\.venv-e1\Scripts\python.exe" .\e3_early_rescue\scripts\compare_rescue.py
```

This writes:

```text
e3_early_rescue/results/E3_EARLY_RESCUE_COMPARISON.md
```

## Notes

- `e3_4` runs on CPU with `lightning.qubit` because `lightning.gpu` is not
  installed in the current Windows environment.
- `e2_matched` runs on CUDA when available.
- The default checkpoint metric is `val_early_stage_recall` because this folder's
  purpose is early-detection rescue, not general Macro F1 optimization.
