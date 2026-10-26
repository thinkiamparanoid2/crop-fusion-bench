# Canonical E0 image-only run

Frozen snapshot copied 2026-06-03 from the original run `e0_manual_bs768_w0_20260602`.

- Canonical name: `e0_bs768_mobilenetv3large` (scheme: e#_batchsize_architecture)
- Architecture: MobileNetV3-Large, batch size 768

This folder is the authoritative copy referenced by `docs/by_stage/03_baselines_e0_e1_e2/CANONICAL_RUNS.md`.
The original artifacts remain in `e0_image_only/model` and `e0_image_only/learning/results`.
Training scripts still write regular experiment outputs, but downstream E2/E3
steps read this canonical snapshot by default. If you retrain, refresh this
snapshot manually before using the new run downstream.
