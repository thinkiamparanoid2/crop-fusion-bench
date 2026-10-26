# Canonical E2 classical fusion run

Frozen snapshot copied 2026-06-03 from the original run `e2_frozen_e0_manual_e1_h128_64_fusion`.

- Canonical name: `e2_bs512_fusionmlp` (scheme: e#_batchsize_architecture)
- Architecture: frozen-encoder fusion MLP (1344-dim input), batch size 512

This folder is the authoritative copy referenced by `docs/by_stage/03_baselines_e0_e1_e2/CANONICAL_RUNS.md`.
The original artifacts remain in `e2_multimodal_fusion/checkpoints` and
`e2_multimodal_fusion/results`. Training scripts still write regular experiment
outputs, but downstream E3 comparison work should read this canonical snapshot
by default. If you retrain, refresh this snapshot manually before using the new
run downstream.
