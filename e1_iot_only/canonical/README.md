# Canonical E1 IoT-only run

Frozen snapshot copied 2026-06-03 from the original run `e1_h128_64_nodrop_wd_bs512_sched`.

- Canonical name: `e1_bs512_mlp128x64` (scheme: e#_batchsize_architecture)
- Architecture: MLP 128/64, batch size 512

This folder is the authoritative copy referenced by `docs/by_stage/03_baselines_e0_e1_e2/CANONICAL_RUNS.md`.
The original artifacts remain in `e1_iot_only/checkpoints` and `e1_iot_only/results`.
Training scripts still write regular experiment outputs, but downstream E2/E3
steps read this canonical snapshot by default. If you retrain, refresh this
snapshot manually before using the new run downstream.
