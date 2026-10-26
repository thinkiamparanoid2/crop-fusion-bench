# E2 Canonical 128-to-25 Variant

This is an isolated classical E2 fusion variant for testing whether the practical
fusion head benefits from removing the 4D bottleneck.

It reuses the existing frozen E0 and E1 embeddings and does not modify canonical
snapshots.

## Architecture

```text
input: 1344 = 1280 image embedding + 64 IoT embedding
Linear(1344,128)
BatchNorm1d(128)
ReLU
Dropout(0.1)
Linear(128,25)
```

## Purpose

Compare against:

```text
canonical E2: Linear(1344,128) -> BN -> ReLU -> Dropout -> Linear(128,4) -> ReLU -> Linear(4,25)
E2-matched-8: Linear(1344,128) -> BN -> ReLU -> Dropout -> Linear(128,8) -> tanh -> Linear(8,8,bias=False) -> tanh -> Linear(8,25)
```

This run is exploratory and is not canonical unless manually promoted and
documented later.
