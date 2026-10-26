# Secondary Augmented-Training Scaffold

Supervisor update on 2026-06-10 requested a separate training arm:

```text
Train the model with augmented images and clean images.
Test once with only clean data.
Test secondly with both clean and augmented data.
Compare performance.
```

This is not part of the completed primary clean-trained corruption study. It should be implemented
under a separate folder:

```text
augmentation_training_study/
```

## Recommended First Scope

Start with E0 image-only, because the request specifically says augmented images.

```text
existing baseline: clean-trained E0
new model:         E0 trained on clean + augmented images
```

E1 has no image input, so image augmentation does not apply to E1. E2/E2-8 can be extended later
after E0 is complete.

## Proposed Evaluation

```text
1. clean-trained E0 -> clean test
2. clean-trained E0 -> corrupted test conditions
3. augmented-trained E0 -> clean test
4. augmented-trained E0 -> corrupted test conditions
5. augmented-trained E0 -> combined clean + augmented/corrupted test
```

Primary metric:

```text
Macro-F1
```

Supporting metrics:

```text
accuracy
macro precision
macro recall
weighted F1
early-stage recall
per-stage recall
ECE if probabilities are saved
```

## Recommended Augmentations

```text
gaussian_noise
brightness_decrease
motion_blur
defocus_blur
```

Train on severities:

```text
2 and 3
```

Keep severity 5 primarily for testing as a harder held-out stress level.

## Guardrails

```text
Do not overwrite canonical checkpoints.
Do not mix outputs into corruption_study/results.
Do not retrain E3/VQC.
Do not use test statistics for model selection or preprocessing.
Do not claim robustness improvement unless Macro-F1 drop improves.
Report clean-test degradation as a trade-off if augmented training hurts clean Macro-F1.
```

No code in this folder trains anything. It is a documentation scaffold only.
