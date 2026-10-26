# Task: Build a corruption-robustness evaluation harness for a multimodal crop-staging project

> **Purpose:** This document is the build plan / Codex prompt for the P2 corruption-robustness
> study. Direction set with supervisor on 2026-06-09: P2 implements up to E2 only (E0, E1,
> E2-canonical, E2-8); E3 is proposed as architecture but not implemented. The focus is the
> claim that **multimodal fusion degrades more gracefully than unimodal models on corrupted
> test data**. Corruption is applied to TEST data only (primary arm); a train-on-corrupted arm
> is scaffolded but not run.

## Context

This thesis compares unimodal and multimodal models for crop-stage / crop-disease classification.

Implemented model groups:

* E0 = image-only model
* E1 = IoT / sensor-only model
* E2-canonical = standard multimodal fusion model
* E2-8 = multimodal fusion model with 8-dimensional bottleneck / compact fused representation

E3 / VQC / quantum model is **not implemented in P2**.

The P2 thesis claim is:

> Multimodal fusion degrades more gracefully than unimodal models when TEST data is corrupted.

Your job is to build the corruption-robustness evaluation harness that produces evidence for this claim.

Main rules:

* Do **not** implement E3 / VQC / quantum code in this phase.
* Do **not** modify training of the existing clean models.
* Do **not** retrain E0, E1, E2-canonical, or E2-8 in the primary experiment.
* Do **not** alter existing checkpoints.
* Reuse the already-trained clean model artifacts: E0 weights, E1 weights, E2-canonical
  fusion weights, E2-8 fusion weights, label mappings, scalers/normalizers, and existing
  clean metrics.
* Corruption is applied to TEST data only.
* Corrupted test data must be materialized once to disk.
* The same materialized corrupted data must be reused by all relevant models.
* A secondary train-on-corrupted arm may be scaffolded, but it must **not** be run and must not delay the primary harness.

This is an inference/evaluation study, not a new training campaign. The only new E2/E2-8
artifacts required for corrupted conditions are regenerated TEST embeddings produced by
passing corrupted TEST inputs through the already-trained frozen E0/E1 encoders.

---

## Step 0 â€” Inspect the repository before writing anything

Before creating or modifying files, inspect the repo and report:

* Existing E0 training/evaluation scripts
* Existing E1 training/evaluation scripts
* Existing E2-canonical training/evaluation scripts
* Existing E2-8 training/evaluation scripts
* How each model loads the TEST set
* Whether E0 and E1 are evaluated end-to-end on raw inputs, or also from a cached feature/embedding file
* The E2 / E2-8 input path: the cached embedding file (e.g. `e2_multimodal_fusion/features/test_embeddings.npz`), how it was generated, and the dim layout (image-embedding dim + IoT-embedding dim)
* The frozen E0 and E1 encoder modules used to produce E2 embeddings, and exactly how to call them to regenerate embeddings from raw inputs
* The clean loader's exact geometric transforms (resize size, crop, interpolation method)
* Image directory structure
* IoT CSV schema
* IoT feature columns
* Label mapping / class list
* Existing metric output format
* Existing clean-test metrics
* Existing checkpoints
* How checkpoints are loaded
* Whether multiple training-seed checkpoints exist
* Any existing corruption / OOD / robustness utilities or result files

Reuse existing code wherever possible.

Do not rewrite working code unnecessarily.

If something is ambiguous, state your assumption clearly and proceed.

Save the audit summary to:

```text
corruption_study/IMPLEMENTATION_REPORT.md
```

---

## Main deliverable folder

Put everything under a new top-level folder:

```text
corruption_study/
```

Expected structure:

```text
corruption_study/
  data/
    corrupted_images/                   # SEPARATE root (configurable; may live on its own drive)
      {corruption_type}/
        sev_2/
        sev_3/
        sev_5/
    corrupted_iot/                      # SEPARATE root (configurable; may live on its own drive)
      {corruption_type}/
        sev_2/
        sev_3/
        sev_5/
    corrupted_embeddings/               # SEPARATE root: regenerated E2/E2-8 inputs (frozen E0/E1 on corrupted data)
      {condition}/
        {corruption_type}/
          sev_2/
          sev_3/
          sev_5/                        # embeddings.npz per condition/type/severity[/cseed]

  gen/
    corrupt_images.py
    corrupt_iot.py
    regen_embeddings.py                 # frozen E0/E1 -> fresh embeddings from corrupted inputs

  eval/
    run_corruption_eval.py

  results/
    confusion/
    predictions/
      {run_name}__predictions.csv   # per-sample sample_id, y_true, y_pred
    all_metrics.csv
    robustness_summary.csv
    robustness_summary.md
    per_class_recall_by_condition.csv
    severity0_sanity_check.md
    passthrough_sanity_check.md
    corrupted_iot_columns.json

  scaffold_train_corrupted/
    README.md

  README.md
  IMPLEMENTATION_REPORT.md
```

The `scaffold_train_corrupted/` folder is optional and should be created only if it does not delay the primary harness.

---

## Primary experiment rule

Train data remains clean.

Validation data remains clean.

Test data is evaluated under:

* severity 0 = original clean test data
* severity 2 = corrupted test data
* severity 3 = moderately corrupted test data
* severity 5 = severely corrupted test data

Primary P2 scope as of 2026-06-09:

```text
0, 2, 3, 5
```

Severity 1 may be added later if time allows. Severity 6 is not currently defined by the
corruption schedules and must not be used unless explicit severity-6 parameters are added
for every corruption type.

Severity 0 must reproduce the existing clean-test numbers.

Severity 0 is a blocking wiring sanity check.

---

## Checkpoint selection rule

Do **not** use the phrase "same checkpoint across all models."

Each model has its own checkpoint.

Correct rule:

* Use the same selected checkpoint **per model** across all corruption conditions.
* For each model, choose the checkpoint selected by clean validation Macro-F1.
* Prefer checkpoint names like:

  * `best_macro`
  * `best_f1`
  * `best_val_macro_f1`
  * or the repo's equivalent best-validation checkpoint
* Once selected, reuse that exact checkpoint for severity 0â€“5.
* Do not switch checkpoints between corruption conditions.
* Do not retrain or fine-tune any model in the primary corruption study.

Example:

* E0 uses its own best clean-validation checkpoint for all E0 clean/corrupted-image evaluations.
* E1 uses its own best clean-validation checkpoint for all E1 clean/corrupted-IoT evaluations.
* E2-canonical uses its own best clean-validation checkpoint for all E2-canonical corruption conditions.
* E2-8 uses its own best clean-validation checkpoint for all E2-8 corruption conditions.

Macro-F1 is the primary metric for every model and condition.

---

## Training-seed checkpoint rule

If multiple clean-trained checkpoints exist for different training seeds, evaluate all available seed checkpoints.

Example:

```text
E0_seed42
E0_seed13
E0_seed37
```

If multiple checkpoints exist:

* evaluate all seed checkpoints
* report mean Â± std across training seeds
* keep per-seed raw metrics
* encode the training seed in EVERY output filename (`__tseed{n}`) so per-seed runs do not overwrite each other

If only one checkpoint exists per model:

* proceed with one checkpoint
* document that training-seed variance could not be estimated
* omit the `__tseed{n}` suffix

Do not retrain just to create missing seeds unless explicitly requested.

---

## Image corruption generation

Create or update:

```text
corruption_study/gen/corrupt_images.py
```

The script must generate materialized corrupted image test sets.

Use the official ImageNet-C corruption definitions and severity schedules where possible.

If the official ImageNet-C `corruptions.py` file is not already available locally:

* reimplement the required corruptions locally using PIL / OpenCV / scipy / skimage
* use the exact severity parameter schedules listed below
* do not require internet access
* do not silently change the corruption definitions

If a corruption such as motion blur or defocus blur must be approximated because of missing dependencies:

* document the approximation clearly in `corruption_study/README.md`
* record the implementation details in `corruption_study/IMPLEMENTATION_REPORT.md`

Implement these image corruption types:

```text
gaussian_noise
motion_blur
defocus_blur
brightness_increase
brightness_decrease
contrast
jpeg_compression
```

Use these severity schedules, indexed by `severity - 1`:

```text
gaussian_noise:
  std = [.08, .12, .18, .26, .38]

motion_blur:
  (radius, sigma) = [(10,3), (15,5), (15,8), (15,12), (20,15)]

defocus_blur:
  (radius, alias) = [(3,.1), (4,.5), (6,.5), (8,.5), (10,.5)]

brightness_increase:
  delta = [.1, .2, .3, .4, .5]

brightness_decrease:
  delta = [-.1, -.2, -.3, -.4, -.5]

contrast:
  factor = [.4, .3, .2, .1, .05]

jpeg_compression:
  quality = [25, 18, 15, 10, 7]
```

Use fixed seeds.

The generator must be deterministic.

---

## Image corruption resolution rule

Apply image corruptions at the model input geometry, not at the original full image resolution.

Default model input resolution:

```text
224 Ã— 224
```

**Materialized-image geometry rule:**

* The materialized corrupted image must be the EXACT post-geometry image the model consumes. Replicate the clean loader's FULL geometric pipeline (Resize, CenterCrop, aspect handling, and the same interpolation method) BEFORE applying corruption.

* Corruption order:

```text
load clean full-resolution image
â†’ convert RGB
â†’ apply the clean loader's EXACT geometric transforms (resize, crop, etc.) to reach the model input geometry
â†’ apply corruption
â†’ save as PNG
```

* The corrupted-eval dataloader must apply NO geometric transforms (no resize, no crop). It may only do ToTensor + the trained normalization. Geometry on the corrupted path must be identity.

* Inspect the clean loader first and record its exact geometric transforms (size, crop, interpolation) in `IMPLEMENTATION_REPORT.md`.

Reason:

ImageNet-C severity parameters are calibrated for approximately 224 Ã— 224 images. Applying blur radius 15 to a 1024 px image would be much weaker than intended. Re-resizing an already-corrupted image at eval time would also change the intended severity.

Document the final image size and corruption order in:

```text
corruption_study/README.md
corruption_study/IMPLEMENTATION_REPORT.md
```

---

## Corrupted image storage rule

All materialized corrupted images must be saved as **lossless PNG**.

Do not save corrupted outputs as JPEG, because that would add unwanted JPEG artifacts to non-JPEG corruptions.

For `jpeg_compression` corruption:

1. Load clean image.
2. Convert RGB.
3. Apply the clean loader's exact geometric transforms (resize/crop) to model input geometry.
4. Apply JPEG compression in memory using encode â†’ decode.
5. Save the decoded corrupted result as PNG.

For all other image corruptions:

1. Load clean image.
2. Convert RGB.
3. Apply the clean loader's exact geometric transforms (resize/crop) to model input geometry.
4. Apply corruption.
5. Save corrupted result as PNG.

This prevents accidental double corruption.

---

## Corrupted image path-preservation rule

The corrupted image datasets must preserve the original test image structure.

For every corrupted image set:

* preserve the exact relative folder structure
* preserve the exact filename stem
* save output as PNG
* preserve the exact class / disease labels
* preserve the same number of samples as the clean test set
* preserve sample ordering where possible

If the original file is:

```text
Data/test/class_x/img001.jpg
```

the corrupted file should be:

```text
corruption_study/data/corrupted_images/gaussian_noise/sev_2/class_x/img001.png
```

The existing dataset loader should be able to swap:

```text
clean_test_image_root
```

with:

```text
corruption_study/data/corrupted_images/{corruption_type}/sev_{k}/
```

without changing labels, CSV rows, or sample ordering.

Severity 0 should use the original clean test image root, not a regenerated copy.

---

## IoT corruption generation

Create or update:

```text
corruption_study/gen/corrupt_iot.py
```

There is no standard benchmark for IoT corruption, so implement the exact specification below.

**Feature-space decision (state explicitly).** If E1 consumes extracted summary features (a
fixed per-sample feature vector) rather than raw sensor time series, corrupt the extracted
continuous features directly â€” this matches the model input space. Label this in all reports
as **"feature-space sensor corruption,"** NOT raw-device corruption. Honesty caveat: summary
features from one sensor (mean / min / max / std) are correlated, so independent per-feature
noise is an approximation; raw-series corruption (rerun feature extraction per corruption) is
more realistic and may be added later as an extension. Default for the primary arm:
feature-space corruption.

Corruption severity is expressed in units of each feature's TRAINING-set standard deviation.

Compute per-feature mean and standard deviation from the training split only.

Do not compute imputation values from validation or test data.

Do not leak validation or test statistics into preprocessing.

Implement these IoT corruption types:

```text
sensor_noise
dropout
drift
```

Use these severity schedules, indexed by `severity - 1`:

```text
sensor_noise:
  add Gaussian noise with std = [.1, .2, .3, .5, 1.0] Ã— feature_SD

dropout:
  null fraction = [.05, .10, .20, .30, .40]
  then impute using TRAIN-set feature mean

drift:
  add constant offset = [.25, .5, 1.0, 1.5, 2.0] Ã— feature_SD
```

Additional rules:

* **dropout granularity**: apply the null mask PER CELL (each continuous feature value independently dropped with probability = severity fraction), not per row. Document this.
* **drift sign**: apply drift as a per-feature signed offset with the sign fixed by the corruption seed, so drift is not a uniform shift of all features in the same direction. Document the sign assignment. (A constant `+kÂ·SD` is acceptable if you prefer the simplest version â€” but state it.)

Use fixed seeds.

The generator must be deterministic.

---

## IoT continuous-column corruption rule

Only corrupt continuous IoT / sensor features.

Do not corrupt:

```text
ID columns
filename columns
path columns
label columns
class columns
crop code columns
disease code columns
stage code columns
categorical string columns
one-hot encoded categorical columns
split columns
metadata columns
```

Before corruption, inspect the IoT CSV schema and identify the continuous sensor columns.

Save the list of corrupted columns to:

```text
corruption_study/IMPLEMENTATION_REPORT.md
```

Also save a machine-readable file:

```text
corruption_study/results/corrupted_iot_columns.json
```

If categorical or one-hot features exist and are currently used by E1/E2, leave them unchanged unless the repo has an existing safe categorical corruption method.

For IoT corruption types:

```text
sensor_noise
dropout
drift
```

apply them only to the selected continuous sensor columns.

---

## IoT corruption preprocessing rule

IoT corruption must happen in the same feature space expected by the existing evaluation pipeline.

Before implementing IoT corruption, inspect how the current E1/E2 loaders process IoT features.

If the existing loader expects raw CSV values:

* corrupt raw test CSV values
* then let the existing loader apply the trained scaler / normalizer

If the existing loader expects already-scaled CSV values:

* corrupt the scaled values directly
* compute train-set means and standard deviations in that same scaled feature space

In all cases:

* compute feature means and standard deviations from the training split only
* do not compute imputation values from validation or test data
* do not leak test statistics into preprocessing
* document the feature space used for corruption in `corruption_study/README.md`

For dropout corruption:

* create missing values according to the severity fraction
* impute using training-set feature means
* save the final imputed corrupted CSV used by the model

---

## E2 / E2-8 corrupted-embedding regeneration (critical fairness rule)

E2 and E2-8 do NOT consume raw images or raw IoT. They consume cached embeddings (e.g.
`e2_multimodal_fusion/features/test_embeddings.npz`) that were produced by running CLEAN
inputs through the frozen E0 and E1 encoders.

Therefore, swapping the corrupted image root or corrupted IoT CSV does NOTHING to E2. You MUST
regenerate the embeddings from the corrupted inputs.

Required path for every E2 / E2-8 corrupted condition:

```text
corrupted image  â†’ frozen E0 encoder â†’ fresh image embedding
corrupted IoT    â†’ frozen E1 encoder â†’ fresh IoT embedding
concat (image_emb + iot_emb) â†’ E2 / E2-8 fusion head â†’ prediction
```

Hard rules:

* Reuse the existing clean-trained model artifacts. Do not retrain E0, E1, E2-canonical,
  or E2-8 just to run this study.
* Do NOT corrupt the cached clean embeddings directly. Adding noise to the cached vectors is
  REPRESENTATION corruption, not INPUT corruption, and makes the E0/E1 vs E2 comparison invalid
  (E0/E1 would be hit by corrupted inputs while E2 is hit by scrambled internals).
* Use the SAME frozen E0 / E1 encoder weights that produced the original clean embeddings and
  that E0 / E1 themselves use. Do not retrain or alter encoders.
* For every (corruption type Ã— severity Ã— corruption seed), produce a NEW embeddings file from
  the corrupted inputs and feed that to E2 / E2-8. Materialize under:

```text
corruption_study/data/corrupted_embeddings/{condition}/{corruption_type}/sev_{k}[/cseed_{seed}]/embeddings.npz
```

* The corrupted image fed to the E0 encoder for E2 must be the IDENTICAL materialized PNG that
  E0 is evaluated on. The corrupted IoT fed to the E1 encoder for E2 must be the IDENTICAL
  corrupted CSV that E1 is evaluated on. Same bytes, same corruption seed â€” no separate draw.
* Efficiency: a single forward pass over a corrupted image through the frozen E0 backbone yields
  BOTH E0's prediction and the image embedding for E2 â€” compute them together so they provably
  share the same corrupted pixels. Same for E1 â†’ E1 prediction + IoT embedding.

Per-condition embedding sources:

* `image_corrupted_only` = image embedding regenerated from corrupted images + IoT embedding from
  CLEAN IoT (reuse the cached clean IoT embedding, or regenerate from clean IoT â€” must match).
* `iot_corrupted_only`   = IoT embedding regenerated from corrupted IoT + image embedding from
  CLEAN images.
* `both_corrupted`       = both embeddings regenerated from corrupted inputs.

Clean regenerated-embedding sanity check:

* Before trusting corrupted E2 / E2-8 evaluations, regenerate the clean TEST embeddings from
  clean TEST inputs using the same frozen E0/E1 encoder path.
* Evaluate E2-canonical and E2-8 on these regenerated clean embeddings.
* The metrics must match the existing clean E2/E2-8 metrics within the documented severity-0
  tolerance, or the embedding-regeneration path is considered broken.
* This check is verification only. It is not retraining and it does not replace the original
  clean checkpoint or clean cached embedding artifact.

Step 0 must confirm whether E0 and E1 are themselves evaluated end-to-end or also from a cache.
If E0 / E1 evaluation also reads cached features, the SAME regeneration requirement applies to
them â€” never evaluate any model on stale clean caches under a corruption condition.

---

## Severity-0 blocking sanity check

Severity 0 is the original clean test set.

For images:

```text
sev_0 = original clean test image root
```

For IoT:

```text
sev_0 = original clean test CSV
```

Do not regenerate severity-0 images or CSVs.

Severity-0 evaluation must reproduce the existing clean-test metrics within a small tolerance.

Use tolerance:

```text
absolute metric tolerance = 1e-4
```

If severity-0 Macro-F1 or accuracy differs from the existing clean-test result by more than tolerance:

```text
STOP THE RUN
treat it as a blocking wiring bug
do not run the full corruption matrix
write the mismatch to severity0_sanity_check.md
fix the loader/checkpoint/config issue first
```

Do not merely report the mismatch and continue.

Note: the 1e-4 tolerance assumes deterministic evaluation. If GPU nondeterminism causes a sub-threshold miss, document it and use a slightly looser, explicitly stated tolerance rather than treating it as a wiring bug.

---

## Passthrough (identity) sanity check

Severity 0 uses the original clean root through the normal loader, so it does NOT exercise the materialized-image path (resize â†’ corrupt â†’ PNG â†’ reload). Add a second check that does.

Procedure:

* Take the clean test images.
* Run them through the materialized-image path with NO corruption applied (same geometry replication + PNG save + reload + no-op eval loader).
* Evaluate each model on these passthrough images.

Requirement:

* Passthrough Macro-F1 and accuracy must match the existing clean-test metrics within a documented tolerance.
* Use a LOOSER tolerance than severity 0, because PNG encoding and interpolation are not bit-identical to the in-memory loader:

```text
passthrough absolute metric tolerance = 5e-3 (document the exact value used)
```

* If passthrough exceeds tolerance, treat it as a BLOCKING geometry/roundtrip bug: STOP, write details to `passthrough_sanity_check.md`, and fix before running the corruption matrix.

Reason: this is the ONLY check that validates the resize â†’ corrupt â†’ PNG â†’ reload pipeline. A geometry mismatch here would silently change corruption severity on sev 1â€“5 and would NOT be caught by the severity-0 check.

To match interpolation, use the same interpolation method as the clean loader when replicating geometry.

---

## Evaluation matrix

Create or update:

```text
corruption_study/eval/run_corruption_eval.py
```

This script must loop over:

* model
* condition
* corruption type
* severity
* corruption seed where applicable
* training-seed checkpoint where available

Evaluate severities:

```text
0, 2, 3, 5
```

Where:

* severity 0 = original clean test set
* severity 2, 3, 5 = materialized corrupted test sets for the primary P2 scope

The non-corrupted modality always stays clean.

### E0 evaluation

E0 is image-only.

Evaluate:

```text
E0 clean images
E0 corrupted images
```

Conditions:

```text
clean
image_corrupted
```

Applicable corruption types:

```text
gaussian_noise
motion_blur
defocus_blur
brightness_increase
brightness_decrease
contrast
jpeg_compression
```

IoT corruption is not applicable to E0.

### E1 evaluation

E1 is IoT-only.

Evaluate:

```text
E1 clean IoT
E1 corrupted IoT
```

Conditions:

```text
clean
iot_corrupted
```

Applicable corruption types:

```text
sensor_noise
dropout
drift
```

Image corruption is not applicable to E1.

### E2-canonical evaluation

E2-canonical is multimodal.

Evaluate:

```text
E2-canonical clean image + clean IoT
E2-canonical corrupted image + clean IoT
E2-canonical clean image + corrupted IoT
E2-canonical corrupted image + corrupted IoT
```

Conditions:

```text
clean
image_corrupted_only
iot_corrupted_only
both_corrupted
```

For `image_corrupted_only`:

* image = corrupted image set
* IoT = clean test IoT

For `iot_corrupted_only`:

* image = clean test image set
* IoT = corrupted IoT CSV

For `both_corrupted`:

* image = corrupted image set
* IoT = corrupted IoT CSV
* pair image severity k with IoT severity k

The clean reference (`clean_macro_f1`, `clean_accuracy`) for every E2 condition is the model's OWN both-clean result, not E0's or E1's clean.
For E2-canonical, E2-8, and exploratory E2-128-to-25 conditions evaluated from regenerated
embeddings (`clean_regen`, `image_corrupted_only`, `iot_corrupted_only`, and `both_corrupted`),
use that same model's `clean_regen` severity-0 result as the clean reference for point-estimate
drops and paired bootstrap drop CIs. For E0 and E1, use the normal severity-0 `clean` result.
This keeps point-estimate drops and CI drops on the same reference baseline.

NOTE: E2 / E2-8 never read raw inputs. Every condition above requires regenerating embeddings
from the corrupted inputs through the frozen E0 / E1 encoders â€” see "E2 / E2-8 corrupted-embedding
regeneration." Do NOT corrupt cached embeddings.

### E2-8 evaluation

E2-8 is multimodal with compact 8-dimensional bottleneck.

Evaluate the same conditions as E2-canonical:

```text
E2-8 clean image + clean IoT
E2-8 corrupted image + clean IoT
E2-8 clean image + corrupted IoT
E2-8 corrupted image + corrupted IoT
```

Conditions:

```text
clean
image_corrupted_only
iot_corrupted_only
both_corrupted
```

For `both_corrupted`:

* pair image severity k with IoT severity k

---

## both_corrupted pairing rule

The `both_corrupted` condition must not be left ambiguous.

Image corruptions:

```text
gaussian_noise
motion_blur
defocus_blur
brightness_increase
brightness_decrease
contrast
jpeg_compression
```

IoT corruptions:

```text
sensor_noise
dropout
drift
```

Do **not** automatically run every possible 7 x 3 pairing unless explicitly requested.

For the primary P2 study, use exactly two representative paired stress conditions:

```text
canonical paired stress:
  image gaussian_noise + IoT sensor_noise

data-driven worst-case distinct paired stress:
  choose from completed single-modality results
```

Worst-case distinct selection rule:

```text
1. For each image corruption type, compute E0 mean Macro-F1 drop across severities 2, 3, and 5.
   Average over corruption seeds for stochastic corruptions.
2. Select the image corruption with the largest mean E0 Macro-F1 drop as A_worst.
3. For each IoT corruption type, compute E1 mean Macro-F1 drop across severities 2, 3, and 5.
   Average over corruption seeds for stochastic corruptions.
4. Select the IoT corruption with the largest mean E1 Macro-F1 drop as B_worst.
5. If (A_worst, B_worst) equals the canonical pair
   (gaussian_noise, sensor_noise), use the next-largest IoT corruption so the two
   paired stress conditions remain distinct.
```

Wording caveat:

```text
If distinctness from the canonical pair is enforced, call the second pair
"data-driven worst-case distinct paired stress", not "absolute worst pair".
```

Actual selection from the completed single-modality matrix:

```text
E0 worst image corruption by mean Macro-F1 drop: defocus_blur = 0.903679
E1 worst IoT corruption by mean Macro-F1 drop:   drift        = 0.821596

Selected worst-case distinct paired stress:
  image defocus_blur + IoT drift

This pair is already distinct from the canonical gaussian_noise + sensor_noise pair,
so no fallback to the second-largest IoT corruption was needed.
```

Use matched severity index:

```text
image severity k + IoT severity k
```

IMPORTANT framing: matched index is NOT calibrated-equal damage. "image sev-3 + IoT sev-3" does
not mean equal real-world severity across modalities. In all reports, call these
**"representative paired stress conditions,"** never "equivalent severity levels."

The `both_corrupted` filename must include both corruption types:

```text
{model}__both_corrupted__img-{image_corruption_type}__iot-{iot_corruption_type}__sev{k}[__cseed{seed}][__tseed{n}]__metrics.json
```

Example:

```text
e2_canonical__both_corrupted__img-gaussian_noise__iot-sensor_noise__sev3__metrics.json
e2_8__both_corrupted__img-defocus_blur__iot-drift__sev5__metrics.json
```

### Full both_corrupted grid (optional, not run)

The full 7 x 3 image/IoT paired grid is optional future work and was not run for the primary
P2 result. If needed later, it can be produced cheaply by recombination instead of re-running
the frozen E0/E1 encoders:

```text
corruption_study/gen/recombine_embeddings.py
```

Expected helper behavior:

```text
1. Load image_embedding from an existing image_corrupted_only embedding file.
2. Load iot_embedding from an existing iot_corrupted_only embedding file.
3. Verify image_id, label, stage, and y arrays are exactly aligned.
4. Concatenate the existing image and IoT embeddings into a both_corrupted test_embeddings.npz.
5. Write an embedding manifest that records both source embedding files.
```

This is valid because the single-modality corrupted embeddings were already produced from
byte-identical materialized corrupted inputs through the frozen E0 and E1 encoders. This
optional helper must still verify alignment before writing outputs.

---

## Corruption seed rule

Use multiple corruption seeds for stochastic corruption types.

Stochastic image corruptions:

```text
gaussian_noise
```

Stochastic IoT corruptions:

```text
sensor_noise
dropout
```

Use at least 3 corruption seeds:

```text
42
123
999
```

For stochastic corruptions, report:

```text
mean Â± std across corruption seeds
```

For deterministic corruptions, one seed is enough:

```text
motion_blur
defocus_blur
brightness_increase
brightness_decrease
contrast
jpeg_compression
drift
```

If runtime is too high, prioritize:

```text
3 seeds for gaussian_noise
3 seeds for sensor_noise
3 seeds for dropout
1 seed for all deterministic corruptions
```

---

## Evaluation determinism rule

During evaluation:

```text
model.eval()
torch.no_grad()
disable train-time augmentation
disable dropout
disable stochastic transforms in dataloader
fixed batch order
fixed random seeds
```

Set seeds for:

```text
python random
numpy
torch
torch cuda if available
```

If CUDA nondeterminism prevents exact bit-identical results, document it.

---

## Metrics

Compute and save:

```text
accuracy
macro_precision
macro_recall
macro_f1
weighted_f1
confusion_matrix
per_class_recall
```

Macro-F1 is the primary metric.

Also compute robustness degradation:

```text
macro_f1_drop = clean_macro_f1 - corrupted_macro_f1
accuracy_drop = clean_accuracy - corrupted_accuracy
```

Also compute relative degradation:

```text
relative_macro_f1_drop_percent = 100 Ã— (clean_macro_f1 - corrupted_macro_f1) / clean_macro_f1
```

---

## Bootstrap confidence interval rule

For every evaluation condition, compute bootstrap confidence intervals over the test set.

Use:

```text
1000 bootstrap resamples
95% confidence interval
fixed bootstrap seed
```

Bootstrap:

```text
macro_f1
accuracy
macro_f1_drop
accuracy_drop
```

Save CI fields in every metrics JSON:

```text
macro_f1_ci_low
macro_f1_ci_high
accuracy_ci_low
accuracy_ci_high
macro_f1_drop_ci_low
macro_f1_drop_ci_high
accuracy_drop_ci_low
accuracy_drop_ci_high
```

Also include confidence intervals in:

```text
all_metrics.csv
robustness_summary.csv
robustness_summary.md
```

**Paired bootstrap and prediction saving:**

* Save per-sample predictions for every evaluation run: `sample_id` (stable, e.g. relative file path), `y_true`, `y_pred`. Save to:

```text
corruption_study/results/predictions/{run_name}__predictions.csv
```

where `{run_name}` is the metrics filename stem (without `__metrics.json`).

* Compute ALL bootstrap CIs OFFLINE from these saved predictions, not by re-running inference.

* For drop CIs (`macro_f1_drop`, `accuracy_drop`), use a PAIRED bootstrap: in each of the 1000 iterations, draw ONE set of resampled sample indices and apply the SAME indices to BOTH the clean and the corrupted predictions, then compute the drop on that resample. Clean and corrupted are evaluated on the same samples, so independent resampling gives INVALID drop CIs.

* Align clean and corrupted predictions by `sample_id` before pairing.

* Empty-class handling: in an imbalanced resample a rare class may have zero true samples, making its per-class F1 undefined. Compute macro-F1 only over classes present in the resample (skip-and-renormalize) and document this so CIs do not become NaN.

* Use a fixed bootstrap seed.

Optional (stronger claim): also bootstrap the DIFFERENCE of drops between models on the same images (e.g. E0 drop minus E2 drop under the same image corruption), using the same resampled indices across both models, to put a CI directly on "fusion degrades less."

This makes the robustness claim defensible instead of relying only on point estimates.

---

## Robustness interpretation rule

Do not claim that E2 is better only because it has higher clean-test accuracy.

The main robustness claim depends on degradation.

Main formula:

```text
Macro-F1 drop = clean Macro-F1 - corrupted Macro-F1
```

The thesis claim is supported if:

* E2-canonical and/or E2-8 have smaller Macro-F1 drops than E0 under image corruption
* E2-canonical and/or E2-8 have smaller Macro-F1 drops than E1 under IoT corruption
* E2-canonical and/or E2-8 remain more stable when both modalities are corrupted

Report both:

* absolute corrupted Macro-F1
* Macro-F1 drop from clean severity 0

The best robustness result is the model that degrades more gracefully, not necessarily the model with the highest clean score.

---

## Hard fairness and reproducibility rules

Do not violate these:

1. Materialize corrupted sets once to disk with fixed seeds.
2. Never corrupt on-the-fly per model.
3. Every model must read the identical corrupted files for the same corruption condition.
4. Use the same selected checkpoint per model across all corruption conditions.
5. Use Macro-F1 as the primary metric for all models and conditions.
6. Severity 0 must reproduce existing clean-test metrics.
7. Severity 0 mismatch beyond tolerance must halt the run.
8. Do not retrain or fine-tune models in the primary corruption study.
9. Do not corrupt training data in the primary corruption study.
10. Do not corrupt validation data in the primary corruption study.
11. Do not leak validation or test statistics into scaling, imputation, or corruption.
12. Save the exact config used for every run.
13. Do not silently overwrite previous results.
14. Evaluate with `model.eval()` and deterministic settings.
15. Corrupt only continuous IoT features.
16. Save materialized corrupted images as PNG.
17. Apply image corruption after replicating the clean loader's geometry to model input size.
18. For `both_corrupted`, use explicit image+IoT pairings and encode both names in output files.
19. When multiple training-seed checkpoints exist, include the training seed in every output filename to prevent overwrites.
20. Compute drop confidence intervals with a PAIRED bootstrap, not independent resampling.
21. E2 / E2-8 corrupted conditions MUST use embeddings regenerated from corrupted inputs through the frozen E0 / E1 encoders. Never corrupt cached embeddings directly.
22. The corrupted inputs fed to the E0 / E1 encoders for E2 must be byte-identical to those E0 / E1 are evaluated on (same materialized files, same corruption seed).
23. Regenerated clean TEST embeddings must reproduce existing clean E2/E2-8 metrics before corrupted E2/E2-8 results are trusted.
24. Existing clean checkpoints, label mappings, scalers/normalizers, and clean metrics should be reused; do not create replacement trained artifacts.

---

## Output files

For each evaluation run, save one metrics JSON.

For single corruption:

```text
corruption_study/results/{model}__{condition}__{corruption_type}__sev{k}[__cseed{seed}][__tseed{n}]__metrics.json
```

Examples:

```text
corruption_study/results/e0__image_corrupted__gaussian_noise__sev2__cseed42__metrics.json
corruption_study/results/e1__iot_corrupted__dropout__sev3__cseed123__metrics.json
```

For both-corrupted condition:

```text
corruption_study/results/{model}__both_corrupted__img-{image_corruption_type}__iot-{iot_corruption_type}__sev{k}[__cseed{seed}][__tseed{n}]__metrics.json
```

Example:

```text
corruption_study/results/e2_canonical__both_corrupted__img-defocus_blur__iot-drift__sev5__metrics.json
```

Filename rules:

* Include `__cseed{seed}` only for stochastic corruptions.
* Include `__tseed{n}` only when more than one training-seed checkpoint is evaluated.
* Apply the same `__cseed`/`__tseed` suffixes to confusion-matrix and predictions filenames so nothing overwrites.

Each metrics JSON should contain:

```text
model
condition
image_corruption_type
iot_corruption_type
corruption_type
severity
corruption_seed
training_seed
checkpoint_path
image_root_used
iot_csv_used
num_samples
accuracy
macro_precision
macro_recall
macro_f1
weighted_f1
clean_accuracy
clean_macro_f1
accuracy_drop
macro_f1_drop
relative_macro_f1_drop_percent
accuracy_ci_low
accuracy_ci_high
macro_f1_ci_low
macro_f1_ci_high
accuracy_drop_ci_low
accuracy_drop_ci_high
macro_f1_drop_ci_low
macro_f1_drop_ci_high
seed
config_path
timestamp
```

Also save per-sample predictions under:

```text
corruption_study/results/predictions/{run_name}__predictions.csv   # sample_id, y_true, y_pred
```

Also save confusion matrices under:

```text
corruption_study/results/confusion/
```

Suggested naming:

```text
{model}__{condition}__{corruption_type}__sev{k}[__cseed{seed}][__tseed{n}]__confusion.csv
```

For both-corrupted, include both corruption names.

Also generate summary files:

```text
corruption_study/results/all_metrics.csv
corruption_study/results/robustness_summary.csv
corruption_study/results/robustness_summary.md
corruption_study/results/per_class_recall_by_condition.csv
corruption_study/results/severity0_sanity_check.md
corruption_study/results/passthrough_sanity_check.md
corruption_study/results/difference_of_drops_bootstrap.csv
corruption_study/results/difference_of_drops_summary.csv
corruption_study/results/plots/plot_manifest.csv
```

---

## Summary table requirements

The final summary table should include at least:

```text
model
condition
image_corruption_type
iot_corruption_type
corruption_type
severity
corruption_seed
training_seed
accuracy
macro_precision
macro_recall
macro_f1
weighted_f1
clean_accuracy
clean_macro_f1
accuracy_drop
macro_f1_drop
relative_macro_f1_drop_percent
macro_f1_ci_low
macro_f1_ci_high
accuracy_ci_low
accuracy_ci_high
checkpoint_path
```

Rows should include at minimum:

```text
E0, clean
E0, image_corrupted

E1, clean
E1, iot_corrupted

E2-canonical, clean
E2-canonical, image_corrupted_only
E2-canonical, iot_corrupted_only
E2-canonical, both_corrupted

E2-8, clean
E2-8, image_corrupted_only
E2-8, iot_corrupted_only
E2-8, both_corrupted
```

---

## Per-type summary rule

For each model and corruption type, compute:

```text
mean_macro_f1_across_primary_corrupted_severities
std_macro_f1_across_primary_corrupted_severities
mean_macro_f1_drop_across_primary_corrupted_severities
std_macro_f1_drop_across_primary_corrupted_severities
mean_accuracy_across_primary_corrupted_severities
std_accuracy_across_primary_corrupted_severities
mean_accuracy_drop_across_primary_corrupted_severities
std_accuracy_drop_across_primary_corrupted_severities
```

For stochastic corruptions, also compute mean Â± std across corruption seeds.

Do not call this official ImageNet-C mCE unless the implementation exactly matches ImageNet-C mCE.

Call it:

```text
mean corruption robustness summary
```

---

## Downstream figure support

The output must support plotting:

### Degradation curves

```text
x-axis = severity 0â€“5
y-axis = macro-F1
one line per model
one plot per corruption type
error bars where available
```

### Drop curves

```text
x-axis = severity 0â€“5
y-axis = Macro-F1 drop from clean
one line per model
one plot per corruption type
error bars where available
```

### Summary tables

```text
mean Macro-F1 across primary corrupted severities 2, 3, 5
mean Macro-F1 drop across primary corrupted severities 2, 3, 5
95% confidence intervals where available
```

Do not necessarily generate all plots unless easy, but the CSV files must make plotting straightforward.

---

## Phased execution rule

Do not try to complete the full matrix in one uncontrolled run.

**MVP-first scope.** Before generating the full matrix (~173k PNGs), get a thin slice fully green
end-to-end:

```text
E0:           gaussian_noise, brightness_increase, brightness_decrease
E1:           sensor_noise, dropout
E2-canonical: the same four modality-specific corruptions (via regenerated embeddings)
E2-8:         the same four
severities:   0, 2, 3, 5 first
```

Only after severity 0 AND the passthrough check pass on this slice, expand to all planned
corruption types for severities 2, 3, and 5. Severity 1 can be added later if time allows.
Severity 6 requires new explicit schedules before use.

Execute in this order:

### Phase 1 â€” Audit

* inspect repo
* identify scripts
* identify loaders and their exact geometric transforms
* identify checkpoints
* identify clean metrics
* identify feature columns
* identify training-seed checkpoints if present
* write audit summary

### Phase 2 â€” Generators

* implement image corruption generator
* implement IoT corruption generator
* implement the E2 / E2-8 embedding-regeneration step (frozen E0/E1 on corrupted inputs)
* generate one tiny sample first
* verify path preservation
* verify PNG output
* verify image geometry matches the loader's model-input geometry
* verify sample counts
* verify IoT continuous-column selection
* verify regenerated embeddings have the same shape/dim layout as the cached clean embeddings
* verify regenerated clean TEST embeddings reproduce existing clean E2/E2-8 metrics before
  running corrupted E2/E2-8 evaluations
* estimate disk usage before full generation (primary scope â‰ˆ test_count Ã— #types Ã— 3 corrupted severities; e.g. 5,767 x 7 x 3 = 121k PNGs) and confirm free space

### Phase 3 â€” Smoke test

Run:

```text
one model
one corruption type
one severity
```

Also run severity 0 and the passthrough (identity) check.

Confirm:

* severity 0 matches clean metric within tolerance
* passthrough (identity) matches clean metric within tolerance
* metrics JSON is saved
* per-sample predictions CSV is saved
* confusion matrix is saved
* no training is triggered
* sample count is correct
* output filenames are correct
* evaluation uses `model.eval()`

### Phase 4 â€” Full primary matrix

Only after smoke test passes:

* generate full corrupted test sets
* run full evaluation matrix
* save per-sample predictions for every run
* compute robustness drops
* compute bootstrap CIs (paired for drops)
* compute corruption-seed summaries where applicable
* compute training-seed summaries where applicable

### Phase 5 â€” Reports

Generate:

```text
all_metrics.csv
robustness_summary.csv
robustness_summary.md
per_class_recall_by_condition.csv
severity0_sanity_check.md
passthrough_sanity_check.md
IMPLEMENTATION_REPORT.md
README.md
```

---

## Secondary augmented-training plan

Supervisor update on 2026-06-10:

```text
Train the model with augmented images and clean images and test the model in
two approaches:
  1. only clean data
  2. both clean and augmented data
Compare the performance.
```

This is a secondary training arm and must be reported separately from the completed primary
corruption study.

Primary completed study:

```text
train clean -> test clean/corrupted
```

New secondary study:

```text
train clean + augmented images -> test clean
train clean + augmented images -> test clean + augmented/corrupted
```

### Scope recommendation

Start with E0 image-only because the supervisor specifically mentioned augmented images.
This is the fastest and cleanest interpretation of the request.

Recommended first model:

```text
E0 image-only MobileNetV3
```

Optional extension after E0:

```text
E2/E2-8 with augmented-image embeddings + clean IoT embeddings
```

Do not start by retraining all E0/E1/E2/E2-8 variants unless explicitly requested. E1 has no image
input, so image augmentation does not apply to E1.

### Proposed folder

Keep this arm separate from `corruption_study/` primary outputs:

```text
augmentation_training_study/
```

Suggested structure:

```text
augmentation_training_study/
  README.md
  RUN_COMMANDS.md
  CHECKLIST.md
  IMPLEMENTATION_REPORT.md
  data/
    augmented_train_images/
    augmented_test_images/
  scripts/
    make_augmented_image_manifest.py
    train_e0_augmented.py
    eval_e0_augmented.py
    summarize_augmented_results.py
  results/
    metrics/
    predictions/
    confusion/
    plots/
  checkpoints/
```

### Training data design

Use the fixed root split:

```text
Data/train.csv
Data/val.csv
Data/test.csv
```

Training set for augmented model:

```text
clean train images
+ materialized augmented train images
```

Validation set:

```text
clean validation images
```

Keep validation clean for model selection unless a separate augmentation-validation experiment is
explicitly requested. Clean validation keeps the checkpoint selection comparable with the existing
clean-trained E0.

Do not corrupt or augment the official test labels. Test labels remain unchanged.

### Augmentation choices

Use only image corruptions that are realistic and already implemented or easy to reproduce:

```text
gaussian_noise
brightness_decrease
motion_blur
defocus_blur
```

Rationale:

```text
gaussian_noise: sensor/camera noise
brightness_decrease: low light, shade, underexposure
motion_blur: camera/plant movement
defocus_blur: focus errors
```

Recommended training severities:

```text
2 and 3
```

Do not train on severity 5 initially; keep severity 5 as a harder held-out stress level for testing.

Use fixed augmentation seeds:

```text
42
123
999
```

To control training size, do not include every corruption x severity x seed copy in the first run.
Use one augmented copy per clean training image, sampled deterministically from the configured
augmentation pool, or cap the augmented ratio:

```text
augmented_ratio = 1.0
```

This means one augmented image per clean image, so the training set is approximately:

```text
clean train rows + one augmented row per clean train row
```

### Test approach 1: clean-only test

Evaluate the augmented-trained E0 model on:

```text
Data/test
```

Compare against existing clean-trained E0:

```text
clean-trained E0 on clean test
augmented-trained E0 on clean test
```

Main question:

```text
Does augmentation training preserve clean Macro-F1 and accuracy?
```

### Test approach 2: clean + augmented/corrupted test

Evaluate the augmented-trained E0 model on a combined test manifest:

```text
clean test images
+ materialized augmented/corrupted test images
```

Use the same labels for augmented/corrupted copies.

Recommended corrupted test components:

```text
gaussian_noise severity 2, 3, 5
brightness_decrease severity 2, 3, 5
motion_blur severity 2, 3, 5
defocus_blur severity 2, 3, 5
```

For stochastic corruptions, include seeds:

```text
42, 123, 999
```

Report both:

```text
clean-test metrics
combined clean+augmented-test metrics
per-corruption/per-severity corrupted-test metrics
```

Do not collapse everything into only one combined score; the combined score can hide failures on
specific corruptions.

### Required comparisons

At minimum:

```text
existing clean-trained E0 -> clean test
existing clean-trained E0 -> corrupted test conditions
augmented-trained E0 -> clean test
augmented-trained E0 -> corrupted test conditions
augmented-trained E0 -> clean + augmented/corrupted combined test
```

Metrics:

```text
accuracy
macro_precision
macro_recall
macro_f1
weighted_f1
early-stage recall
per-stage recall
ECE if probabilities are saved
confusion matrix
per-class recall
```

Primary metric:

```text
Macro-F1
```

Clean-performance guardrail:

```text
If augmented-trained E0 clean Macro-F1 drops by more than 0.005 absolute versus clean-trained E0,
report the robustness gain as a trade-off, not an unqualified improvement.
```

Robustness metric:

```text
Macro-F1 drop = clean Macro-F1 - corrupted Macro-F1
```

The augmentation training is successful only if it reduces corrupted-test Macro-F1 drop without an
unacceptable clean-test penalty.

### E2 extension, if requested

After E0 augmented training, an E2 extension can be run:

```text
1. Use augmented-trained E0 as the image encoder.
2. Keep the existing clean-trained E1 IoT encoder.
3. Regenerate train/val/test embeddings.
4. Train a new E2 fusion head on clean+augmented-image embeddings + clean IoT.
5. Evaluate clean test and corrupted-image test embeddings.
```

This E2 extension must not overwrite canonical E2 artifacts. Use a separate folder:

```text
augmentation_training_study/e2_augmented_image_encoder/
```

### Output requirements

Save:

```text
augmentation_training_study/results/all_metrics.csv
augmentation_training_study/results/clean_vs_augmented_training_summary.csv
augmentation_training_study/results/clean_vs_augmented_training_summary.md
augmentation_training_study/results/per_class_recall_by_condition.csv
augmentation_training_study/results/confusion/
augmentation_training_study/results/predictions/
```

Every metrics JSON should include:

```text
model
training_regime: clean_only | clean_plus_augmented
test_regime: clean_only | corrupted_only | clean_plus_augmented
augmentation_types
augmentation_severities
augmentation_seed
checkpoint_path
accuracy
macro_precision
macro_recall
macro_f1
weighted_f1
early_stage_recall
per_stage_recall
ece
num_samples
```

### Guardrails

```text
Do not overwrite canonical E0/E1/E2 checkpoints.
Do not mix augmented-training outputs into corruption_study/results.
Do not claim augmentation improves robustness unless Macro-F1 drop improves.
Do not claim clean improvement if clean Macro-F1 decreases.
Do not retrain E3/VQC for this arm.
Do not use test statistics for augmentation selection, normalization, or model selection.
Do not use pseudo-labels as ground truth.
```

### Definition of done for secondary arm

```text
augmented train manifest created
augmented E0 model trained in isolated folder
augmented E0 evaluated on clean test
augmented E0 evaluated on corrupted test conditions
augmented E0 evaluated on combined clean+augmented/corrupted test
results compared against existing clean-trained E0
summary Markdown written
restart context updated
```

---

## README requirements

Create:

```text
corruption_study/README.md
```

It must document:

* What the corruption study tests
* Why train data stays clean
* Why validation data stays clean
* How to regenerate corrupted image test sets
* How to regenerate corrupted IoT test CSVs
* Exact image corruption types and severity schedules
* Exact IoT corruption types and severity schedules
* IoT imputation choice
* IoT dropout granularity and drift sign choices
* Which IoT columns were corrupted
* Which IoT columns were not corrupted
* Whether IoT corruption was applied in raw or scaled feature space
* Why corrupted images are saved as PNG
* Why corruption is applied after replicating the loader geometry
* The model input geometry used (size, crop, interpolation)
* Which checkpoints were used per model
* Which metric is primary
* How the severity-0 sanity check works
* How the passthrough (identity) sanity check works
* How to rerun evaluation
* How to read output files
* How to interpret Macro-F1 drop
* How bootstrap confidence intervals are computed (including paired bootstrap for drops)
* How corruption seeds and training seeds are used
* How E2 / E2-8 corrupted embeddings are regenerated from corrupted inputs (frozen E0/E1)
* The IoT corruption space used (feature-space vs raw) and why
* What the secondary train-on-corrupted scaffold is for, if created

---

## Implementation report requirements

Create:

```text
corruption_study/IMPLEMENTATION_REPORT.md
```

It must include:

* Files created
* Files modified
* Existing scripts reused
* Existing checkpoints found
* Selected checkpoint for each model
* Whether multiple training-seed checkpoints were available
* Clean-test metric found for each model
* Severity 0 reproduced metric for each model
* Passthrough (identity) reproduced metric for each model
* Any mismatch between existing clean metric and severity 0 / passthrough metric
* Whether any mismatch halted execution
* Corruption settings used
* Seeds used (corruption + training + bootstrap)
* Corrupted image size and corruption order
* Confirmation that images were saved as PNG
* The clean loader's geometric transforms (size, crop, interpolation)
* IoT feature space used for corruption
* Continuous IoT columns corrupted
* IoT columns left untouched
* IoT dropout granularity and drift sign used
* Whether bootstrap CIs were computed (and that drops used paired bootstrap)
* Whether multiple corruption seeds were used
* The E2 / E2-8 embedding-regeneration path used and the frozen encoder modules called
* Regenerated clean TEST embedding sanity-check metrics for E2-canonical and E2-8
* Confirmation that cached clean embeddings were NOT reused under corruption conditions
* The IoT corruption space (feature-space vs raw) decision
* Commands run
* Failed runs, if any
* Missing checkpoints, if any
* Assumptions made
* Next commands to run if anything remains unfinished

---

## Suggested CLI commands

Use command styles like these, adapting to the repo:

```bash
python corruption_study/gen/corrupt_images.py --input-root Data/test --output-root corruption_study/data/corrupted_images --image-size 224 --seed 42
```

```bash
python corruption_study/gen/corrupt_iot.py --train-csv Data/train.csv --test-csv Data/test.csv --output-root corruption_study/data/corrupted_iot --seed 42

# regenerate E2/E2-8 inputs by running corrupted images/IoT through the frozen E0/E1 encoders
python corruption_study/gen/regen_embeddings.py --images-root corruption_study/data/corrupted_images --iot-root corruption_study/data/corrupted_iot --output-root corruption_study/data/corrupted_embeddings --seed 42
```

```bash
python corruption_study/eval/run_corruption_eval.py --models e0 e1 e2_canonical e2_8 --severities 0 2 3 5
```

If exact paths differ, inspect the repo and adapt.

---

## Smoke test requirement

Before running the full evaluation:

1. Generate one small corrupted image subset or run on a tiny sample.
2. Generate one IoT corrupted CSV.
3. Run severity 0 evaluation for one model.
4. Run the passthrough (identity) check for one model.
5. Run severity 2 evaluation for one model.
6. Confirm severity 0 matches existing clean metric within tolerance.
7. Confirm passthrough matches existing clean metric within tolerance.
8. Confirm metrics JSON, predictions CSV, and confusion matrix are saved.
9. Confirm sample count matches clean test set.
10. Confirm no training is triggered.
11. Confirm output images are PNG.
12. Confirm corrupted images are at the model input geometry.
13. Confirm IoT corruption touches only continuous columns.

Only after smoke test passes, run the full evaluation.

---

## Definition of done

The task is complete when:

* Repo audit is printed and saved.
* Corrupted image generator works deterministically.
* Corrupted IoT generator works deterministically.
* Full materialized corrupted test tree is produced.
* Corrupted image sets preserve original folder structure and filename stems.
* Corrupted images are saved as PNG.
* JPEG corruption is applied in memory then saved as PNG.
* Image corruption is applied after replicating the clean loader geometry to model input size.
* Corrupted IoT CSVs preserve labels and sample ordering.
* IoT corruption only touches continuous sensor columns.
* Severity 0 uses original clean image root and original clean IoT CSV.
* Severity 0 mismatch beyond tolerance halts execution.
* Passthrough (identity) sanity check passes within tolerance.
* `run_corruption_eval.py` evaluates all required model Ã— condition Ã— corruption Ã— severity combinations.
* both_corrupted pairings are explicit.
* both_corrupted output names contain both image and IoT corruption type.
* One metrics JSON is saved per evaluation run.
* Per-sample predictions are saved for every run.
* Confusion matrices are saved.
* Bootstrap confidence intervals are computed, with drops using paired bootstrap.
* Stochastic corruptions use at least 3 corruption seeds.
* Training-seed checkpoint variance is reported if multiple seed checkpoints exist.
* Training seed is encoded in filenames when multiple seed checkpoints exist.
* `all_metrics.csv` is generated.
* `robustness_summary.csv` is generated.
* `robustness_summary.md` is generated.
* `per_class_recall_by_condition.csv` is generated.
* `README.md` explains reproduction and interpretation.
* `IMPLEMENTATION_REPORT.md` explains what was done and what remains.
* No training data was corrupted in the primary arm.
* No validation data was corrupted in the primary arm.
* No model was retrained in the primary arm.
* No E3 / VQC / quantum implementation was added.
* Evaluation uses `model.eval()` and deterministic settings.
* E2 / E2-8 corrupted conditions use embeddings regenerated from corrupted inputs via frozen E0/E1 (cached clean embeddings never reused under corruption).
* Regenerated clean TEST embeddings reproduce existing clean E2/E2-8 metrics within tolerance.
* Existing clean-trained weights/checkpoints/scalers/label mappings are reused; no replacement trained artifacts are created.
* IoT corruption space (feature-space vs raw) is decided, documented, and labeled honestly.
* MVP slice passed severity 0 + passthrough before full-matrix expansion.

---

## Do NOT

* Do not implement E3 / VQC in this phase.
* Do not claim quantum advantage.
* Do not corrupt train data in the primary arm.
* Do not corrupt validation data in the primary arm.
* Do not retrain existing models in the primary arm.
* Do not alter existing checkpoints.
* Do not corrupt on-the-fly per model.
* Do not use a non-fixed seed.
* Do not compute IoT imputation or scaling values from test data.
* Do not corrupt categorical / ID / label / metadata columns.
* Do not save corrupted images as JPEG.
* Do not corrupt full-resolution images and then downscale them.
* Do not apply geometric transforms to already-corrupted images at eval time.
* Do not compute drop confidence intervals with independent (unpaired) bootstrap.
* Do not let multiple training-seed runs overwrite each other.
* Do not silently overwrite old results.
* Do not break the existing repo structure.
* Do not make E2 look better by giving it different corrupted data than E0 or E1.
* Do not interpret clean accuracy as the main robustness result.
* Do not continue if severity 0 or passthrough fails to reproduce clean metrics beyond tolerance.
* Do not run all 7 x 3 both-corrupted pairings unless explicitly requested.
* Do not corrupt cached E2 / E2-8 embeddings directly (that is representation corruption, not input corruption).
* Do not evaluate E2 / E2-8 on stale clean embeddings under a corruption condition.
* Do not retrain just to regenerate E2 / E2-8 corrupted-condition embeddings; use the existing frozen encoders.
* Do not describe matched severity index as equivalent cross-modal severity.

---

## Final thesis interpretation

The study supports the P2 claim if:

* E2-canonical and/or E2-8 lose less Macro-F1 than E0 when images are corrupted.
* E2-canonical and/or E2-8 lose less Macro-F1 than E1 when IoT is corrupted.
* E2-canonical and/or E2-8 remain more stable when both modalities are corrupted.

The main result is not:

> Which model has the highest clean accuracy?

The main result is:

> Which model degrades more gracefully under corrupted test inputs?

Use Macro-F1 degradation, confidence intervals, and corruption-seed / training-seed summaries as the central evidence.


