# E3 Beat E2 Attack Plan

Date: 2026-06-05

**Goal:** beat `E2-matched-8` on **test Macro F1**.

Current bar and attack status:

```text
E2-matched-8 test Macro F1:            0.973356
Original E3-8 test Macro F1:           0.943716
Warm-start E3-8 seed 42 test Macro F1: 0.956587
```

This document is the focused execution plan for closing the gap between the parameter-matched classical comparator and the 8-qubit VQC head. Everything below targets the actual failure mode seen in saved results, not generic tuning.

---

## 1. Know exactly where we are losing

Current models are **tied on severe recall** (0.8036) and E3 is **already ahead on early recall** (0.9872 vs 0.9812).

The battle is **mid-stage**:

| Metric | E2-matched-8 | E3-8 | Gap |
|---|---:|---:|---:|
| Mid recall | **0.9364** | 0.8023 | **−13.4 pp** |
| Macro F1 | **0.9734** | 0.9437 | −3.0 pp |
| Accuracy | **0.9875** | 0.9736 | −1.4 pp |

Worst class examples from saved metrics:

- `3_a9_2`: E2 **0.748** vs E3 **0.216**
- `3_b3_1`: E2 **0.994** vs E3 **0.843**

**Rule:** every experiment below should target **mid-stage / stage-2 confusion**. Early and severe are not the primary problem at 8D.

Source artifacts:

```text
e3_bottleneck_sweep/results/e2_matched_8_metrics.json
e3_bottleneck_sweep/results/e3_8_vqc_metrics.json
e3_bottleneck_sweep/results/e2_matched_8_history.csv
e3_bottleneck_sweep/results/e3_8_vqc_history.csv
```

---

## 2. Highest-leverage move: warm-start E3 from E2

Current E3-8 trains **projection + VQC + readout from scratch on CPU**, while E2-matched-8 converged quickly on **CUDA** and peaked at epoch 11.

E2 and E3 share the **same projection and readout**. Only the middle differs (`Linear(8,8)` vs VQC).

### Stage A — transplant E2 weights

1. Load `e3_bottleneck_sweep/checkpoints/best_e2_matched_8.pt`
2. Copy into E3-8:
   - `projection.*`
   - `readout.*`
3. Keep VQC randomly initialized

### Stage B — train VQC only (10–20 epochs)

- Freeze projection + readout
- Optimizer only on VQC params
- VQC learning rate: `1e-2` or `5e-3`

### Stage C — joint fine-tune (20–40 epochs)

- Unfreeze all
- Parameter groups:
  - projection/readout: `lr=1e-4`
  - VQC: `lr=5e-4` or `1e-3`
- Checkpoint on **validation Macro F1**

**Why this is priority #1:** E2 already solved most of the representation. E3 only needs to replace the 8→8 classical block without destroying mid-stage separation.

---

## 3. Fix training instability

From `e3_8_vqc_history.csv`, E3 validation mid recall is volatile:

- epoch 7: 0.928
- epoch 8: 0.747
- epoch 10: 0.961
- epoch 11: 0.745

E2-matched-8 is smoother and peaks earlier.

### Training changes for all new E3 runs

| Setting | Current E3-8 | Try next |
|---|---|---|
| Device | CPU (`lightning.qubit`) | Keep CPU if no GPU sim; reduce batch |
| Batch size | 512 | **128 or 256** |
| Epochs | 60 | **120** |
| Patience | 8 | **15–20** |
| LR | 1e-3 global | **split LRs** (projection vs VQC) |
| Optimizer | Adam | **AdamW** or **SPSA on VQC only** |
| Seeds | 42 only | **42, 13, 37, 73** — keep best val Macro F1 |

### Dual checkpoint rule

Epoch 15–16 had **better val mid/severe** than epoch 14 but lower Macro F1. Save two checkpoints:

```text
best by val_macro_f1
best by val_mid_stage_recall
```

Evaluate both on test; keep whichever has higher **test Macro F1**.

---

## 4. Mid-stage targeted loss (not blind focal)

Focal at 4D destroyed accuracy. At 8D, use a **surgical** loss:

```text
L = weighted_CE + λ * mid_stage_CE
```

Where `mid_stage_CE` is cross-entropy on samples whose true label ends in `_2` (mid stage).

Start with `λ ∈ {0.5, 1.0, 2.0}`.

**Fairness rule:** run the **same loss** on both models:

- `E2-matched-8 + mid-aware loss`
- `E3-8 + mid-aware loss`

Goal: lift E3 mid recall **above** E2, not just above the old E3 baseline.

Do **not** claim victory from early-recall-only gains if Macro F1 still loses.

---

## 5. VQC architecture search at 8D (matched params)

Current E3-8 uses **4 VQC layers** (64 params, matched to `Linear(8,8)`).

Run a small grid. For every E3 config, train the **parameter-matched** E2-matched-8 comparator with the same bottleneck width and post-bottleneck param budget.

| Experiment | Change | Why |
|---|---|---|
| **E3-8-warm** | warm-start from E2 | highest priority |
| **E3-8-emb** | AngleEmbedding on **RX+RY** instead of X-only | richer input encoding |
| **E3-8-line** | linear CNOT chain vs ring | less symmetric entanglement |
| **E3-8-RZ** | add RZ gates with matched classical comparator | more expressivity |
| **E3-8-adamw** | same ansatz, AdamW + split LR | training stability |

Do **not** jump to 12 qubits until warm-start + training fixes are exhausted. The current gap looks like **optimization/representation**, not missing capacity alone.

---

## 6. Bottleneck 10 or 12 only if 8D plateaus

If after warm-start + mid-loss + training fixes E3 is within **~0.005–0.01 Macro F1** of E2, expand:

```text
E2-matched-10 vs E3-10
E2-matched-12 vs E3-12
```

More capacity can help mid-stage separation. Run this **after** fixing training, not before.

---

## 7. Execution order

### Phase 1 — close the gap

1. Warm-start E3-8 from E2-matched-8 (Stages A→B→C)
2. Retrain with batch 128, epochs 120, patience 20, split LR
3. At least 3 seeds: 42, 13, 37
4. Dual checkpoint: best Macro F1 + best mid recall

**Update (2026-06-06):** seed 42 warm-start is complete. It improved E3-8 Macro F1 from 0.943716 to 0.956587, but did not beat E2-matched-8 at 0.973356. Additional plain warm-start seeds are postponed because seed 42 took about 8 hours. The next active run is pair-targeted warm-start for `3_a9_2 -> 3_a9_1`.

### Phase 2 — if still behind

5. **Diagnostic-targeted runs** (Section 13.2): progression-pair loss, residual shortcut, classical VQC init, hard-example replay
6. Mid-stage weighted loss (paired E2/E3)
7. 8D paired focal — one run max (Section 13.3.1)
8. Encoding/ansatz variants (2–3 configs max)
9. **E3-8-reupload vs E2-matched-8-deep** (see Section 12)
10. Re-scan training histories for missed peaks

### Phase 3 — only if needed

11. Bottleneck 10/12 matched sweep
12. SPSA on VQC block
13. Problem-specific ansatz ablation (Section 11.2)
14. Tier 2 architecture gaps (Section 13.4): deeper readout, QNG, layer-wise VQC

### Phase 4 — post-thesis / future work only

15. Amplitude encoding tier (Section 11.3)
16. Real-hardware sanity check
17. Out-of-scope upstream changes (Section 13.5)

---

## 8. What “beating E2” means

Primary win condition:

```text
test Macro F1(E3) > test Macro F1(E2-matched-8)
current bar: 0.973356
```

Fairness constraints (non-negotiable):

- same frozen embeddings (`e2_multimodal_fusion/features/*_embeddings.npz`)
- same split and 25-class order
- matched post-bottleneck parameter count
- if loss changes, apply the **same loss family** to both E2 and E3

Secondary metrics (ECE, early recall) are supporting evidence only. They do **not** count as beating E2 unless Macro F1 also wins.

---

## 9. Fast diagnostic before new training

Compare prediction files:

```text
e3_bottleneck_sweep/results/e2_matched_8_test_predictions.csv
e3_bottleneck_sweep/results/e3_8_vqc_test_predictions.csv
```

Count:

1. samples E2 correct, E3 wrong → filter `stage == 2`
2. top confused class pairs for E3 only
3. whether errors are **mid→early** or **mid→severe**

That tells us whether the next run should emphasize warm-start, mid-loss, or ansatz change.

**Update (2026-06-05):** diagnostic complete — see **Section 13.1**. Primary failure mode is **`3_a9_2 → 3_a9_1`** (85 cases). Next runs should add progression-pair loss (Section 13.2.1) after full warm-start.

---

## 10. One-sentence strategy

**Do not train E3 from scratch again. Transplant E2’s solved projection/readout, then optimize only the quantum 8→8 block with stable VQC training aimed at mid-stage recall.**

That is the most direct path from **0.9437 → 0.973+**.

---

## 11. Advanced VQC directions (thesis vs future work)

The current E3-8 circuit uses **one-shot angle encoding** (X rotations only), a **hardware-efficient ansatz** (RX/RY + CNOT ring), and **Pauli-Z readout**. That is a weak but fair first comparator. If Phase 1–2 plateau below E2 Macro F1, these are the advanced quantum upgrades — ranked by **thesis feasibility** and **likelihood of helping mid-stage separation**.

### 11.1 Data re-uploading — **best advanced quantum bet (Phase 2)**

**Idea:** feed the same 8D bottleneck into the circuit **multiple times**, interleaved with variational layers:

```text
encode(x) -> VQC layer -> encode(x) -> VQC layer -> measure
```

**Why it may help us:**

- Current circuit sees 8D input **once**; mid-stage boundaries (`_2` labels) may need stronger nonlinear processing of the same compressed features.
- Increases expressivity **without more qubits**.
- Pairs naturally with **warm-start from E2-matched-8** (Section 2).

**Thesis fit:** Good — if we build a **parameter-matched classical deep comparator** (Section 12).

**Risk:** deeper circuits train slower, gradients may vanish if too many re-upload blocks.

**Verdict:** Run **E3-8-reupload** after warm-start + stable training. Full spec in Section 12.

### 11.2 Problem-specific ansatz — **secondary ablation (Phase 3)**

**Idea:** replace generic RX/RY + ring with task-shaped connectivity, e.g.:

- line vs ring vs all-to-all CNOT
- split 8 bottleneck dims into image/IoT groups and entangle across groups
- add ZZ or CRZ interactions

**Why it may help:** mid-stage confusion (`3_a9_2`, etc.) is task-specific; generic ansätze may underfit progression boundaries.

**Thesis fit:** Medium — must still **match trainable params** and justify connectivity choice.

**Verdict:** One ablation after re-uploading, not the main path.

### 11.3 Amplitude encoding — **future work (Phase 4)**

**Idea:** encode up to `2^n` values in `n` qubit amplitudes (8 qubits → 256D state in principle), bypassing a hard 8D bottleneck.

**Why it is attractive:** more information into the quantum block without 256 physical qubits.

**Why not for current thesis sprint:**

- state preparation is expensive and must preserve normalization
- training is harder (barren plateaus, unstable gradients)
- **no simple E2-matched comparator** — fair comparison needs a new classical baseline definition
- simulation cost on CPU is high
- P1 listed angle vs amplitude (E11) but repo has not run it

**Verdict:** Strong **post-thesis** direction. Mention in Discussion/Future Work; do not rely on it to beat E2-matched-8 before submission.

### 11.4 Recommended stack order

```text
1. Warm-start E3 from E2-matched-8          (Section 2 — classical optimization)
2. Stable VQC training                      (Section 3)
3. Mid-stage targeted loss                  (Section 4)
4. Data re-uploading at 8D                  (Section 12 — best quantum-native upgrade)
5. Problem-specific ansatz                  (Section 11.2)
6. Amplitude encoding / large bottleneck    (Section 11.3 — new research tier)
```

---

## 12. E3-8-reupload experiment spec

Primary advanced-quantum experiment to run if Phase 1 warm-start does not reach E2 Macro F1.

### 12.1 Models

**E3-8-reupload (quantum)**

```text
shared projection : Linear(1344,128) -> BN -> ReLU -> Dropout(0.1) -> Linear(128,8)
middle            : [AngleEmbedding(x) -> VQC block] x R  (R = 2 or 3 re-upload rounds)
readout           : Linear(8,25)
```

Suggested re-upload block (one round):

```text
AngleEmbedding(x, wires=0..7, rotation="X")   # or RX+RY for richer encoding
for layer in 1..L:
    RX/RY trainable per qubit
    CNOT ring (or line — ablation)
return Pauli-Z expectations on 8 qubits
```

Full middle = chain `R` blocks, re-encoding the **same** scaled bottleneck `x = tanh(projection) * pi` each round.

**E2-matched-8-deep (classical comparator)**

Replace single `Linear(8,8,bias=False)` with a **parameter-matched** deep block, e.g.:

```text
tanh -> Linear(8,8,bias=False) -> tanh -> Linear(8,8,bias=False) -> ...  (R times)
```

Or an equivalent stacked block with **the same post-bottleneck trainable param count** as E3-8-reupload (report `total_trainable_params` and `post_bottleneck_trainable_params` in every JSON).

Naming:

```text
E2-matched-8-deep
E3-8-reupload
```

### 12.2 Parameter matching rule

Same as `e2_multimodal_fusion/PARAM_MATCH_SPEC.md`:

- frozen E0/E1 encoders do not count
- shared projection + readout must match E3-8 baseline structure
- post-bottleneck params: E3-8-reupload vs E2-matched-8-deep within ±10%
- document exact gate/param counts in results JSON (`architecture`, `reupload_rounds`, `vqc_layers_per_round`)

Example budget at 8 qubits (illustrative — verify in code before training):

```text
1 re-upload round, 4 VQC layers, RX+RY per qubit: 4 * 8 * 2 = 64 params per round
R = 2 rounds -> 128 post-bottleneck params
Match E2 with two stacked Linear(8,8,bias=False) blocks -> 8*8*2 = 128 params
```

Adjust `R` and layers so counts match exactly before any claim.

### 12.3 Initialization (required)

Do **not** train re-upload from scratch.

1. Load `best_e2_matched_8.pt`
2. Copy `projection.*` and `readout.*` into both E3-8-reupload and E2-matched-8-deep
3. Initialize new middle blocks randomly (VQC) or small random (classical stack)
4. Stage B: train middle only (10–20 epochs)
5. Stage C: joint fine-tune with split LR (Section 2)

### 12.4 Training config

Use Section 3 defaults unless noted:

```text
batch size:     128 or 256
epochs:         120
patience:       20
seeds:          42, 13, 37 (minimum)
simulator:      lightning.qubit
checkpoint:     val_macro_f1 (also save val_mid_stage_recall checkpoint)
loss:           weighted_CE, or mid-aware loss (Section 4) if Phase 2 step 5 already ran
```

Optional combined run:

```text
E3-8-reupload + warm-start + mid_stage_CE (λ=1.0)
vs
E2-matched-8-deep + warm-start + mid_stage_CE (λ=1.0)
```

### 12.5 Metrics to report

Same as all E2/E3 comparisons (`common/thesis_eval.py`):

```text
accuracy, macro_f1, macro_recall
early_stage_recall, per_stage_recall (especially stage 2)
ece, confusion_matrix
total_trainable_params, post_bottleneck_trainable_params
reupload_rounds, vqc_layers_per_round, encoding type
```

Primary win condition unchanged: **test Macro F1 > 0.973356**.

### 12.6 Acceptance criteria

Experiment succeeds (technically) if:

```text
shapes correct: VQC receives [B,8], logits [B,25]
param match documented in JSON
trains without NaN/shape errors
comparison table E2-matched-8-deep vs E3-8-reupload generated
```

Experiment succeeds (scientifically) if:

```text
test Macro F1(E3-8-reupload) > test Macro F1(E2-matched-8)   [or E2-matched-8-deep baseline]
AND mid_stage_recall(E3) >= mid_stage_recall(E2) - 0.005
```

Do not claim beat if only ECE or early recall improves.

### 12.7 Suggested implementation layout

```text
e3_bottleneck_sweep/
  quantum_layer_reupload.py    # VQC with R re-upload blocks
  models_reupload.py           # E3VQCKReupload, E2MatchedKDeep
  scripts/train_reupload.py    # CLI mirroring train_bottleneck.py
  results/
    e3_8_reupload_metrics.json
    e2_matched_8_deep_metrics.json
```

Reuse:

```text
e2_multimodal_fusion/features/*_embeddings.npz
common/thesis_eval.py
e3_bottleneck_sweep/checkpoints/best_e2_matched_8.pt
```

### 12.8 Thesis wording if re-upload wins

> Under an parameter-matched 8-dimensional bottleneck, a re-uploading VQC fusion head exceeded the matched deep classical comparator on test Macro F1, suggesting that repeated quantum feature encoding can improve progression-stage separation when initialized from a strong classical fusion baseline.

If re-upload does **not** beat E2, report as:

> Re-uploading increased circuit expressivity but did not close the Macro F1 gap; calibration advantages of the simpler E3-8 architecture remained.

---

## 13. Diagnostic-driven possibilities and gap checklist

Section 9 asked for a prediction-level diagnostic before new training. That diagnostic is **done** and narrows the problem substantially. This section records what it found, what the attack plan already covers, and what is still missing.

### 13.1 What the diagnostic found

Run: `e3_beat_e2_attack/scripts/diagnose_mid_stage_errors.py`

Report: `e3_beat_e2_attack/results/diagnostics/MID_STAGE_ERROR_DIAGNOSTIC.md`

Key findings on the **test set**:

| Finding | Detail |
|---|---|
| Mid-stage gap | E2 correct / E3 wrong: **64** samples (stage 2) |
| Error direction | **85 of 87** E3 mid errors predicted as **early** (stage 1), only **2** as severe |
| Dominant confusion | **`3_a9_2 → 3_a9_1`**: **85** cases (almost the entire mid failure) |
| Secondary confusions | `3_b3_1 → 5_b6_1` (25), `3_a9_1 → 3_a9_2` (13) — mostly early-stage, not mid |

**Interpretation:** E3 is not failing mid-stage randomly. It is **collapsing mid → early** on a **single progression pair** within crop 3. Generic mid-stage loss (Section 4) is directionally correct but may be too blunt; the highest-ROI fix is **progression-pair targeting** for `3_a9_2` vs `3_a9_1`.

### 13.2 Progression-pair and hard-class experiments (Tier 1 — add now)

These are **not yet in Sections 2–12** but are justified by Section 13.1.

#### 13.2.1 Class-pair weighted loss

Add an extra term when the model confuses adjacent progression labels:

```text
L = weighted_CE + λ_mid * mid_stage_CE + λ_pair * pair_CE
```

Where `pair_CE` is cross-entropy on samples with true label `3_a9_2`, with an additional penalty when predicted label is `3_a9_1` (or vice versa for the reverse confusion).

Suggested starting weights:

```text
λ_pair ∈ {1.0, 2.0, 5.0}
λ_mid ∈ {0.5, 1.0}   (keep Section 4 term if already running)
```

**Fairness rule:** apply the **same pair weighting** to both:

```text
E2-matched-8 + pair-aware loss
E3-8 + pair-aware loss
```

**Why:** 85/87 mid errors share one label pair. A generic mid-stage boost may not fix a boundary that is already sharp for E2 but blurred for E3.

#### 13.2.2 Hard-example replay / oversampling

During training, oversample:

- all stage-2 labels (`*_2`), and/or
- crop-3 progression labels (`3_a9_*`, `3_b3_*`, …), and/or
- the 64 samples in `mid_stage_e2_correct_e3_wrong.csv`

Use the same sampler for E2 and E3 comparators.

#### 13.2.3 Stage-aware auxiliary head (multi-task)

Add a small auxiliary classifier on the 8D bottleneck:

```text
bottleneck (8D) -> stage head (4 classes: healthy/early/mid/severe)
bottleneck (8D) -> VQC -> readout (25 classes)
L = CE_25 + α * CE_stage
```

**Fairness:** add the **same auxiliary head** to E2-matched-8 (on the 8D middle output before readout).

**Why:** mid recall is a stage-boundary problem; an explicit stage signal may stabilize the VQC block without changing frozen encoders.

#### 13.2.4 VQC init from classical middle (identity-ish start)

Warm-start (Section 2) copies projection + readout only. Also try:

1. Load E2 `Linear(8,8,bias=False)` weights
2. Initialize VQC rotations so the first forward pass approximates the classical map (small angles near identity, or layer-wise classical-to-quantum transfer)
3. Then run Stage B (VQC-only) and Stage C (joint)

**Why:** E3 should **start as E2** in the middle block and only deviate where quantum helps — not learn a random 8→8 map from scratch.

#### 13.2.5 Residual bottleneck shortcut

Add a ResNet-style shortcut around the post-bottleneck decision block:

```text
E2 residual: readout(x + Linear(8,8)(x))
E3 residual: readout(x + VQC(x))
```

where `x` is the 8D bottleneck feature entering the middle block. This targets the
`3_a9_2 -> 3_a9_1` collapse by preserving the projected bottleneck signal that E2
already separates well, while the middle block learns a correction instead of
having to re-transmit all information through the VQC.

Fairness rule:

```text
Do not run only E3 residual.
Run E2-matched-8-residual with the same shortcut and compare against it.
```

Scaling rule:

```text
Use a bounded shortcut input, e.g. tanh(x), or a learnable residual scale alpha.
Avoid raw unbounded x + VQC(x), because Pauli-Z VQC outputs are bounded in [-1, 1]
and can be drowned out by an unconstrained classical bottleneck.
```

Suggested first variant:

```text
readout(tanh(x) + alpha * middle(tanh(x)))
alpha initialized to 0.1 or 1.0, reported in the architecture notes
```

**Update (2026-06-06):** the paired free-alpha residual diagnostic completed. It did **not** beat the
E2 bar:

```text
E2 residual diag best Macro F1: 0.972224, alpha_final ~= 0.8076
E3 residual diag best Macro F1: 0.961233, alpha_final ~= 13.6115
E3 residual diag best mid recall: 0.915909
```

The free E3 residual scalar grew far beyond the intended shortcut scale. The next residual run should
therefore use the paired bounded-alpha variant:

```text
--residual --alpha-bounded --alpha-max 1.0
```

This keeps one trainable scalar in both models, preserves the matched 65 post-bottleneck parameter
count, and prevents the shortcut branch from drowning the bounded VQC output.

**Update (2026-06-07):** the bounded-alpha residual diagnostic completed:

```text
E2 bounded residual diag best Macro F1: 0.973469, alpha_final ~= 0.7114
E3 bounded residual diag best Macro F1: 0.966532, alpha_final ~= 0.9999
E3 bounded residual diag best mid recall: 0.929545
```

This is the best E3 attack result so far and closes most of the warm-start gap, but it still does not
beat the original E2 bar (`0.973356`) or the paired bounded-residual E2 comparator (`0.973469`).
The bounded E3 alpha saturated near `1.0`, so this path may need a slightly larger bounded scale or
to be combined with the progression-pair loss rather than repeated as-is.

**Why:** this is cheaper than re-uploading and directly tests whether E3's
mid-stage loss is caused by the VQC block erasing bottleneck information rather
than by a lack of useful projected features.

### 13.3 Training recipe gaps (Tier 1–2)

| # | Possibility | In plan before §13? | Priority |
|---|---|---|---|
| 1 | Full warm-start seed 42 | Yes (§2) — **complete; improved but did not beat E2** | **Done / baseline for next attack** |
| 2 | Progression-pair loss (`3_a9_2 → 3_a9_1`) | No | **Tier 1** |
| 3 | VQC init from classical `Linear(8,8)` | No | **Tier 1** |
| 4 | Residual bottleneck shortcut | Added in §13.2.5 | **Tier 1** |
| 5 | Focal / weighted-focal at 8D (paired E2/E3) | Discussed; not in doc | **Tier 1** |
| 6 | E3-8-reupload | Yes (§12) | **Tier 1** |
| 7 | Composite checkpoint metric (`macro_f1 + α * mid_recall`) | Dual save only (§3) | Tier 2 |
| 8 | Embedding noise augmentation (train time) | No | Tier 2 |
| 9 | IoT corruption during training (not just eval) | Eval only (E8) | Tier 2 |
| 10 | Label smoothing | No | Tier 2 |
| 11 | Knowledge distillation from E2 logits | No | Tier 2 |

#### 13.3.1 Focal / weighted-focal at 8D

At 4D, weighted focal boosted early recall but hurt accuracy (`e3_early_rescue/`). At 8D the failure mode is **mid → early**, not early miss. Still worth **one paired run**:

```text
E2-matched-8 + weighted_focal(γ=2)
E3-8 + weighted_focal(γ=2)
```

Use class weights from training split (same as current weighted CE). Do **not** use focal alone without the E2 comparator.

#### 13.3.2 Composite checkpoint objective

Section 3 saves two checkpoints separately. Also try selecting checkpoints by:

```text
score = val_macro_f1 + α * val_mid_stage_recall
α ∈ {0.05, 0.10, 0.20}
```

Evaluate on test Macro F1; keep α that maximizes test Macro F1, not val mid alone.

#### 13.3.3 Knowledge distillation (optional)

```text
L = CE(y, ŷ_E3) + β * KL(softmax(E2/τ), softmax(E3/τ))
```

Use frozen E2-matched-8 as teacher. Apply distillation **only to E3** training; E2 baseline stays unchanged. Document clearly as an E3-only training aid, not a matched comparator change.

### 13.4 Architecture and quantum gaps (Tier 2–3)

| # | Possibility | In plan? | Priority |
|---|---|---|---|
| 11 | Deeper readout `Linear(8,32)→ReLU→Linear(32,25)` vs deeper VQC | Partial (§12 deep classical) | Tier 2 |
| 12 | Pauli-X/Y or multi-basis measurement | No | Tier 2 |
| 13 | IQP / Hamiltonian / complex encodings | No | Tier 2 |
| 14 | Quantum natural gradient (QNG) on VQC | SPSA only (§3) | Tier 2 |
| 15 | Layer-wise VQC training (one layer at a time) | No | Tier 2 |
| 16 | Identity-block / barren-plateau init | No | Tier 2 |
| 17 | Modality-split bottleneck (4 image + 4 IoT dims) | Mentioned (§11.2) | Tier 3 |
| 18 | Amplitude encoding | Yes (§11.3) | Tier 3 / future |
| 19 | Real hardware + noise mitigation | Yes (§7 Phase 4) | Tier 3 / future |

### 13.5 Out-of-scope but real possibilities (Tier 3 — changes the experiment)

These can improve absolute performance but **break or weaken** the current fair E2/E3 head comparison if applied without a new baseline definition:

| # | Possibility | Notes |
|---|---|---|
| 20 | Re-extract embeddings (new E0/E1) | New frozen features → re-run all E2/E3 |
| 21 | End-to-end encoder fine-tune | Master plan Stage 2; not head-only |
| 22 | AgriFusionNet tier (E4/E5) | Different architecture entirely |
| 23 | Multi-seed **data splits** (13, 37, 73) | Changes dataset protocol |
| 24 | Beat `E2-current` instead of `E2-matched-8` | Easier bar; different claim |
| 25 | Ensemble E2 + E3 | Not a single fair model |
| 26 | Post-hoc calibration only (temperature scaling) | ECE win ≠ Macro F1 win |

### 13.6 Implementation status (`e3_beat_e2_attack/`)

**Current status update (2026-06-08):** full warm-start seed 42 is complete. `e3_8_warm_s42` improved E3-8 Macro F1 from 0.943716 to 0.956587, but did not beat E2-matched-8 at 0.973356. The paired free-alpha residual diagnostic improved E3 to 0.961233 but let E3 alpha grow to about 13.6. The paired bounded-alpha residual diagnostic improved E3 to 0.966532. The combined bounded-residual + pair-loss diagnostic is now the best E3 result at 0.968655, but it still does not beat the original E2 bar. The pair-loss component improves mid recall, but the remaining top error is still `3_a9_2 -> 3_a9_1`.

```text
e3_beat_e2_attack/
  scripts/diagnose_mid_stage_errors.py   # DONE — produced §13.1 findings
  scripts/train_warmstart_e3.py          # DONE — warm-start + pair-loss + residual/bounded-alpha CLI
  scripts/compare_attack_runs.py         # DONE — reports completed attack metrics and alpha mode
  results/diagnostics/                   # DONE
  results/ATTACK_RUN_COMPARISON.md       # DONE — includes e3_8_warm_s42 best_macro/best_mid
```

**Current gap:** Phase 1 warm-start, pair-loss, free-alpha residual, bounded-alpha residual, and combined bounded-residual + pair-loss all have seed `42` completed, but the E2 bar is still unbeaten by E3. Additional plain warm-start seeds are postponed because seed 42 took about 8 hours. The next execution step should be a small targeted sweep, not a repeat of the same command: either tune pair-loss strength under bounded residual or try a larger bounded residual scale.

Suggested smoke vs full defaults (`train_warmstart_e3.py`):

| Setting | Smoke | Full run |
|---|---|---|
| `vqc_only_epochs` | 1 | 15 |
| `joint_epochs` | 1 | 60 |
| `batch_size` | 64 | 128 |
| `patience` | 2 | 20 |

### 13.7 Updated execution order (includes §13)

Insert **after Phase 1 step 4** if warm-start alone does not beat E2:

```text
Phase 1b — diagnostic-targeted (if mid recall still < E2 after warm-start)
  5. Progression-pair loss (§13.2.1) — prioritize 3_a9_2 vs 3_a9_1
  6. Residual bottleneck shortcut (§13.2.5) — preserve the 8D bottleneck signal
  7. VQC init from classical Linear(8,8) (§13.2.4)
  8. Hard-example replay (§13.2.2)
  9. 8D paired focal (§13.3.1) — one γ=2 run max
```

Renumber existing Phase 2–4 steps accordingly. Minimum order:

```text
Phase 1  — warm-start + stable training + multi-seed + dual checkpoint     (§2, §3)
Phase 1b — pair loss + residual shortcut + classical VQC init + hard replay + optional focal  (§13)
Phase 2  — mid-stage loss + ansatz grid + re-upload                       (§4, §5, §12)
Phase 3  — bottleneck 10/12, SPSA, problem-specific ansatz, Tier 2 arch  (§6, §11, §13.4)
Phase 4  — amplitude encoding, hardware                                   (§11.3, §7)
```

### 13.8 Coverage summary — are we leaving anything out?

**Current status update (2026-06-06):** full warm-start is no longer “not run yet.” Seed 42 completed and improved Macro F1, but did not beat E2. Pair-loss is implemented and smoke-tested; the full pair-loss run is the next execution step.

| Bucket | Status |
|---|---|
| **Completed** | Diagnostic (`3_a9_2 -> 3_a9_1`), full warm-start seed 42, pair-loss paired run, free-alpha residual paired diagnostic, bounded-alpha residual paired diagnostic, combined bounded-residual + pair-loss paired diagnostic, attack-run comparison |
| **Implemented / smoke-tested, not complete** | Alpha-max and pair-loss-strength sweeps via existing flags |
| **Not run yet** | Small alpha-max / pair-loss-strength sweep, re-upload, 8D focal |
| **Possible but not primary** | Classical VQC init, hard replay, composite checkpoint |
| **In doc, partially explored** | Mid-loss, ansatz grid, bottleneck sweep |
| **Real but out of fair-comparison scope** | Encoder retrain, Tier B, hardware, different splits |

**Bottom line:** the attack plan is strong on generic quantum and optimization levers. Section 13 closes the main scientific gap: **E3 loses mid-stage almost entirely via one progression collapse (`3_a9_2 → 3_a9_1`)**. Target that pair explicitly before declaring the search exhausted.

### 13.9 Suggested experiment names

```text
e3_8_warm_s42              # Phase 1 warm-start (§2)
e3_8_warm_pair_s42         # warm-start + pair_CE (§13.2.1)
e3_8_warm_residual_s42     # warm-start + residual bottleneck shortcut (§13.2.5)
e2_matched_8_residual_s42  # paired residual classical comparator (§13.2.5)
e3_8_warm_residual_bounded_diag_s42     # bounded-alpha residual diagnostic (§13.2.5)
e2_matched_8_residual_bounded_diag_s42  # paired bounded-alpha residual diagnostic (§13.2.5)
e3_8_warm_classinit_s42    # warm-start + classical middle init (§13.2.4)
e3_8_warm_focal8_s42       # warm-start + 8D weighted focal (§13.3.1)
e3_8_reupload_s42          # Phase 2 re-upload (§12)
e2_matched_8_deep_s42      # paired comparator for re-upload
```

Write all results under `e3_beat_e2_attack/results/` and update `ATTACK_RUN_COMPARISON.md` via `compare_attack_runs.py`.

---

## Related docs

```text
docs/by_stage/07_final_claim_revision/FINAL_CLAIM_REVISION.md
docs/by_stage/07_final_claim_revision/THESIS_RESULTS_NARRATIVE_DRAFT.md
docs/by_stage/05_e3_rescue_bottleneck_robustness/E3_4_vs_E3_8_BOTTLENECK_COMPARISON.md
e3_bottleneck_sweep/scripts/train_bottleneck.py
e3_early_rescue/E3_EARLY_RESCUE_COMPARISON.md
e2_multimodal_fusion/PARAM_MATCH_SPEC.md
e3_beat_e2_attack/results/diagnostics/MID_STAGE_ERROR_DIAGNOSTIC.md
e3_beat_e2_attack/scripts/train_warmstart_e3.py
```
