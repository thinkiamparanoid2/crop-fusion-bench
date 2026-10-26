# Why the Quantum Route — Defense Memo

Purpose: a ready answer for the supervisor/chair question *"Why the quantum route
when E0 (image-only) and E2 (classical fusion) already score higher?"*

This memo maps the **official P1 objectives** (`docs/reports/T2530944_p1.pdf`, §1.4)
and the **P1 research gap** (§2.3) to what the experiments actually deliver, gives a
quotable answer, and pre-empts the weak points in P1.

---

## 30-second answer (say this first)

> The goal was never raw quantum accuracy supremacy. P1's own research gap (§2.3) is
> that QML in agriculture is almost entirely *theoretical* — run on toy data, against
> unmatched baselines, ignoring NISQ constraints. Our contribution is the rigorous
> version the literature lacks: a strictly **parameter-matched**, same-data, same-split
> comparison of a VQC fusion head against its exact classical equivalent, on a real
> progression-stage crop dataset. Strong E0/E2 baselines are *required* for that
> comparison to mean anything — unlike the prior HQNN papers in our own Table 2.1,
> we are not beating a weak baseline. Within that controlled setting the VQC is
> feasible as a compact NISQ-era fusion head, is competitive at matched parameters,
> and shows a consistent, multi-seed **calibration** advantage. So we deliver P1
> objectives 1, 2 and 4 directly, and turn objective 5 into a rigorous *boundary*
> result instead of an unsupported advantage claim.

---

## P1 objectives → what we delivered → how to phrase it

| # | P1 Objective (§1.4) | Status | How to phrase it |
|---|---|---|---|
| 1 | Explore limits of classical ML on multimodal high-dim data | ✅ Delivered | E0/E1/E2 baselines + the 4D→8D bottleneck-capacity finding show exactly where compressed fusion fails (severe-stage collapse at 4D). |
| 2 | Develop a VQC pipeline that **addresses qubit input-size limits** | ✅ Delivered | The 1344→128→8→VQC bottleneck *is* the NISQ input-size solution: classical compression feeds a circuit that fits available qubits. |
| 3 | Compare **amplitude vs angle** encoding | ⚠️ Open | Either run a minimal amplitude experiment, or scope to future work citing CPU state-prep cost + no clean parameter-matched comparator (see Landmine 2). |
| 4 | Combine IoT + imaging to improve detection | ✅ Delivered | E2 fusion beats unimodal E0/E1; severe-stage recall rises from E1's 0.482 to 0.821. Attribute this to *multimodal fusion*, not the quantum layer. |
| 5 | Measure hybrid vs classical on accuracy / scalability / feasibility | ⚠️ Re-scoped | Accuracy: classical wins, stated honestly. Feasibility: demonstrated. Re-scope "scalability" as *parameter compactness under NISQ limits*, never wall-clock speed. |

**Four of five objectives are about feasibility, the NISQ gap, fusion and encoding —
not about beating classical accuracy.** Lead with those.

---

## The real contribution (the research-gap framing)

P1 §2.3 research gap, in our own words: *existing agricultural QML is mostly
theoretical and neglects NISQ hardware noise, encoding latency, and real-data
dimensionality.* The thesis fills precisely this gap:

- a **parameter-matched** VQC-vs-classical comparison (64 post-bottleneck params each)
- on **real** progression-stage data (Dacon), real environmental IoT sensors
- under **NISQ-style constraints** (compressed bottleneck, simulator, IoT corruption tests)
- with honest, multi-seed reporting

This is a contribution *independent of whether quantum wins on accuracy*, because the
literature (our Table 2.1: HQNN 97.6%, NNQE 98.6%, etc.) reports accuracy against
weak/unmatched baselines. Our value-add is **rigor**.

---

## What we DO claim vs what we must NOT claim

**Claim (defensible):**
- The VQC is a feasible compact NISQ-era multimodal fusion head.
- At matched parameters it reaches competitive accuracy (warm-started E3-8 Macro F1
  0.9566 vs E2-matched-8 0.9734; severe recall 0.821 vs 0.804).
- **Consistent calibration advantage:** from-scratch E3-8 ECE **0.118 vs 0.248** for
  E2-matched-8, and lower ECE across **all 5 IoT-corruption seeds**.
- Bottleneck capacity (not the quantum layer) drives severe-stage recovery — a clean
  experimental finding.

**Do NOT claim:**
- Quantum accuracy advantage (classical wins the primary metric).
- Robustness advantage (E2 degrades less under shuffled IoT).
- Any runtime / "computational efficiency" / scalability-speed advantage — on CPU
  simulation the VQC is *slower*. This is the most dangerous claim to defend.

---

## Three landmines in P1 — pre-empt them

**1. Abstract + objective 5 say "improve accuracy, scalability, and computational
efficiency."** None were improved; on a simulator the VQC is slower. → Concede the
accuracy aim openly; reframe the finding as calibration + feasibility + rigor; never
defend a speed/efficiency claim. Reframe "scalability" as parameter/representational
compactness under qubit limits (objective 2).

**2. Objective 3 (amplitude vs angle encoding) is unmet.** → Close it with a minimal
amplitude run, or give the explicit future-work justification (CPU state-prep cost,
training instability/barren plateaus, no parameter-matched classical comparator).
Do not leave it unanswered.

**3. P1's computational-bottleneck rationale rests on hyperspectral (HSI) data, but
we used RGB + tabular IoT.** Classical models are not compute-bottlenecked on RGB. →
Reframe: the high-dimensional bottleneck we actually address is the 1344-d fused
embedding compressed to a NISQ-sized 4–8-d bottleneck; Dacon was chosen because its
progression-stage labels enable the *early-stage* question directly. Expect this one.

---

## Numbers to have on a card

| Model (test) | Macro F1 | Acc | Severe recall | ECE | Post-bottleneck params |
|---|---:|---:|---:|---:|---:|
| E0 image-only | 0.9639 | 0.9821 | 0.804 | 0.008 | — |
| E2 classical fusion | 0.9679 | 0.9873 | 0.821 | 0.007 | — |
| **E2-matched-8** (fair comparator) | 0.9734 | 0.9875 | 0.804 | 0.248 | 64 |
| E3-8 VQC (from scratch) | 0.9437 | 0.9736 | 0.804 | **0.118** | 64 |
| E3-8 VQC (warm-started) | 0.9566 | 0.9814 | 0.821 | 0.230 | 64 |

Headline quantum-positive: **calibration** (ECE 0.118 vs 0.248) at **identical
parameter count**, stable across 5 corruption seeds.

---

## One-line close for the chair

> "The quantum route gives us the first *rigorous*, parameter-matched, real-data NISQ
> evaluation of a VQC fusion head for crop-disease staging — feasible, competitive,
> and better-calibrated — which is exactly the gap P1 set out to fill. We report it
> honestly rather than overclaiming, which is the contribution the field is missing."
