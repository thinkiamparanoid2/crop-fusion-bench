# Amplitude Encoding — Scoping & Future-Work Justification

Closes P1 Objective 3 (*"examine and compare approaches to quantum feature encoding,
e.g. amplitude and angle encoding"*). The thesis implements and characterises **angle
encoding** in depth (the entire E3 line); **amplitude encoding** is scoped to future
work for the reasons below. Pairs with `WHY_QUANTUM_DEFENSE_MEMO.md`.

---

## Status of Objective 3

- **Angle encoding — done.** RX angle encoding (bounded `theta = pi*tanh(z)`),
  hardware-efficient RX/RY ansatz, CNOT-ring entanglement, Pauli-Z readout, evaluated
  across 4D and 8D bottlenecks, multiple seeds, and IoT-corruption conditions.
- **Amplitude encoding — deferred,** with the technical justification below.

So Objective 3 is *partially addressed*: one encoding is fully characterised; the
second is deferred for principled, stated reasons rather than omitted silently.

---

## Thesis-ready prose (paste into Discussion / Future Work)

> Our hybrid model uses **angle encoding**, in which each value of the learned
> low-dimensional fusion bottleneck is mapped to a single-qubit rotation. We
> deliberately retain angle encoding throughout this work because it is
> hardware-efficient, uses shallow state preparation, and — critically — admits a
> **parameter-matched classical comparator**, which is the basis of our controlled
> E2-vs-E3 evaluation. **Amplitude encoding**, which packs up to `2^n` real values
> into the amplitudes of an `n`-qubit state, is left to future work for three reasons.
>
> First, in our architecture the high-dimensional-to-low-dimensional reduction is
> already performed **classically**, by the frozen encoders and the learned
> `1344 -> 128 -> 8` projection. Amplitude encoding's headline benefit — compressing a
> high-dimensional vector into few qubits — is therefore redundant at our 8-dimensional
> bottleneck: there is no high-dimensional vector left to pack at the point the circuit
> receives its input. Its advantage would only materialise in a different architecture
> that feeds a much larger (e.g. full 1344-dimensional) vector directly into the
> quantum layer, which is a separate research design.
>
> Second, amplitude encoding **breaks the parameter-matched comparison** that gives our
> study its rigour. Angle encoding has a clear classical analogue (the matched
> `Linear(8,8)` decision block), letting us attribute any difference solely to the
> quantum substitution. Amplitude encoding has no comparably clean classical
> counterpart and requires its own baseline definition; introducing it without that
> baseline would reintroduce exactly the unmatched-comparison weakness we identify in
> the prior literature (P1 §2.3, Table 2.1).
>
> Third, amplitude encoding is **costly and less NISQ-faithful** under our constraints.
> Exact amplitude state preparation generally needs circuit depth that grows with the
> state dimension and assumes normalised inputs, which conflicts with the shallow,
> hardware-efficient, NISQ-oriented design this thesis targets; it is also markedly
> more expensive to simulate. Our quantum layer runs only on a CPU state-vector
> simulator (no `lightning.gpu` was available in our environment), where even the
> shallow angle-encoded circuit dominates training time. A fair multi-seed
> amplitude-encoding study was therefore not feasible within the project's compute
> budget.
>
> Future work should evaluate amplitude encoding (and IQP / re-uploading feature maps)
> in an architecture that routes a higher-dimensional fused vector directly into the
> circuit, paired with a newly defined classical comparator and, ideally, GPU-backed
> or hardware-backed execution so that state-preparation cost is properly accounted
> for.

---

## Defense Q&A — if the chair pushes

**Q: "P1 said you'd compare amplitude and angle encoding. Where is amplitude?"**
> We fully characterised angle encoding and made a deliberate, documented decision to
> defer amplitude encoding. The core reason is scientific, not just compute: our
> dimensionality reduction is already done classically before the circuit, so at an
> 8-dimensional bottleneck amplitude encoding's main advantage doesn't apply — there's
> nothing high-dimensional left to pack. And it would break our parameter-matched
> comparison, which is the whole point of the study. We scope it as future work in an
> architecture where it can be tested fairly.

**Q: "Isn't that just avoiding the hard part?"**
> No — we kept the rigorous comparison and dropped the part that couldn't be made
> rigorous under our constraints. Amplitude encoding without a matched classical
> baseline would have reproduced the exact unmatched-comparison weakness we criticise
> in the literature. Deferring it protects the integrity of the result.

**Q: "Could you not have run even a small version?"**
> A small *unmatched* run would not have answered Objective 3's comparison question —
> it would have produced two numbers that aren't comparable. The honest deliverable is
> a deep, fair characterisation of one encoding plus a principled scope for the other,
> rather than two weakly-controlled results.

---

## One-line summary

> Angle encoding is fully evaluated; amplitude encoding is deferred because, in an
> architecture that already reduces dimensionality classically, it adds no
> representational benefit at the bottleneck, breaks the parameter-matched comparison,
> and is not feasible to study fairly on a CPU simulator — so it belongs to a distinct
> future architecture with its own baseline.
