# Quantum Advantage Pivot Starter

Last updated: 2026-06-30

## Purpose

This note is a starting point for a possible thesis pivot. The goal is not to force the current crop-disease hybrid E3 model to win. The goal is to identify any defensible setting where a quantum method has a real advantage or useful bounded benefit over strong classical baselines.

The current crop-disease setup is still useful as one comparison point, but it should not be treated as the only battlefield. Full QML on raw images or high-dimensional fused embeddings is not feasible in this project. The better question is:

```text
Where does a quantum model have a plausible structural reason to help?
```

## Working Opinion

A broad claim of quantum advantage on normal high-dimensional image classification is unlikely. Classical CNNs, MLPs, SVMs, and boosted trees are too strong, and encoding high-dimensional classical data into a small circuit usually destroys information.

The most promising route is to search in domains where the data itself has structure that a quantum kernel, quantum circuit, or quantum reservoir might model naturally:

- scientific/physical measurements
- molecular or materials data
- spectroscopy or diffraction patterns
- low-data active learning
- sensor/time-series dynamics
- synthetic quantum-advantage benchmarks used as sanity checks

The thesis can pivot if one of these directions produces a real empirical win under fair comparison.

## What Counts As A Useful Advantage

Do not limit the search to test accuracy only. A useful quantum advantage could be any of the following, as long as it is measured against strong matched classical baselines:

- higher Accuracy or Macro F1
- better sample efficiency at small training sizes
- better performance with fewer trainable parameters
- better kernel-target alignment
- better robustness under drift/corruption
- better uncertainty/calibration under distribution shift
- lower active-learning label cost
- better performance on a known hard synthetic quantum-kernel task

For every possible win, define the classical comparator first. A weak classical baseline does not prove anything.

## Priority 1: Quantum Kernel SVM On Structured Scientific Data

This is the strongest near-term direction.

Why it is promising:

- Quantum kernels have clearer theory than generic variational QNNs.
- Kernel methods work on small and medium datasets, which fits simulator limits.
- Scientific data may contain structured similarity patterns that are not well captured by simple RBF or polynomial kernels.

Models to compare:

- Classical SVM with RBF kernel
- Classical SVM with polynomial kernel
- Random forest or XGBoost if the data is tabular
- Small MLP
- Quantum kernel SVM with one or more feature maps

Metrics:

- Accuracy or Macro F1
- low-data curves at n = 50, 100, 200, 500, 1000 where possible
- kernel-target alignment
- runtime and simulation cost

Datasets to investigate:

| Dataset/source | Type | Why it fits | Start priority |
| --- | --- | --- | --- |
| Materials Project | materials properties/structures | Physical domain; can build small classification/regression tasks | High |
| NOMAD | materials archive | Broad materials/spectroscopy source | Medium |
| XRD or synthetic diffraction datasets | spectra/patterns | Peak-structured data, good for kernels | High |
| OpenML/PMLB small scientific/tabular datasets | tabular | Fast screening across many datasets | Medium |

Links:

- Materials Project: https://materialsproject.org/
- NOMAD: https://nomad-lab.eu/
- PMLB: https://github.com/EpistasisLab/pmlb
- OpenML: https://www.openml.org/

## Priority 2: Molecular / Quantum Chemistry Datasets

This is also a strong pivot direction because the target domain is naturally quantum/chemical.

Why it is promising:

- The data-generating process is closer to quantum physics than crop images.
- Molecular property prediction is a recognized ML benchmark area.
- Small molecular datasets can be screened quickly.

Models to compare:

- Morgan fingerprint + SVM/RBF
- Morgan fingerprint + random forest/XGBoost
- MLP on descriptors
- Quantum kernel on compact molecular descriptors
- Optional: quantum kernel on PCA-reduced fingerprints, but document the compression carefully

Candidate datasets:

| Dataset | Task type | Why it fits | Start priority |
| --- | --- | --- | --- |
| ESOL | regression | Small solubility dataset; quick screening | High |
| FreeSolv | regression | Small hydration free energy dataset | High |
| Lipophilicity | regression | Molecular property benchmark | Medium |
| BACE | classification | Binary molecular classification | High |
| BBBP | classification | Binary molecular classification | Medium |
| QM9 | regression | Quantum-chemistry properties; larger than ESOL/FreeSolv | Medium |
| QH9 | Hamiltonian/quantum chemistry benchmark | More directly quantum, but heavier | Medium/Low initially |

Links:

- MoleculeNet datasets: https://moleculenet.org/datasets-1
- QM9 information through MoleculeNet: https://moleculenet.org/datasets-1
- QH9 paper: https://arxiv.org/abs/2306.09549

## Priority 3: Sensor Drift And Time-Series / Reservoir Direction

This is useful if we want to explore quantum reservoir computing or quantum kernels for dynamic systems.

Why it is promising:

- Quantum reservoir computing is more naturally tied to temporal/dynamical behavior than static image classification.
- Sensor drift is close to the agricultural-sensor theme of the current thesis.
- Drift gives a meaningful robustness/generalization target.

Candidate datasets:

| Dataset/source | Type | Why it fits | Start priority |
| --- | --- | --- | --- |
| UCI Gas Sensor Array Drift | sensor drift classification | 13,910 measurements, 16 sensors, 6 gases, temporal batches | High |
| UCI HAR | human activity time series | Common benchmark, easy to process | Medium |
| PhysioNet datasets | biomedical time series | Rich source, but dataset choice needs care | Medium |
| Current crop IoT files | agricultural sensor time series | Directly connected to current thesis | Medium |

Links:

- UCI Gas Sensor Array Drift Dataset: https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset
- PhysioNet databases: https://physionet.org/about/database/
- UCI Machine Learning Repository: https://archive.ics.uci.edu/

## Priority 4: Synthetic Quantum-Advantage Benchmarks

This is necessary as a pipeline sanity check.

Why it matters:

If our implementation cannot reproduce a known or expected quantum-kernel win on a synthetic benchmark, then a failure on real data is hard to interpret. Synthetic tests tell us whether the code path is capable of showing a quantum benefit when one should exist.

Use synthetic datasets to answer:

- Does the quantum kernel implementation work?
- Can it beat RBF/poly kernels on a quantum-structured label function?
- How sensitive is the result to sample size, noise, and shot count?

This should not be the final thesis by itself unless the supervisor accepts a theory/benchmark-focused pivot, but it is a useful first gate.

## Priority 5: Current Crop Dataset As A Comparison Track

The crop dataset should remain in the study, but not as the only target.

Useful crop-side experiments:

- Quantum kernel on reduced E2 fused embeddings
- Quantum kernel on IoT-only summary features
- Quantum reservoir on raw IoT time-series summaries
- Low-data comparison against classical baselines
- Drift/corruption uncertainty comparison

Important limitation:

If we use E0/E1/E2 embeddings, this is still a hybrid/classical-feature pipeline. It is not full QML. That is acceptable if documented clearly.

## Recommended First Screening Study

Run a small screening study before committing to a full pivot.

Datasets:

1. UCI Gas Sensor Array Drift
2. ESOL or FreeSolv from MoleculeNet
3. One Materials Project or XRD/spectroscopy-derived task
4. Current crop IoT or reduced crop-fusion features
5. One synthetic quantum-kernel benchmark

Baselines:

1. SVM RBF
2. SVM polynomial
3. logistic regression or ridge regression where appropriate
4. random forest/XGBoost for tabular data
5. small MLP
6. quantum kernel SVM

Training sizes:

```text
n = 50, 100, 200, 500, 1000
```

Use fewer points for expensive quantum simulation if needed, but keep the same sample subset for every model in each run.

Core outputs:

```text
results.csv
per_dataset_summary.md
kernel_alignment.csv
runtime_summary.csv
plots/low_data_curves.png
plots/quantum_vs_best_classical.png
```

Decision rule:

- If quantum wins on at least one real dataset with a strong comparator, expand that track.
- If quantum only wins on synthetic data, keep it as a benchmark/theory section and continue searching real datasets.
- If quantum loses everywhere, the project can still report a rigorous negative result, but that is not the goal of this pivot search.

## Fairness Rules

Use these rules before making any advantage statement:

1. Same train/validation/test split for all models.
2. Same low-data sample IDs for all models.
3. Tune classical baselines seriously.
4. Report best classical baseline, not only average classical baseline.
5. Track runtime and simulator cost.
6. Report confidence intervals or repeated seeds where feasible.
7. Do not call PCA-reduced or embedding-based pipelines full QML.
8. Do not claim quantum advantage from one lucky seed.

## Immediate Next Steps

1. Pick three starter datasets: UCI Gas Sensor Drift, ESOL/FreeSolv, and one materials/spectroscopy source.
2. Build a small dataset-loader folder outside the current E0/E1/E2 pipeline.
3. Implement a shared benchmark harness for classical SVM/MLP and quantum kernel SVM.
4. Start with CPU/simulator-friendly sample sizes.
5. Write every result into a single comparison table.
6. Only then decide whether to pivot the thesis claim, dataset, or field.

## Recommended Thesis-Framing If This Works

Possible revised direction:

```text
Empirical search for quantum machine-learning advantage regimes across structured scientific datasets, with controlled comparisons against strong classical baselines.
```

Possible thesis question:

```text
Under what data regimes do quantum kernel or quantum reservoir methods provide measurable benefit over matched classical baselines?
```

Possible contribution:

```text
A controlled screening framework identifying where near-term QML is and is not useful across low-data, structured scientific learning tasks.
```

## Do Not Do First

Avoid starting with these:

- full QML image classification
- quantum CNNs on normal image datasets
- large high-dimensional embeddings compressed directly to 4 or 8 qubits without a serious comparator
- weak classical baselines
- one-off accuracy comparisons without repeated seeds
- claims based only on toy data

## Bottom Line

The best path is not to force quantum advantage into the current crop-disease E3 result. The best path is to search across domains where quantum structure is plausible: materials/spectroscopy, molecular chemistry, sensor drift/time-series, and synthetic sanity checks. If a real advantage appears, pivot the thesis around that evidence.
