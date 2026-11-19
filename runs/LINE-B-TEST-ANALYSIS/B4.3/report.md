# Line B B4.3 — Formal Test Scaffold-Clustered Paired Bootstrap

> **Classification: exploratory / post-hoc.** The Test split was opened before B4. Confidence intervals describe scaffold-resampling uncertainty under the fixed three-seed models and fixed Test set; they are not training-randomness estimates and are not confirmatory significance tests.

- Replicates: **2000** (RNG seed `20261117`), scaffold clusters = Test scaffolds.
- One shared scaffold draw matrix for both tasks, all 18 models, all 3 seeds, all labels and all comparisons.
- AP/AUROC weighted by scaffold draw counts; undefined when a weighted class is empty (never imputed). F1 uses zero_division=0; both `fixed_0.5` and `validation_f1` policies are reported.

## 1. Scaffold clusters

| Task | Test rows | Scaffolds | Min size | Median | Max | Singletons | Resampled molecules (mean) |
|---|---|---|---|---|---|---|---|
| Shared14 | 12981 | 2853 | 1 | 1.0 | 1792 | 2517 | 12864.7 |
| UVSPEC15 | 12981 | 2853 | 1 | 1.0 | 1792 | 2517 | 12864.7 |

## 2. Full-Test point estimates (3-seed mean ± sample SD)

| Task | Comparison | Metric | Point (3-seed mean) | Seed SD (ddof=1) |
|---|---|---|---|---|
| Shared14 | IR-Raman | AP | +0.0143 | 0.0025 |
| Shared14 | IR-Raman | AUROC | +0.0044 | 0.0005 |
| Shared14 | IR-Raman | F1 (fixed_0.5) | +0.0308 | 0.0039 |
| Shared14 | IR-Raman | F1 (validation_f1) | +0.0205 | 0.0063 |
| Shared14 | UV-IR | AP | -0.4570 | 0.0055 |
| Shared14 | UV-IR | AUROC | -0.1431 | 0.0011 |
| Shared14 | UV-IR | F1 (fixed_0.5) | -0.5054 | 0.0240 |
| Shared14 | UV-IR | F1 (validation_f1) | -0.4184 | 0.0061 |
| Shared14 | UV-Raman | AP | -0.4427 | 0.0039 |
| Shared14 | UV-Raman | AUROC | -0.1387 | 0.0008 |
| Shared14 | UV-Raman | F1 (fixed_0.5) | -0.4746 | 0.0270 |
| Shared14 | UV-Raman | F1 (validation_f1) | -0.3978 | 0.0031 |
| UVSPEC15 | IR-Raman | AP | -0.0051 | 0.0089 |
| UVSPEC15 | IR-Raman | AUROC | -0.0005 | 0.0006 |
| UVSPEC15 | IR-Raman | F1 (fixed_0.5) | -0.0050 | 0.0118 |
| UVSPEC15 | IR-Raman | F1 (validation_f1) | -0.0093 | 0.0147 |
| UVSPEC15 | UV-IR | AP | -0.4329 | 0.0590 |
| UVSPEC15 | UV-IR | AUROC | -0.0725 | 0.0216 |
| UVSPEC15 | UV-IR | F1 (fixed_0.5) | -0.4859 | 0.0749 |
| UVSPEC15 | UV-IR | F1 (validation_f1) | -0.4127 | 0.0494 |
| UVSPEC15 | UV-Raman | AP | -0.4380 | 0.0661 |
| UVSPEC15 | UV-Raman | AUROC | -0.0729 | 0.0221 |
| UVSPEC15 | UV-Raman | F1 (fixed_0.5) | -0.4909 | 0.0834 |
| UVSPEC15 | UV-Raman | F1 (validation_f1) | -0.4220 | 0.0625 |

## 3. Three-seed bootstrap 95% percentile CI (macro)

| Task | Comparison | Metric | Point | CI low | CI high | Valid reps | Undefined | Labels defined |
|---|---|---|---|---|---|---|---|---|
| Shared14 | IR-Raman | AP | +0.0143 | -0.0192 | +0.0224 | 2000 | 0 | 14 |
| Shared14 | IR-Raman | AUROC | +0.0044 | +0.0030 | +0.0064 | 2000 | 0 | 14 |
| Shared14 | IR-Raman | F1 (fixed_0.5) | +0.0308 | +0.0216 | +0.0397 | 2000 | 0 | 14 |
| Shared14 | IR-Raman | F1 (validation_f1) | +0.0205 | -0.0005 | +0.0282 | 2000 | 0 | 14 |
| Shared14 | UV-IR | AP | -0.4570 | -0.4871 | -0.3857 | 2000 | 0 | 14 |
| Shared14 | UV-IR | AUROC | -0.1431 | -0.1630 | -0.1304 | 2000 | 0 | 14 |
| Shared14 | UV-IR | F1 (fixed_0.5) | -0.5054 | -0.5370 | -0.4221 | 2000 | 0 | 14 |
| Shared14 | UV-IR | F1 (validation_f1) | -0.4184 | -0.4478 | -0.3408 | 2000 | 0 | 14 |
| Shared14 | UV-Raman | AP | -0.4427 | -0.4739 | -0.3843 | 2000 | 0 | 14 |
| Shared14 | UV-Raman | AUROC | -0.1387 | -0.1575 | -0.1265 | 2000 | 0 | 14 |
| Shared14 | UV-Raman | F1 (fixed_0.5) | -0.4746 | -0.5064 | -0.3946 | 2000 | 0 | 14 |
| Shared14 | UV-Raman | F1 (validation_f1) | -0.3978 | -0.4290 | -0.3257 | 2000 | 0 | 14 |
| UVSPEC15 | IR-Raman | AP | -0.0051 | -0.0154 | +0.0013 | 2000 | 0 | 15 |
| UVSPEC15 | IR-Raman | AUROC | -0.0005 | -0.0012 | +0.0000 | 2000 | 0 | 15 |
| UVSPEC15 | IR-Raman | F1 (fixed_0.5) | -0.0050 | -0.0196 | +0.0037 | 2000 | 0 | 15 |
| UVSPEC15 | IR-Raman | F1 (validation_f1) | -0.0093 | -0.0233 | +0.0001 | 2000 | 0 | 15 |
| UVSPEC15 | UV-IR | AP | -0.4329 | -0.4910 | -0.3931 | 2000 | 0 | 15 |
| UVSPEC15 | UV-IR | AUROC | -0.0725 | -0.0978 | -0.0540 | 2000 | 0 | 15 |
| UVSPEC15 | UV-IR | F1 (fixed_0.5) | -0.4859 | -0.5224 | -0.4617 | 2000 | 0 | 15 |
| UVSPEC15 | UV-IR | F1 (validation_f1) | -0.4127 | -0.4584 | -0.3872 | 2000 | 0 | 15 |
| UVSPEC15 | UV-Raman | AP | -0.4380 | -0.5006 | -0.3973 | 2000 | 0 | 15 |
| UVSPEC15 | UV-Raman | AUROC | -0.0729 | -0.0987 | -0.0545 | 2000 | 0 | 15 |
| UVSPEC15 | UV-Raman | F1 (fixed_0.5) | -0.4909 | -0.5325 | -0.4672 | 2000 | 0 | 15 |
| UVSPEC15 | UV-Raman | F1 (validation_f1) | -0.4220 | -0.4696 | -0.3964 | 2000 | 0 | 15 |

## 4. Undefined handling summary

| Group | Undefined replicates |
|---|---|
| Shared14|AP|three_seed | 762 |
| Shared14|AUROC|three_seed | 762 |

Per-seed CIs, per-label CIs and all undefined counts are in `bootstrap_ci_per_seed.csv` / `bootstrap_ci_three_seed.csv`.

## 5. Interpretation boundaries

- Bootstrap CI covers scaffold-sampling uncertainty only, with the models and Test set held fixed.
- Seed SD (ddof=1, n=3) describes seed-to-seed variability and is **not** combined with the bootstrap CI.
- A CI including 0 is not evidence of directional stability; a CI excluding 0 is not confirmatory significance.
- Very-low-support labels are unstable under scaffold resampling and must not carry modality conclusions alone.
- No Test inference, threshold fitting, model selection or Bootstrap-based tuning was performed in B4.3.

## 6. Provenance

- `analysis_manifest.json` records all input/output SHA256; `verification.json` is the independent check.
