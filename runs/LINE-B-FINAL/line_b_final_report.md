# Line B — Final Report (B5)

> **Classification: exploratory / post-hoc.** Test was opened before B4; results are not a blind test and are not confirmatory. This report summarizes frozen B4 assets; no training, inference, threshold fitting or bootstrap was performed in B5.

## 1. Executive Summary

Line B evaluates whether IR, Raman and UV-Vis carry complementary structural evidence on QM9S. Across 18 frozen models (2 tasks × 3 modalities × 3 seeds; Test n=12,981), IR/Raman strongly outperform UV-Vis on Shared14; UV-Vis does **not** show the expected advantage on UVSPEC15. Error sets differ, but positive OR does not beat the best single modality, True Oracle gives a high diagnostic ceiling (not fusion performance), and UV conditional rescue is below its marginal rate. Structural subgroup gaps vary by task and subgroup. H1 is partially supported, H2 rejected, H3 partially supported, H4 descriptively supported.

**Three Phase-0 WARNs are fixed in this report:** (1) B4.3 CIs are paired-difference CIs, not absolute-performance CIs; (2) H3's `only_UV/only_IR/only_Raman × subgroup` enrichment is **not directly tested** and is not claimed; (3) AP/AUROC ≠ chemical-information content, QM9S ≠ experimental domain, and representative molecules are qualitative only.

## 2. Research Questions and Hypotheses

- H1: vibration modalities (IR/Raman) have a structural advantage on Shared14, especially on vibration-sensitive labels. **Verdict: PARTIAL SUPPORT.**
- H2: UV-Vis has an electronic-structure advantage on UVSPEC15. **Verdict: REJECTED** under the current task/metric system; this does not mean UV lacks electronic-structure information.
- H3: errors are complementary; positive OR may help but is only a fixed rule; true Oracle assesses complementarity; UV exclusive-correct samples should enrich in conjugated/aromatic subgroups. **Verdict: PARTIAL SUPPORT**; the subgroup enrichment clause is **not directly tested**.
- H4: modality gaps vary by structural subgroup; no universal winner. **Verdict: DESCRIPTIVELY SUPPORTED.**

## 3. Dataset and Experimental Design

- QM9S: 129,817 molecules with IR, Raman and UV-Vis computed spectra; Bemis–Murcko scaffold split 80/10/10; Test n=12,981; 2,853 Test scaffolds.
- Tasks: Shared14 (14 functional-group labels) and UVSPEC15 (15 UV-specific semantic labels); probabilities from frozen FP32 Test exports.
- Seeds: 42, 123, 2026; no Test-driven seed/model/label/subgroup selection.

## 4. Model Architecture and Training Configuration

- Frozen single-modality 1D-CNN encoders (`training/models/spectrum_cnn1d.py`), embedding dimension 256; input length 3,501 (IR/Raman) or 701 (UV-Vis); sigmoid multi-label heads (14/15).
- Training (frozen v1 manifests): AdamW, batch 512, lr 1e-3, weight decay 1e-4, up to 50 epochs with early stopping (patience 8), BCE/WBCE as recorded per frozen run; max normalization.
- B5 loaded only the frozen Test probability arrays; no checkpoint was loaded and no inference was run.

## 5. Single-Modality Performance

Macro AP/AUROC are threshold-free; micro AP/AUROC are flattened sample-label summaries computed in B5 from the frozen B4.1 probabilities (not a replacement for macro metrics); F1 is shown for both policies. All values are seed means; full per-seed and per-label tables are in `tables/`.

| Task | Modality | micro AP | micro AUROC | macro AP | macro AUROC | macro F1 (0.5) | macro F1 (Val-F1) | micro F1 (0.5) |
|---|---|---|---|---|---|---|---|---|
| Shared14 | IR | 0.9873 | 0.9974 | 0.9695 | 0.9955 | 0.9192 | 0.9203 | 0.9405 |
| Shared14 | RAMAN | 0.9767 | 0.9950 | 0.9551 | 0.9911 | 0.8884 | 0.8997 | 0.9151 |
| Shared14 | UVVIS | 0.6295 | 0.8940 | 0.5125 | 0.8524 | 0.4138 | 0.5019 | 0.5159 |
| UVSPEC15 | IR | 0.9896 | 0.9985 | 0.9709 | 0.9977 | 0.9245 | 0.9212 | 0.9512 |
| UVSPEC15 | RAMAN | 0.9914 | 0.9989 | 0.9760 | 0.9982 | 0.9295 | 0.9305 | 0.9538 |
| UVSPEC15 | UVVIS | 0.6845 | 0.9379 | 0.5380 | 0.9252 | 0.4386 | 0.5085 | 0.6108 |

- Shared14 best macro AP: **IR**; UVSPEC15 best macro AP: **RAMAN**.
- Full 36-row micro/macro table: `tables/micro_macro_summary.csv`; 36-row overall table: `tables/overall_metrics.csv`; 522-row per-label table: `tables/per_label_metrics.csv`.

## 6. Seed Stability

Seed Mean±SD is in `tables/seed_mean_sd.csv` (96 rows: task × policy × modality × metric). Per-seed values are in `tables/overall_metrics.csv`; the fixed-0.5 macro-F1 seed trajectories are in `figures/fig04_seed_stability_*.png`.

| Task | Modality | macro AP | macro AUROC | macro F1 (0.5) | macro F1 (Val-F1) |
|---|---|---|---|---|---|
| Shared14 | IR | 0.9695 ± 0.0056 | 0.9955 ± 0.0008 | 0.9192 ± 0.0049 | 0.9203 ± 0.0070 |
| Shared14 | RAMAN | 0.9551 ± 0.0043 | 0.9911 ± 0.0004 | 0.8884 ± 0.0080 | 0.8997 ± 0.0053 |
| Shared14 | UVVIS | 0.5125 ± 0.0007 | 0.8524 ± 0.0004 | 0.4138 ± 0.0192 | 0.5019 ± 0.0021 |
| UVSPEC15 | IR | 0.9709 ± 0.0063 | 0.9977 ± 0.0004 | 0.9245 ± 0.0083 | 0.9212 ± 0.0099 |
| UVSPEC15 | RAMAN | 0.9760 ± 0.0029 | 0.9982 ± 0.0002 | 0.9295 ± 0.0038 | 0.9305 ± 0.0048 |
| UVSPEC15 | UVVIS | 0.5380 ± 0.0645 | 0.9252 ± 0.0220 | 0.4386 ± 0.0814 | 0.5085 ± 0.0581 |

SD is the sample SD across the three seeds (n=3, descriptive only).

## 7. Paired Scaffold Bootstrap

**WARN-1 fixed:** B4.3 reports CIs for paired differences `Δ = Metric(base) − Metric(other)`, not for absolute IR/Raman/UV-Vis performance. Absolute single-modality AP/AUROC/F1 are B4.2 point estimates. The B4.3 CI therefore answers “is the modality difference stable under scaffold resampling?”, not “what is the absolute performance CI?”.

- 2,000 scaffold replicates; one shared draw matrix; 2,853 Test scaffolds; 1,488 point rows; 1,116 per-seed CI rows; 372 three-seed CI rows.
- Three-seed paired differences and 95% CIs: `tables/paired_differences_three_seed.csv`; per-seed: `tables/paired_differences_per_seed.csv`; forest plot: `figures/fig05_paired_bootstrap_forest.png`.
- Macro differences whose 95% CI excludes 0: **18 / 24**.

## 8. Error Complementarity

Macro error Jaccard, FN rescue and FP correction (fixed 0.5, three-seed mean; full 558-row table in `tables/pairwise_error_complementarity.csv`):

| Task | Comparison | F1 base/other | error Jaccard | FN rescue | FP correction | both wrong |
|---|---|---|---|---|---|---|
| Shared14 | IR-Raman | 0.904/0.894 | 0.1005 | 45476 | 57184 | 9996 |
| Shared14 | UV-IR | 0.475/0.920 | 0.0739 | 60231 | 62888 | 11346 |
| Shared14 | UV-Raman | 0.445/0.896 | 0.0756 | 74433 | 68501 | 13341 |
| UVSPEC15 | IR-Raman | 0.910/0.936 | 0.0823 | 24293 | 33675 | 3473 |
| UVSPEC15 | UV-IR | 0.484/0.930 | 0.0530 | 30025 | 41627 | 4096 |
| UVSPEC15 | UV-Raman | 0.455/0.936 | 0.0488 | 35875 | 49527 | 4653 |

Error sets are not identical, but this alone does **not** establish conditional complementarity (Section 10). McNemar/Holm results: `tables/mcnemar_holm.csv`.

## 9. Positive OR / AND and True Oracle

Positive OR/AND are fixed rules, **not** True Oracle; True Oracle is a diagnostic ceiling, **not** a deployable fusion model.

| Task | Pair | OR F1 | AND F1 | OR gain vs max | +gain / −gain labels |
|---|---|---|---|---|---|
| Shared14 | IR-Raman | 0.918 | 0.886 | -0.0052 | 3.7/10.3 |
| Shared14 | UV-IR | 0.859 | 0.450 | -0.0601 | 0.3/12.7 |
| Shared14 | UV-Raman | 0.839 | 0.435 | -0.0496 | 0.7/12.3 |
| UVSPEC15 | IR-Raman | 0.924 | 0.931 | -0.0117 | 2.0/13.0 |
| UVSPEC15 | UV-IR | 0.801 | 0.536 | -0.1239 | 0.0/14.3 |
| UVSPEC15 | UV-Raman | 0.803 | 0.541 | -0.1265 | 0.0/14.3 |

| Task | best_single | F1 best_single | F1 three-OR | three-OR gain |
|---|---|---|---|---|
| Shared14 | ir | 0.9192 | 0.8569 | -0.0623 |
| UVSPEC15 | raman | 0.9295 | 0.7982 | -0.1314 |

| Task | Oracle any | exactly one | all wrong | all correct | only IR | only Raman | only UV |
|---|---|---|---|---|---|---|---|
| Shared14 | 0.9968 | 0.0149 | 0.0032 | 0.8578 | 1423 | 777 | 509 |
| UVSPEC15 | 0.9989 | 0.0055 | 0.0011 | 0.9150 | 393 | 478 | 205 |

Full tables: `tables/pairwise_or_and.csv`, `tables/three_modality_or.csv`, `tables/true_oracle_summary.csv`, `tables/true_oracle_per_label.csv`.

## 10. Conditional Complementarity

Conditional rates are unweighted full-Test descriptive rates; 95% CIs use 2,000 B4.3 scaffold draws. UV rescue is far below UV marginal, so the data do not support a conditional-complementarity claim.

| Task | Direction | P(other correct | base wrong) | P(other) marginal | delta |
|---|---|---|---|---|
| Shared14 | IR→RAMAN | 0.6670 | 0.9747 | -0.3077 |
| Shared14 | RAMAN→IR | 0.7644 | 0.9820 | -0.2176 |
| Shared14 | UVVIS→IR | 0.9380 | 0.9820 | -0.0440 |
| Shared14 | IR→UVVIS | 0.5854 | 0.8799 | -0.2945 |
| Shared14 | UVVIS→RAMAN | 0.9084 | 0.9747 | -0.0663 |
| Shared14 | RAMAN→UVVIS | 0.5657 | 0.8799 | -0.3142 |
| UVSPEC15 | IR→RAMAN | 0.7578 | 0.9916 | -0.2338 |
| UVSPEC15 | RAMAN→IR | 0.7501 | 0.9913 | -0.2412 |
| UVSPEC15 | UVVIS→IR | 0.9500 | 0.9913 | -0.0412 |
| UVSPEC15 | IR→UVVIS | 0.6003 | 0.9246 | -0.3243 |
| UVSPEC15 | UVVIS→RAMAN | 0.9550 | 0.9916 | -0.0366 |
| UVSPEC15 | RAMAN→UVVIS | 0.6343 | 0.9246 | -0.2903 |

UV-specific: 509/181,734 pairs on Shared14 and 205/194,715 on UVSPEC15 where IR and Raman are both wrong but UV is correct; UV rescue rates 0.469/0.498 vs marginal 0.880/0.925, delta −0.411/−0.426. 95% CI for rescue: Shared14 [0.4204, 0.5128], UVSPEC15 [0.4282, 0.5550]; delta CI: Shared14 [-0.4572, -0.3677], UVSPEC15 [-0.5001, -0.3639] (entirely below zero). Full table and CIs: `tables/conditional_complementarity.csv`, `tables/conditional_bootstrap_ci.csv`; figure: `figures/fig08_conditional_complementarity.png`.

## 11. Structural Subgroup Analysis

Predefined D1/B0 §7.5 subgroups; support requires n≥100 and positive≥30. Heavy-atom quartiles are degenerate (Q1=Q2=Q3=9.0); `functional_group=aromatic` is empty under the frozen token definition. Subgroup CIs use 500 within-subgroup paired scaffold draws and are exploratory.

- Supported subgroups: Shared14 **21**, UVSPEC15 **21**; each task has 22 non-empty subgroups, but `aromatic_ring_count=≥2` has no label meeting the positive≥30 support rule.
- Best-AP modality counts: Shared14 **{'ir': 19, 'raman': 2}**, UVSPEC15 **{'raman': 16, 'ir': 5}**; UV-Vis is never the best modality in any supported subgroup.
- At fixed 0.5, all 84 UV−IR / UV−Raman subgroup comparisons have CI upper < 0; no supported subgroup shows a positive UV advantage with CI lower > 0.
- Full tables: `tables/subgroup_metrics.csv`, `tables/subgroup_bootstrap_ci.csv`; heatmap: `figures/fig09_subgroup_performance_heatmap.png`.

## 12. Representative Molecule Analysis

The B4.4 supplement is archived byte-identically under `representative_molecules/` (88 cases: 44 per task; 22 correct + 22 error each). Each case keeps RDKit PNG/SVG, spectra PNG and metadata with labels, predictions, probabilities and errors. This is a qualitative case display, **not XAI** and **not statistical evidence**; no case is used to estimate performance.

| Task | Cases | Correct | Error | Non-empty subgroups | Empty subgroups |
|---|---|---|---|---|---|
| Shared14 | 44 | 22 | 22 | 22 | 4 |
| UVSPEC15 | 44 | 22 | 22 | 22 | 4 |

Index: `tables/representative_molecules.csv`; contact sheet: `figures/fig10_representative_molecules.png`.

## 13. H1–H4 Hypothesis Evaluation

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1 | **PARTIAL SUPPORT** | Shared14 macro AP IR/Raman/UV=0.9695/0.9551/0.5125; vibration gap=0.4451 vs other=0.4653 |
| H2 | **REJECTED** | UVSPEC15 macro AP UV/IR/Raman=0.5380/0.9709/0.9760; AUROC=0.9252/0.9977/0.9982 |
| H3 | **PARTIAL SUPPORT** | Oracle any=0.9968/0.9989; three-OR gain range=[-0.2321,-0.0598]; UV rescue=0.4692/0.4984 vs marginal=0.8799/0.9246 |
| H4 | **DESCRIPTIVELY SUPPORTED** | best-AP modality counts={'Shared14': {'ir': 19, 'raman': 2}, 'UVSPEC15': {'raman': 16, 'ir': 5}} |

- **H1 PARTIAL SUPPORT:** the main IR/Raman > UV margin holds on Shared14, but the vibration-sensitive-label gap (0.4451) is not larger than the other-label gap (0.4653).
- **H2 REJECTED:** on UVSPEC15, UV macro AP 0.5380 vs IR 0.9709 / Raman 0.9760; AUROC 0.9252 vs 0.9977 / 0.9982. This does not mean UV has no electronic-structure information or no potential complementary value.
- **H3 PARTIAL SUPPORT:** errors are complementary and Oracle any-correct is high, but positive OR does not beat best_single; UV conditional rescue 0.469/0.498 is below marginal 0.880/0.925. **The `only_UV/only_IR/only_Raman × structural subgroup` enrichment is not directly tested**; no UV-conjugation enrichment claim is made.
- **H4 DESCRIPTIVELY SUPPORTED:** best-AP modality differs by task and subgroup; no universal modality ranking.

All hypothesis statements are exploratory/post-hoc; no confirmatory significance is claimed.

## 14. Implications for Line C

1. Do not build a fusion model on the assumption that UV adds conditional value on UVSPEC15; the current evidence shows UV below IR/Raman and conditional rescue below marginal.
2. Positive OR is not a sufficient fusion target; it can trade precision for recall and is not Oracle. Line C should optimize a learned/gated policy and compare against best_single, not only OR.
3. Because H3's subgroup enrichment is untested, Line C should not route UV specifically to conjugated/aromatic subgroups based on this evidence; that routing hypothesis needs a direct test.
4. IR/Raman are the strong baselines on both tasks; any fusion gain must be demonstrated against the best single modality with scaffold-aware uncertainty.

## 15. Limitations

- **WARN-3 boundaries:** AP/AUROC measure ranking/separation of labels, **not** the total chemical information content of a modality.
- QM9S is a computational spectra dataset; it does not directly represent experimental measurement conditions.
- Results must **not** be extrapolated to NIST, SDBS or API-Raman experimental domains.
- Representative molecules are qualitative illustrations, **not XAI** and **not independent statistical evidence**.
- Heavy-atom quartiles are degenerate; `functional_group=aromatic` is empty under the frozen token definition; subgroup CIs use 500 replicates (exploratory).
- B4.3 CIs are paired-difference CIs under fixed models/Test set; they are not training-randomness or absolute-performance CIs.
- Test was opened before B4; all results are exploratory/post-hoc, not a blind test.

## 16. Reproducibility and Data Provenance

- B0 v0.2 SHA256 `713fa92f5fd1f2134c1512bd2e49f689433a2158d120e41c9460ff5c287dd46a`; FREEZE `65a84959356a9a1f7a6e1d6434449f607c75a4c224a17450f3345ba25bb92a0c`.
- B4.4 tree SHA256 `d3e69052171ae753849a47139f21a03460ba7f772bd8d95e5143f749b9007a69`; B4.2/B4.3/B4.4/Supplement verification all PASS.
- All 18 models, 3 seeds and 14/15 labels are represented in `tables/overall_metrics.csv`, `tables/per_label_metrics.csv` and `tables/micro_macro_summary.csv`.
- Output SHA256 values and input hashes are in `asset_manifest.json`; independent verification is in `verification.json`.
- Figures are generated from the CSV/JSON assets with Pillow (no matplotlib); no schematic data are used.
