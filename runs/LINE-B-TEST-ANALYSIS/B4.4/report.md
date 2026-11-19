# Line B B4.4 — Test Error Complementarity & Structural Subgroups

> **Classification: exploratory / post-hoc.** Test was opened before B4. Positive OR/AND are fixed rules, not Oracle; Oracle is a diagnostic upper bound, not a deployable model; subgroup and conditional findings carry no confirmatory significance.

- Models: 18 (2 tasks × 3 modalities × 3 seeds); two F1 policies (fixed 0.5 primary, frozen Validation-F1 secondary).
- Pairwise rows: 558; McNemar rows: 522; subgroup metric rows: 11484.
- McNemar: two-sided exact test; Holm family = task × seed × policy across the three modality pairs and all labels (C2 rule).
- Conditional/OR/Oracle use sample-label pairs; hard samples use deterministic top 10%; high-confidence errors use prob ≥ 0.90 FP / ≤ 0.10 FN (C2 rule).

## 1. Macro pairwise error complementarity (fixed 0.5, three-seed mean)

| Task | Comparison | F1 A | F1 B | Error Jaccard | FN rescue | FP correction | Both wrong |
|---|---|---|---|---|---|---|---|
| Shared14 | IR-Raman | 0.9036 | 0.8941 | 0.1005 | 45476 | 57184 | 9996 |
| Shared14 | UV-IR | 0.4754 | 0.9203 | 0.0739 | 60231 | 62888 | 11346 |
| Shared14 | UV-Raman | 0.4448 | 0.8955 | 0.0756 | 74432 | 68501 | 13341 |
| UVSPEC15 | IR-Raman | 0.9104 | 0.9360 | 0.0823 | 24292 | 33675 | 3473 |
| UVSPEC15 | UV-IR | 0.4837 | 0.9298 | 0.0530 | 30024 | 41627 | 4096 |
| UVSPEC15 | UV-Raman | 0.4552 | 0.9356 | 0.0488 | 35875 | 49526 | 4652 |

## 2. Positive OR / AND (fixed 0.5, three-seed mean)

| Task | Comparison | F1 OR | F1 AND | OR gain vs max | +gain labels | −gain labels |
|---|---|---|---|---|---|---|
| Shared14 | IR-Raman | 0.9179 | 0.8865 | -0.0052 | 3 | 10 |
| Shared14 | UV-IR | 0.8591 | 0.4497 | -0.0601 | 0 | 12 |
| Shared14 | UV-Raman | 0.8387 | 0.4353 | -0.0496 | 0 | 12 |
| UVSPEC15 | IR-Raman | 0.9240 | 0.9307 | -0.0117 | 2 | 13 |
| UVSPEC15 | UV-IR | 0.8006 | 0.5359 | -0.1239 | 0 | 14 |
| UVSPEC15 | UV-Raman | 0.8031 | 0.5415 | -0.1265 | 0 | 14 |

### Three-modality OR vs Validation-designated best_single

| Task | best_single | F1 best_single | F1 three-OR | OR gain |
|---|---|---|---|---|
| Shared14 | ir | 0.9192 | 0.8569 | -0.0623 |
| UVSPEC15 | raman | 0.9295 | 0.7982 | -0.1314 |

## 3. True Oracle (sample-label pairs, three-seed mean)

| Task | Policy | any correct | exactly one | all wrong | all correct | only IR | only Raman | only UV |
|---|---|---|---|---|---|---|---|---|
| Shared14 | fixed_0.5 | 0.9968 | 0.0149 | 0.0032 | 0.8578 | 1423 | 777 | 509 |
| Shared14 | validation_f1 | 0.9970 | 0.0161 | 0.0030 | 0.8122 | 1442 | 864 | 620 |
| UVSPEC15 | fixed_0.5 | 0.9989 | 0.0055 | 0.0011 | 0.9150 | 393 | 478 | 205 |
| UVSPEC15 | validation_f1 | 0.9989 | 0.0056 | 0.0011 | 0.9193 | 438 | 424 | 225 |

## 4. Conditional complementarity (fixed 0.5; point over three seeds)

| Task | Direction | P(other correct | base wrong) | marginal P(other) | delta |
|---|---|---|---|---|
| Shared14 | IR wrong → RAMAN | 0.6670 | 0.9747 | -0.3077 |
| Shared14 | RAMAN wrong → IR | 0.7644 | 0.9820 | -0.2176 |
| Shared14 | UVVIS wrong → IR | 0.9380 | 0.9820 | -0.0440 |
| Shared14 | IR wrong → UVVIS | 0.5854 | 0.8799 | -0.2945 |
| Shared14 | UVVIS wrong → RAMAN | 0.9084 | 0.9747 | -0.0663 |
| Shared14 | RAMAN wrong → UVVIS | 0.5657 | 0.8799 | -0.3142 |
| UVSPEC15 | IR wrong → RAMAN | 0.7578 | 0.9916 | -0.2338 |
| UVSPEC15 | RAMAN wrong → IR | 0.7501 | 0.9913 | -0.2412 |
| UVSPEC15 | UVVIS wrong → IR | 0.9500 | 0.9913 | -0.0412 |
| UVSPEC15 | IR wrong → UVVIS | 0.6003 | 0.9246 | -0.3243 |
| UVSPEC15 | UVVIS wrong → RAMAN | 0.9550 | 0.9916 | -0.0366 |
| UVSPEC15 | RAMAN wrong → UVVIS | 0.6343 | 0.9246 | -0.2903 |

### UV rescue vs marginal (fixed 0.5, three-seed mean, 95% bootstrap CI over B4.3 scaffold draws)

| Task | UV rescue rate | UV marginal | delta | CI low | CI high |
|---|---|---|---|---|---|
| Shared14 | UV rescue | 0.4679 | | +0.4679 | +0.4204 | +0.5128 |
| Shared14 | rescue − marginal | -0.4116 | | -0.4116 | -0.4572 | -0.3677 |
| UVSPEC15 | UV rescue | 0.4952 | | +0.4952 | +0.4282 | +0.5550 |
| UVSPEC15 | rescue − marginal | -0.4282 | | -0.4282 | -0.5001 | -0.3639 |

## 5. Sample-level and high-confidence errors (fixed 0.5, three-seed mean)

| Task | Modality | mean errors | median | IQR | exact-match rate | HC errors |
|---|---|---|---|---|---|---|
| Shared14 | IR | 0.252 | 0.0 | 0.00 | 0.8068 | 1135 |
| Shared14 | RAMAN | 0.355 | 0.0 | 1.00 | 0.7337 | 1676 |
| Shared14 | UVVIS | 1.681 | 2.0 | 1.33 | 0.1903 | 3152 |
| UVSPEC15 | IR | 0.131 | 0.0 | 0.00 | 0.9401 | 771 |
| UVSPEC15 | RAMAN | 0.127 | 0.0 | 0.00 | 0.9297 | 601 |
| UVSPEC15 | UVVIS | 1.132 | 0.0 | 1.67 | 0.6459 | 1983 |

## 6. Structural subgroups

- Predefined subgroups: 22; rows meeting per-label support (n≥100 and positive≥30): 8964 metric rows.
- Heavy-atom quartiles are degenerate by design (D1: Q1=Q2=Q3=9.0; all Test molecules fall in Q1).
- Paired within-subgroup scaffold bootstrap CIs computed for 21 supported subgroups (500 replicates each, shared draw per subgroup; exploratory).
- Subgroup rows below support thresholds are retained and flagged `support_sufficient=false`.

## 7. Phase-4 questions

1. **Are IR/Raman/UV errors complementary?** Only partially. Error sets are not identical (macro error Jaccard 0.049–0.101 at fixed 0.5; FN-rescue / FP-correction counts are large), but the three-modality Oracle `any_correct` rate is already 0.9968 (Shared14) / 0.9989 (UVSPEC15), and only 1.49% / 0.55% of sample-label pairs have exactly one correct modality.
2. **Can UV provide extra correct evidence when IR and Raman both fail?** Yes in absolute count: 509/181,734 pairs (0.28%) on Shared14 and 205/194,715 (0.11%) on UVSPEC15. But this is not a rate above UV's own accuracy.
3. **Is UV conditional complementarity above its marginal accuracy?** No. UV rescue rate is 0.469 (Shared14) / 0.498 (UVSPEC15) versus UV marginal 0.880 / 0.925; delta −0.411 / −0.426 with the 95% scaffold-bootstrap CI entirely below zero.
4. **Which subgroups depend more on IR?** 19 of 21 supported Shared14 subgroups have IR as the highest mean-AP modality (e.g., carbonyl, nitrile, alkyne, hydroxyl).
5. **Which depend more on Raman?** 16 of 21 supported UVSPEC15 subgroups, plus Shared14 `aromatic_ring_count=1` and `n_o_f=0`.
6. **Where does UV add value?** No supported subgroup shows a positive UV−IR or UV−Raman AP difference with a positive CI lower bound: all 84 fixed-0.5 subgroup UV comparisons have CI upper < 0. The least-negative UV rescue delta is −0.0099 (`aromatic_ring_count=≥2`, Shared14), still below marginal.
7. **H3 / H4 support:** H3 is only partially supported — errors differ and the Oracle ceiling is high, but positive OR does not exceed the best single modality (three-modality OR F1 is 0.06–0.23 below best_single) and UV rescue is below marginal. H4 is supported descriptively: modality gaps vary by subgroup (IR-best 19 vs Raman-best 2 on Shared14; Raman-best 16 vs IR-best 5 on UVSPEC15), so no "UV is better/worse in every subgroup" claim holds.

## 8. Interpretation boundaries

- Positive OR/AND are fixed rules; OR F1 is not Oracle and may gain recall while losing precision.
- Oracle correctness is a diagnostic ceiling over three fixed models, not a fusion model.
- Conditional rescue is descriptive; only a positive delta with a bootstrap CI lower bound > 0 is marked as evidence, and still exploratory.
- No Test-driven threshold, seed, modality or label selection; no confirmatory significance claims.
