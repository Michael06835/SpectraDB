# Fig3 — Per-label Test AP heatmap

## Provenance

- Source files downloaded to `~/Downloads/Fig3_PerLabel_TestAP_Heatmap.{pdf,png,svg,tex}` on 2026-11-20.
- Archived here byte-identically; the Downloads originals were retained.
- Data source: `runs/LINE-B-TEST-ANALYSIS/B4.2/per_label_metrics.csv` (Test, frozen FP32 exports).
- Metric: per-label **average precision (AP)** at the fixed 0.5 decision policy (AP is threshold-free); values are the **3-seed mean ± sample SD** (seeds 42/123/2026).

## Panels

- **(A) Functional group features** — Shared14 labels: carbonyl, hydroxyl, nitrile, alkene.
- **(B) Electronic structure features** — UVSPEC15 labels: `has_conjugation`, `has_aromatic_system`, `heteroaromatic`, `donor_acceptor_same_conjugated_system`.

The two panels come from different tasks; this must be stated in the caption. The panel-A and panel-B rows are not paired observations.

## Verification

Spot checks against B4.2 (mean ± SD):

| Label | IR | Raman | UV-Vis |
|---|---|---|---|
| Carbonyl | 0.998 ± 0.0003 | 0.998 ± 0.0001 | 0.898 ± 0.0015 |
| Hydroxyl | 1.000 ± 0.0000 | 1.000 ± 0.0000 | 0.514 ± 0.0057 |
| Nitrile | 0.998 ± 0.0007 | 0.999 ± 0.0002 | 0.214 ± 0.0055 |
| Alkene | 0.985 ± 0.0012 | 0.989 ± 0.0018 | 0.580 ± 0.0154 |
| Conjugation | 0.996 ± 0.0007 | 0.998 ± 0.0003 | 0.879 ± 0.0028 |
| Aromatic system | 0.999 ± 0.0002 | 0.999 ± 0.0003 | 0.774 ± 0.0394 |
| Heteroaromatic | 0.998 ± 0.0009 | 0.997 ± 0.0003 | 0.766 ± 0.0479 |
| Donor–acceptor | 0.932 ± 0.0099 | 0.926 ± 0.0078 | 0.347 ± 0.0284 |

## Files and SHA256

| File | Bytes | SHA256 |
|---|---:|---|
| Fig3_PerLabel_TestAP_Heatmap.pdf | 47,435 | `8364a971b6c46f0472a5f6668bd20d83a607748ff600e11baa8d79d1daf732fb` |
| Fig3_PerLabel_TestAP_Heatmap.png | 130,936 | `de52a754a1d92b5b144a619503933bd2aa3c376dd9058d619d598856d13955dc` |
| Fig3_PerLabel_TestAP_Heatmap.svg | 168,461 | `4bb5693388a6826e4cca0f730f8d8a8ab8da23de62f7573576e0add2e89755dc` |
| Fig3_PerLabel_TestAP_Heatmap.tex | 10,369 | `3cd5cc43225e3d7476ec3920116b86337871fd51c0bb295ef5be73f9214c615c` |

## Notes

- This is a **curated subset** (4 + 4 labels); the complete 14/15-label AP and F1 tables are in `runs/LINE-B-TEST-ANALYSIS/B4.2/per_label_metrics.csv` and `runs/LINE-B-FINAL/tables/per_label_metrics.csv`.
- AP measures label ranking/separation, not total chemical-information content; QM9S is computational, not experimental.
- The figure is a descriptive Test visualization (exploratory/post-hoc), not confirmatory evidence.
