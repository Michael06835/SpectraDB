# UVVIS-v1 final 3-seed summary

Status: **CURRENT_FINAL**

The original `EXP-UVVIS-001-SEED2026` was trained with `SEED=42`; the archived
pre-correction snapshot is preserved under
`archive/2026-11-17/UVVIS-SHARED14-INVALID-SEED2026/superseded_uvvis_v1_final/`.
The corrected run uses the same experiment ID, `SEED=2026`, and the original
training/evaluation protocol.

## Test (historical AMP evaluation, 3 seeds)

| Metric | Mean ± sample SD |
|---|---:|
| Micro-F1 | 0.5158 ± 0.0067 |
| Macro-F1 | 0.4130 ± 0.0232 |
| Micro-Precision | 0.6590 ± 0.0113 |
| Micro-Recall | 0.4237 ± 0.0066 |
| mAP | 0.5126 ± 0.0011 |
| Macro-AUROC | 0.8526 ± 0.0016 |

## Validation (B1 FP32 predictions, 3 seeds)

| Metric | Mean ± sample SD |
|---|---:|
| Micro-F1 | 0.5024 ± 0.0047 |
| Macro-F1 | 0.3946 ± 0.0144 |
| Macro-AP | 0.5131 ± 0.0031 |
| Macro-AUROC | 0.8489 ± 0.0008 |

The historical Test summary is separate from Line B B4's unified FP32 Test
evaluation. No Test result was used to select model, loss, threshold, BN or
hyperparameters.
