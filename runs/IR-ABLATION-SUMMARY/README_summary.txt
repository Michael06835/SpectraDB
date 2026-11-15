IR v1 complete ablation summary (24 conditions per split)

2 loss x 3 seed x 2 BN x 2 threshold = 24 conditions; Validation 24 + Test 24.
Per-label rows: 14 labels x 24 conditions x 2 splits = 672.
All values are read from recorded JSON/CSV; no forward pass, no threshold refit.

Test, mean +/- sample SD over 3 seeds (n=3)
Loss | BN | Threshold | Micro-F1 | Macro-F1 | mAP | Macro-AUROC
BCE | Original | 0.5 | 0.9405 +/- 0.0048 | 0.9193 +/- 0.0050 | 0.9688 +/- 0.0050 | 0.9954 +/- 0.0007
BCE | Original | Validation-F1 | 0.9364 +/- 0.0054 | 0.9199 +/- 0.0076 | 0.9688 +/- 0.0050 | 0.9954 +/- 0.0007
BCE | Recalibrated | 0.5 | 0.9376 +/- 0.0071 | 0.9179 +/- 0.0118 | 0.9685 +/- 0.0067 | 0.9953 +/- 0.0006
BCE | Recalibrated | Validation-F1 | 0.9350 +/- 0.0068 | 0.9198 +/- 0.0076 | 0.9685 +/- 0.0067 | 0.9953 +/- 0.0006
WBCE | Original | 0.5 | 0.9328 +/- 0.0085 | 0.9091 +/- 0.0169 | 0.9670 +/- 0.0069 | 0.9949 +/- 0.0009
WBCE | Original | Validation-F1 | 0.9317 +/- 0.0076 | 0.9083 +/- 0.0227 | 0.9670 +/- 0.0069 | 0.9949 +/- 0.0009
WBCE | Recalibrated | 0.5 | 0.9347 +/- 0.0081 | 0.9097 +/- 0.0120 | 0.9676 +/- 0.0058 | 0.9951 +/- 0.0007
WBCE | Recalibrated | Validation-F1 | 0.9324 +/- 0.0079 | 0.9091 +/- 0.0216 | 0.9676 +/- 0.0058 | 0.9951 +/- 0.0007

Validation, mean +/- sample SD over 3 seeds (n=3)
Loss | BN | Threshold | Micro-F1 | Macro-F1 | mAP | Macro-AUROC
BCE | Original | 0.5 | 0.9470 +/- 0.0018 | 0.9165 +/- 0.0066 | 0.9604 +/- 0.0065 | 0.9946 +/- 0.0020
BCE | Original | Validation-F1 | 0.9505 +/- 0.0025 | 0.9268 +/- 0.0045 | 0.9604 +/- 0.0065 | 0.9946 +/- 0.0020
BCE | Recalibrated | 0.5 | 0.9438 +/- 0.0037 | 0.9179 +/- 0.0087 | 0.9607 +/- 0.0067 | 0.9957 +/- 0.0001
BCE | Recalibrated | Validation-F1 | 0.9503 +/- 0.0023 | 0.9299 +/- 0.0035 | 0.9607 +/- 0.0067 | 0.9957 +/- 0.0001
WBCE | Original | 0.5 | 0.9379 +/- 0.0067 | 0.9070 +/- 0.0102 | 0.9588 +/- 0.0027 | 0.9940 +/- 0.0014
WBCE | Original | Validation-F1 | 0.9450 +/- 0.0064 | 0.9221 +/- 0.0085 | 0.9588 +/- 0.0027 | 0.9940 +/- 0.0014
WBCE | Recalibrated | 0.5 | 0.9398 +/- 0.0068 | 0.9068 +/- 0.0049 | 0.9595 +/- 0.0022 | 0.9952 +/- 0.0008
WBCE | Recalibrated | Validation-F1 | 0.9458 +/- 0.0060 | 0.9223 +/- 0.0072 | 0.9595 +/- 0.0022 | 0.9952 +/- 0.0008

Paired effects on Validation (mean +/- sample SD of the per-seed difference)
BN recalibration           BCE/Original/0.5 -> BCE/Recalibrated/0.5: dMicro-F1=-0.0032 +/- 0.0032, dMacro-F1=+0.0014 +/- 0.0023, dmAP=+0.0003 +/- 0.0018
BN recalibration           WBCE/Original/0.5 -> WBCE/Recalibrated/0.5: dMicro-F1=+0.0020 +/- 0.0003, dMacro-F1=-0.0002 +/- 0.0066, dmAP=+0.0007 +/- 0.0017
BN recalibration           BCE/Original/Validation-F1 -> BCE/Recalibrated/Validation-F1: dMicro-F1=-0.0001 +/- 0.0007, dMacro-F1=+0.0030 +/- 0.0016, dmAP=+0.0003 +/- 0.0018
BN recalibration           WBCE/Original/Validation-F1 -> WBCE/Recalibrated/Validation-F1: dMicro-F1=+0.0008 +/- 0.0003, dMacro-F1=+0.0001 +/- 0.0013, dmAP=+0.0007 +/- 0.0017
BN x threshold interaction BCE/Original->Recalibrated @0.5 -> BCE/Original->Recalibrated @Validation-F1: dMicro-F1=+0.0031 +/- 0.0028, dMacro-F1=+0.0016 +/- 0.0024, dmAP=+0.0000 +/- 0.0000
BN x threshold interaction WBCE/Original->Recalibrated @0.5 -> WBCE/Original->Recalibrated @Validation-F1: dMicro-F1=-0.0012 +/- 0.0005, dMacro-F1=+0.0003 +/- 0.0058, dmAP=+0.0000 +/- 0.0000
Threshold calibration      BCE/Original/0.5 -> BCE/Original/Validation-F1: dMicro-F1=+0.0035 +/- 0.0012, dMacro-F1=+0.0103 +/- 0.0043, dmAP=+0.0000 +/- 0.0000
Threshold calibration      WBCE/Original/0.5 -> WBCE/Original/Validation-F1: dMicro-F1=+0.0071 +/- 0.0011, dMacro-F1=+0.0151 +/- 0.0056, dmAP=+0.0000 +/- 0.0000
Threshold calibration      BCE/Recalibrated/0.5 -> BCE/Recalibrated/Validation-F1: dMicro-F1=+0.0066 +/- 0.0016, dMacro-F1=+0.0119 +/- 0.0059, dmAP=+0.0000 +/- 0.0000
Threshold calibration      WBCE/Recalibrated/0.5 -> WBCE/Recalibrated/Validation-F1: dMicro-F1=+0.0060 +/- 0.0016, dMacro-F1=+0.0154 +/- 0.0023, dmAP=+0.0000 +/- 0.0000
WBCE minus BCE             BCE/Original/0.5 -> WBCE/Original/0.5: dMicro-F1=-0.0092 +/- 0.0066, dMacro-F1=-0.0095 +/- 0.0157, dmAP=-0.0017 +/- 0.0076
WBCE minus BCE             BCE/Original/Validation-F1 -> WBCE/Original/Validation-F1: dMicro-F1=-0.0055 +/- 0.0056, dMacro-F1=-0.0047 +/- 0.0076, dmAP=-0.0017 +/- 0.0076
WBCE minus BCE             BCE/Recalibrated/0.5 -> WBCE/Recalibrated/0.5: dMicro-F1=-0.0040 +/- 0.0034, dMacro-F1=-0.0111 +/- 0.0114, dmAP=-0.0012 +/- 0.0067
WBCE minus BCE             BCE/Recalibrated/Validation-F1 -> WBCE/Recalibrated/Validation-F1: dMicro-F1=-0.0046 +/- 0.0047, dMacro-F1=-0.0076 +/- 0.0074, dmAP=-0.0012 +/- 0.0067

Selection policy (per docs/experiments/IR_v1_experimental_strategy.md)
- Checkpoints are frozen; only Validation may be used to compare methods.
- Test is evaluation only and never participates in selection.
- No Seed is chosen by Test score.
- No single primary selection metric is defined in the current documents,
  so this summary reports multi-metric Validation comparisons and leaves
  the configuration decision to the reviewer. It does not invent a rule.
- BN recalibration and threshold calibration are NOT assumed to be beneficial.
- n=3 seeds: no significance claims; SD is descriptive only.

Support tiers for per-label analysis (see per_label_f1_levels.csv)
- very_low: min(val,test) positives < 30 (e.g. nitro: 11 val / 15 test)
- low     : min(val,test) positives < 150
- Threshold fitting on 11 validation positives is unstable; treat nitro results
  as indicative only, and do not let a single sparse label drive the conclusion.

Validation-vs-Test consistency
Per-label thresholds are optimized on Validation, making their in-sample F1
estimates optimistic. Improvement in Micro-F1 is not guaranteed.
Compare the Validation and Test columns; the summary deliberately does not
use Test to re-rank configurations.

Metric sources (archival note)
- F1-dependent metrics (Micro/Macro-F1, Micro-Precision, Micro-Recall) are read
  from the matching threshold_calibration_summary.json:
  0.5 -> validation_default / test_default; Validation-F1 -> *_calibrated.
- Threshold-free metrics (mAP, Macro-AUROC): Test from summary.test_threshold_free;
  Validation from the BN diagnostic block (before=Original, after=Recalibrated),
  because the calibration summaries do not record validation_threshold_free.
- The 0.5 F1 metrics are NOT read from the BN diagnostic JSON; both sources are
  verified to agree within 0.0025 before any condition row is emitted.

Output files
- test_24_conditions.csv
- validation_24_conditions.csv
- mean_sd_8_conditions_each_split.csv
- paired_effects.csv
- paired_effects_summary.csv
- per_label_672_rows.csv
- per_label_f1_levels.csv
- per_label_deltas.csv
- condition_registry.csv
- audit_checks.csv
