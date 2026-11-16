Raman v1 complete ablation summary (24 conditions per split)

2 loss x 3 seed x 2 BN x 2 threshold = 24 conditions; Validation 24 + Test 24.
Per-label rows: 14 labels x 24 conditions x 2 splits = 672.
All values are read from recorded JSON/CSV; no forward pass, no threshold refit.

Test, mean +/- sample SD over 3 seeds (n=3)
Loss | BN | Threshold | Micro-F1 | Macro-F1 | mAP | Macro-AUROC
BCE | Original | 0.5 | 0.9151 +/- 0.0014 | 0.8885 +/- 0.0079 | 0.9548 +/- 0.0044 | 0.9911 +/- 0.0004
BCE | Original | Validation-F1 | 0.9143 +/- 0.0018 | 0.9000 +/- 0.0045 | 0.9548 +/- 0.0044 | 0.9911 +/- 0.0004
BCE | Recalibrated | 0.5 | 0.9190 +/- 0.0022 | 0.8989 +/- 0.0092 | 0.9571 +/- 0.0040 | 0.9915 +/- 0.0006
BCE | Recalibrated | Validation-F1 | 0.9150 +/- 0.0034 | 0.9033 +/- 0.0038 | 0.9571 +/- 0.0040 | 0.9915 +/- 0.0006
WBCE | Original | 0.5 | 0.9124 +/- 0.0090 | 0.9063 +/- 0.0096 | 0.9610 +/- 0.0052 | 0.9918 +/- 0.0006
WBCE | Original | Validation-F1 | 0.9157 +/- 0.0056 | 0.9109 +/- 0.0058 | 0.9610 +/- 0.0052 | 0.9918 +/- 0.0006
WBCE | Recalibrated | 0.5 | 0.9129 +/- 0.0114 | 0.9082 +/- 0.0099 | 0.9620 +/- 0.0055 | 0.9918 +/- 0.0009
WBCE | Recalibrated | Validation-F1 | 0.9141 +/- 0.0069 | 0.9121 +/- 0.0057 | 0.9620 +/- 0.0055 | 0.9918 +/- 0.0009

Validation, mean +/- sample SD over 3 seeds (n=3)
Loss | BN | Threshold | Micro-F1 | Macro-F1 | mAP | Macro-AUROC
BCE | Original | 0.5 | 0.9101 +/- 0.0024 | 0.8746 +/- 0.0120 | 0.9472 +/- 0.0027 | 0.9883 +/- 0.0001
BCE | Original | Validation-F1 | 0.9178 +/- 0.0010 | 0.9048 +/- 0.0033 | 0.9472 +/- 0.0027 | 0.9883 +/- 0.0001
BCE | Recalibrated | 0.5 | 0.9171 +/- 0.0036 | 0.8821 +/- 0.0161 | 0.9488 +/- 0.0034 | 0.9891 +/- 0.0009
BCE | Recalibrated | Validation-F1 | 0.9209 +/- 0.0035 | 0.9078 +/- 0.0052 | 0.9488 +/- 0.0034 | 0.9891 +/- 0.0009
WBCE | Original | 0.5 | 0.9159 +/- 0.0047 | 0.9037 +/- 0.0060 | 0.9535 +/- 0.0056 | 0.9896 +/- 0.0006
WBCE | Original | Validation-F1 | 0.9216 +/- 0.0026 | 0.9117 +/- 0.0045 | 0.9535 +/- 0.0056 | 0.9896 +/- 0.0006
WBCE | Recalibrated | 0.5 | 0.9149 +/- 0.0061 | 0.9034 +/- 0.0067 | 0.9543 +/- 0.0050 | 0.9897 +/- 0.0004
WBCE | Recalibrated | Validation-F1 | 0.9220 +/- 0.0021 | 0.9136 +/- 0.0041 | 0.9543 +/- 0.0050 | 0.9897 +/- 0.0004

Paired effects on Validation (mean +/- sample SD of the per-seed difference)
BN recalibration           BCE/Original/0.5 -> BCE/Recalibrated/0.5: dMicro-F1=+0.0070 +/- 0.0047, dMacro-F1=+0.0075 +/- 0.0050, dmAP=+0.0016 +/- 0.0023
BN recalibration           WBCE/Original/0.5 -> WBCE/Recalibrated/0.5: dMicro-F1=-0.0010 +/- 0.0022, dMacro-F1=-0.0003 +/- 0.0023, dmAP=+0.0008 +/- 0.0010
BN recalibration           BCE/Original/Validation-F1 -> BCE/Recalibrated/Validation-F1: dMicro-F1=+0.0031 +/- 0.0026, dMacro-F1=+0.0030 +/- 0.0026, dmAP=+0.0016 +/- 0.0023
BN recalibration           WBCE/Original/Validation-F1 -> WBCE/Recalibrated/Validation-F1: dMicro-F1=+0.0004 +/- 0.0006, dMacro-F1=+0.0018 +/- 0.0016, dmAP=+0.0008 +/- 0.0010
BN x threshold interaction BCE/Original->Recalibrated @0.5 -> BCE/Original->Recalibrated @Validation-F1: dMicro-F1=-0.0039 +/- 0.0033, dMacro-F1=-0.0045 +/- 0.0038, dmAP=+0.0000 +/- 0.0000
BN x threshold interaction WBCE/Original->Recalibrated @0.5 -> WBCE/Original->Recalibrated @Validation-F1: dMicro-F1=+0.0014 +/- 0.0028, dMacro-F1=+0.0022 +/- 0.0039, dmAP=+0.0000 +/- 0.0000
Threshold calibration      BCE/Original/0.5 -> BCE/Original/Validation-F1: dMicro-F1=+0.0077 +/- 0.0029, dMacro-F1=+0.0302 +/- 0.0093, dmAP=+0.0000 +/- 0.0000
Threshold calibration      WBCE/Original/0.5 -> WBCE/Original/Validation-F1: dMicro-F1=+0.0058 +/- 0.0023, dMacro-F1=+0.0080 +/- 0.0016, dmAP=+0.0000 +/- 0.0000
Threshold calibration      BCE/Recalibrated/0.5 -> BCE/Recalibrated/Validation-F1: dMicro-F1=+0.0038 +/- 0.0004, dMacro-F1=+0.0257 +/- 0.0129, dmAP=+0.0000 +/- 0.0000
Threshold calibration      WBCE/Recalibrated/0.5 -> WBCE/Recalibrated/Validation-F1: dMicro-F1=+0.0072 +/- 0.0041, dMacro-F1=+0.0102 +/- 0.0039, dmAP=+0.0000 +/- 0.0000
WBCE minus BCE             BCE/Original/0.5 -> WBCE/Original/0.5: dMicro-F1=+0.0057 +/- 0.0059, dMacro-F1=+0.0292 +/- 0.0104, dmAP=+0.0063 +/- 0.0048
WBCE minus BCE             BCE/Original/Validation-F1 -> WBCE/Original/Validation-F1: dMicro-F1=+0.0038 +/- 0.0016, dMacro-F1=+0.0069 +/- 0.0027, dmAP=+0.0063 +/- 0.0048
WBCE minus BCE             BCE/Recalibrated/0.5 -> WBCE/Recalibrated/0.5: dMicro-F1=-0.0023 +/- 0.0032, dMacro-F1=+0.0213 +/- 0.0150, dmAP=+0.0055 +/- 0.0017
WBCE minus BCE             BCE/Recalibrated/Validation-F1 -> WBCE/Recalibrated/Validation-F1: dMicro-F1=+0.0011 +/- 0.0016, dMacro-F1=+0.0058 +/- 0.0012, dmAP=+0.0055 +/- 0.0017

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
