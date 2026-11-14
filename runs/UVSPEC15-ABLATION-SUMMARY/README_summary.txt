UVSPEC15 complete ablation summary

All six original checkpoints and six BN checkpoints
24 conditions per split; 8 conditions x 3 seeds

Mean ± sample SD on Test (n=3)

Loss | BN | Threshold | Micro-F1 | Macro-F1 | mAP | Macro-AUROC
BCE | Original | 0.5 | 0.6136 +/- 0.0845 | 0.4397 +/- 0.0744 | 0.5389 +/- 0.0599 | 0.9259 +/- 0.0204
BCE | Original | Validation-F1 | 0.6325 +/- 0.0418 | 0.5104 +/- 0.0545 | 0.5389 +/- 0.0599 | 0.9259 +/- 0.0204
BCE | Recalibrated | 0.5 | 0.6580 +/- 0.0154 | 0.4721 +/- 0.0271 | 0.5728 +/- 0.0203 | 0.9384 +/- 0.0040
BCE | Recalibrated | Validation-F1 | 0.6568 +/- 0.0138 | 0.5395 +/- 0.0143 | 0.5728 +/- 0.0203 | 0.9384 +/- 0.0040
WBCE | Original | 0.5 | 0.5682 +/- 0.1017 | 0.4009 +/- 0.0736 | 0.5095 +/- 0.0204 | 0.9262 +/- 0.0035
WBCE | Original | Validation-F1 | 0.6210 +/- 0.0097 | 0.4933 +/- 0.0126 | 0.5095 +/- 0.0204 | 0.9262 +/- 0.0035
WBCE | Recalibrated | 0.5 | 0.6465 +/- 0.0089 | 0.5264 +/- 0.0166 | 0.5596 +/- 0.0226 | 0.9382 +/- 0.0039
WBCE | Recalibrated | Validation-F1 | 0.6520 +/- 0.0116 | 0.5330 +/- 0.0154 | 0.5596 +/- 0.0226 | 0.9382 +/- 0.0039

Selection policy: use Validation for strategy selection; Test scores are evaluation only.
The result table does not establish that within-training epoch oscillations were eliminated.
All source results, checkpoints, and thresholds were read-only.
