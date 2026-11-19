Line B B3 · UVSPEC15 Validation metrics
Threshold: 0.5
Split: validation only (no Test access)

Per-modality mean ± sample SD (n=3 seeds; see seed_audit.csv)
Modality | Micro-F1 | Macro-F1 | Macro-AP | Macro-AUROC
---|---|---:|---:|---:|---:
ir | 0.932186 ± 0.009083 | 0.890973 ± 0.015049 | 0.950187 ± 0.007271 | 0.996579 ± 0.000505
raman | 0.935184 ± 0.004631 | 0.900942 ± 0.005102 | 0.955733 ± 0.000538 | 0.996334 ± 0.000360
uvvis | 0.573104 ± 0.065093 | 0.412766 ± 0.053242 | 0.501388 ± 0.049551 | 0.911373 ± 0.018558

Per-model metrics: model_metrics.csv
Per-label metrics: per_label_all_models.csv
Per-label mean/SD: per_label_mean_sd.csv
FP16 reference deltas: fp16_reference_delta.csv
Summary: summary.json
