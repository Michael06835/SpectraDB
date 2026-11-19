Line B B1 · Shared14 Validation metrics
Threshold: 0.5
Split: validation only (no Test access)

Per-modality mean ± sample SD (n=3 seeds; see seed_audit.csv)
Modality | Micro-F1 | Macro-F1 | Macro-AP | Macro-AUROC
---|---|---:|---:|---:|---:
ir | 0.947132 ± 0.001773 | 0.916688 ± 0.006500 | 0.961348 ± 0.006308 | 0.995872 ± 0.000247
raman | 0.910127 ± 0.002433 | 0.874298 ± 0.011928 | 0.947568 ± 0.002625 | 0.988322 ± 0.000071
uvvis | 0.502378 ± 0.004709 | 0.394614 ± 0.014393 | 0.513131 ± 0.003081 | 0.848895 ± 0.000822

Per-model metrics: model_metrics.csv
Per-label metrics: per_label_all_models.csv
Per-label mean/SD: per_label_mean_sd.csv
FP16 reference deltas: fp16_reference_delta.csv
Summary: summary.json
