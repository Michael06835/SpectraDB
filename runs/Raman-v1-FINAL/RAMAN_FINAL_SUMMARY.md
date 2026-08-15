# Raman 单模态模型 v1 最终实验总结

状态：**FROZEN**

## 1. 最终模型设置

- Backbone：1D-CNN / residual 1D CNN
- 输入：QM9S Raman，3501 points
- 数据划分：scaffold split
- Loss：BCEWithLogitsLoss
- 统一判定阈值：0.5
- Normalization：per-spectrum max normalization
- Canonical checkpoint：Seed 42

注意：Seed 42 为预先固定的 canonical checkpoint；不依据 test set 在多个随机种子中挑选最佳 checkpoint。

## 2. 三随机种子最终结果

| Metric | Mean ± sample SD |
|---|---:|
| Micro-F1 | 0.9151 ± 0.0014 |
| Macro-F1 | 0.8885 ± 0.0079 |
| Precision | 0.9257 ± 0.0023 |
| Recall | 0.9047 ± 0.0008 |
| mAP | 0.9548 ± 0.0044 |
| Macro-AUROC | 0.9911 ± 0.0004 |

## 3. 阈值校准

逐标签 validation-based threshold calibration 可提升独立 scaffold test 上的 Recall 和 Macro-F1，但 Precision 与 Micro-F1 下降，因此 Raman-v1 保留统一阈值 0.5。

## 4. 类别不平衡实验

采用 sqrt(pos_weight) 并将权重截断至 20 后，模型 Recall、Macro-F1 和 mAP 略有提高，但 Precision 和 Micro-F1 下降。因此 Raman-v1 不采用 class weighting。

## 5. 三随机种子表现相对较弱的类别

| Label | F1 mean | F1 SD |
|---|---:|---:|
| c_f_bond | 0.7299 | 0.0681 |
| ester | 0.7796 | 0.0155 |
| nitro | 0.8204 | 0.1361 |
| ketone | 0.8354 | 0.0186 |
| amine | 0.8399 | 0.0079 |

## 6. 输出文件

- `raman_seed_summary.csv`
- `raman_seed_summary.json`
- `raman_per_label_3seed_summary.csv`
- `raman_ablation_summary.csv`
- `training_loss_seed42.png`
- `validation_metrics_seed42.png`
- `per_label_f1_3seed.png`
- `bce_weighting_comparison.png`
- `raman_final_summary.json`