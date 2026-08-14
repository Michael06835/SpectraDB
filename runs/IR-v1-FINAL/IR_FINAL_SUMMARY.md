# IR 单模态模型 v1 最终实验总结

状态：**FROZEN**

## 1. 最终模型设置

- Backbone：1D-CNN / residual 1D CNN
- 输入：QM9S IR，3501 points
- 数据划分：scaffold split
- Loss：BCEWithLogitsLoss
- 统一判定阈值：0.5
- Normalization：per-spectrum max normalization
- Canonical checkpoint：Seed 42

注意：Seed 42 为预先固定的 canonical checkpoint；不依据 test set 在多个随机种子中挑选最佳 checkpoint。

## 2. 三随机种子最终结果

| Metric | Mean ± sample SD |
|---|---:|
| Micro-F1 | 0.9406 ± 0.0048 |
| Macro-F1 | 0.9193 ± 0.0049 |
| Precision | 0.9384 ± 0.0054 |
| Recall | 0.9427 ± 0.0043 |
| mAP | 0.9689 ± 0.0050 |
| Macro-AUROC | 0.9954 ± 0.0007 |

## 3. 阈值校准

逐标签 validation-based threshold calibration 可提升 validation Macro-F1，但在独立 scaffold test 上未改善整体 Micro-F1，因此 IR-v1 保留统一阈值 0.5。

## 4. 类别不平衡实验

采用 sqrt(pos_weight) 并将权重截断至 20 后，模型 Recall 提高，但 Precision、Micro-F1、Macro-F1 和 mAP 均下降。因此 IR-v1 不采用 class weighting。

## 5. 三随机种子表现相对较弱的类别

| Label | F1 mean | F1 SD |
|---|---:|---:|
| c_f_bond | 0.7765 | 0.0120 |
| amine | 0.8474 | 0.0205 |
| ketone | 0.8686 | 0.0118 |
| nitro | 0.8798 | 0.0291 |
| ester | 0.8867 | 0.0104 |

## 6. 输出文件

- `ir_seed_summary.csv`
- `ir_seed_summary.json`
- `ir_per_label_3seed_summary.csv`
- `ir_ablation_summary.csv`
- `training_loss_seed42.png`
- `validation_metrics_seed42.png`
- `per_label_f1_3seed.png`
- `bce_weighting_comparison.png`
- `ir_final_summary.json`