# UV-Vis 单模态模型 v1 最终实验总结

状态：**FROZEN**

## 1. 最终模型设置

- Backbone：1D-CNN / residual 1D CNN
- 输入：QM9S UV-Vis，701 points
- 数据划分：UV-Vis valid scaffold split
- 任务：14 类共享官能团 / 结构特征多标签预测
- Loss：BCEWithLogitsLoss
- 统一 canonical 判定阈值：0.5
- Normalization：per-spectrum max normalization
- Canonical checkpoint：Seed 42

注意：Seed 42 为预先固定的 canonical checkpoint；不依据 test set 在多个随机种子中挑选最佳 checkpoint。

## 2. 三随机种子最终结果

| Metric | Mean ± sample SD |
|---|---:|
| Micro-F1 | 0.4892 ± 0.0406 |
| Macro-F1 | 0.3737 ± 0.0566 |
| Precision | 0.6536 ± 0.0128 |
| Recall | 0.3921 ± 0.0482 |
| mAP | 0.5053 ± 0.0127 |
| Macro-AUROC | 0.8501 ± 0.0044 |

## 3. 阈值校准

逐标签 validation-based threshold calibration 在独立 scaffold test 上产生：

| Setting | Micro-F1 | Macro-F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| threshold=0.5 | 0.5161 | 0.4229 | 0.6683 | 0.4203 |
| calibrated | 0.5790 | 0.5019 | 0.4849 | 0.7184 |

校准显著提高 Recall、Micro-F1 和 Macro-F1，同时降低 Precision。由于 mAP / AUROC 与阈值无关，且三模态基线需保持统一评价协议，UVVIS-v1 的 canonical threshold 仍固定为 0.5；validation-derived thresholds 作为辅助结果保留。

## 4. 类别不平衡实验

| Setting | Micro-F1 | Macro-F1 | Precision | Recall | mAP | AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary BCE | 0.5161 | 0.4229 | 0.6684 | 0.4203 | 0.5137 | 0.8540 |
| Weighted BCE | 0.4616 | 0.3729 | 0.4808 | 0.4438 | 0.4212 | 0.8151 |

采用 sqrt(pos_weight) 并将权重截断至 20 后，Recall 略有提高，但 Precision、Micro-F1、Macro-F1、mAP 和 Macro-AUROC 均下降。因此 UVVIS-v1 不采用 class weighting。

## 5. BatchNorm 诊断

Recomputing BatchNorm running statistics using the training split changed test mAP by +0.0031 and Macro-AUROC by +0.0026. Therefore BatchNorm calibration is not the primary cause of the low UV-Vis generic-label baseline performance.

## 6. 三随机种子表现相对较弱的类别

| Label | F1 mean | F1 SD | AP mean | AUROC mean |
|---|---:|---:|---:|---:|
| nitro | 0.0000 | 0.0000 | 0.1653 | 0.9837 |
| nitrile | 0.0306 | 0.0284 | 0.2125 | 0.6701 |
| hydroxyl | 0.1632 | 0.0810 | 0.5134 | 0.6790 |
| imine | 0.2177 | 0.1031 | 0.3935 | 0.8370 |
| c_f_bond | 0.2903 | 0.0988 | 0.3191 | 0.9404 |

## 7. 当前解释

在完全统一的共享 14 标签任务下，UV-Vis 对通用官能团 / 局部结构特征的预测能力明显弱于振动光谱基线。BatchNorm 重校准和类别加权均不能解释或消除这一差异。因此该结果作为 UV-Vis 的 generic structural-feature baseline 保留。它并不意味着 UV-Vis 缺乏结构信息；后续将以共轭体系、发色团及电子结构相关标签进一步检验 UV-Vis 的模态专属信息。

## 8. 输出文件

- `uvvis_seed_summary.csv`
- `uvvis_seed_summary.json`
- `uvvis_per_label_3seed_summary.csv`
- `uvvis_ablation_summary.csv`
- `training_loss_seed42.png`
- `validation_metrics_seed42.png`
- `per_label_f1_3seed.png`
- `bce_weighting_comparison.png`
- `uvvis_final_summary.json`
- `UVVIS_FINAL_SUMMARY.md`