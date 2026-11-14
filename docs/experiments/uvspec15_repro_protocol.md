# UVSPEC15 v1 实验复现协议

状态：与 `docs/experiments/UVSPEC15_v1_experimental_strategy.md`（决策记录）配套的**可执行复现指南**。
范围：QM9S 计算域 UV-Vis → UVSPEC15 的 15 类结构语义标签单模态 1D-CNN 基线；不包含多模态融合。

## 1. 环境

- Python 3.12.13、PyTorch（CUDA）、NumPy 2.4.6、RDKit 2026.03.3、scikit-learn（指标）、pandas。
- 训练设备：单卡（本实验在 RTX 5070 Laptop 8GB 上完成；batch 512 + AMP；若 OOM 可降至 256 并在 config 中记录）。
- 环境锁定文件：`environment.yml`、`explicit-win-64.txt`、`requirements-pip-exact.txt`。

## 2. 数据与标签

- 光谱：`processed/qm9s/prepared/uvvis_float32.npy`（129,817×701，1–15 eV；2 条全零谱已在 valid split 中排除）。
- 标签：`processed/qm9s/prepared/uvspec15_labels.npy`（129,817×15，float32），顺序固定于 `uvspec15_label_names.json`；生成脚本 `scripts/prepare_uvspec15_labels.py`，provenance 见 `uvspec15_label_manifest.json`。
- 划分：`processed/qm9s/prepared/uvvis_scaffold_split_valid.npz`（train 103,853 / val 12,981 / test 12,981）；不得重划分。

## 3. 训练（每个随机种子独立一次）

1. 固定配置训练 6 个模型（普通 BCE / WBCE × seed 42/123/2026）：
   - 架构：`training/models/spectrum_cnn1d.py` 的 `SpectrumStructureModel`（embedding 256、15 logits）。
   - 训练：batch 512、num_workers 4、AdamW（lr 1e-3、wd 1e-4）、max 50 epochs、early-stopping patience 8、AMP、per-spectrum max 归一化。
   - WBCE：sqrt(pos_weight) 变换、权重上限 20。
2. 训练脚本：`training/train_uvvis_uvspec15.py`（BCE seed42）、`_seed123`、`_seed2026`；`_weighted.py`（WBCE seed42）、`_weighted_seed123`、`_weighted_seed2026`。
3. 产出：`runs/EXP-UVVIS-003-*`（BCE）与 `runs/EXP-UVVIS-004-*`（WBCE），含 `config.json`、`training_log.csv`、`best_model.pt`、`test_metrics.json`、`test_per_label_metrics.csv`。

## 4. 后处理（每个模型独立执行）

1. **Full BN 诊断**：`training/diagnose_uvspec15_batchnorm_multiseed.py` → `runs/UVSPEC15-BN-DIAGNOSTICS/<run>/batchnorm_full_diagnostic.json`（train-only 重算 running stats，权重不变）。
2. **BN 重估 checkpoint**：`training/save_uvspec15_bn_checkpoint.py` → `runs/UVSPEC15-BN-RECALIBRATED/<run>/best_model_bn_recalibrated.pt` + `bn_recalibration_manifest.json`。
3. **BN 后阈值校准**：`training/calibrate_uvspec15_bn_thresholds.py` → `<run>/threshold_calibration/`（Validation-only 逐标签 F1 最优阈值）。
4. **原始阈值校准（对照）**：`training/calibrate_uvvis_uvspec15_thresholds*.py` 在原始 checkpoint 上执行，产出 `test_per_label_metrics_calibrated.csv` 等。

## 5. 汇总与核验

```powershell
python training\summarize_uvspec15_ablation.py
```

- 只读脚本：仅读取记录 JSON/CSV，不加载 PyTorch、不重训、不重算预测、不修改 checkpoint、不用 Test 选择配置。
- 输出 `runs/UVSPEC15-ABLATION-SUMMARY/`（仅当全部核验通过才创建）；应看到 `PASS: 6 runs, 24 Test conditions, 24 Validation conditions, 720 per-label rows`，且 `audit_checks.csv` 三项检查全为 True。

## 6. 复现验收检查清单

- [ ] 6 个 run 的 `config.seed` 与目录名一致（42/123/2026）。
- [ ] 每个 BN manifest 的 `source_checkpoint` 指向对应原始 run，`seed` 一致。
- [ ] 每个 BN 后阈值 summary 的 `checkpoint` 指向 BN 重估版，`seed` 一致。
- [ ] Test/Validation 各 24 条件、per-label 720 行、audit 全 True。
- [ ] mAP/AUROC 仅在 `test_threshold_free` 中报告（与阈值无关）。

## 7. 注意事项

- 不覆盖原始 checkpoint；BN 重估使用独立副本（train-only、无梯度更新）。
- 不依据 Test 选择损失、BN 处理、阈值或随机种子；配置选择依据 Validation。
- 大文件（`.pt`、`.npy`、split）不入 Git，须另行安全备份并保留与 config/manifest/阈值的对应关系。
