# Line B B3 — Verifier 分析根范围修复记录（唯一 FREEZE 阻塞项）

> **日期**：2026-11-20
> **类型**：用户授权的严格限定 Verifier 最小修改（不改历史 provenance、不改 B3-A 资产）
> **范围**：仅 `training/line_b/verify_line_b_manifests.py`；`training/line_b/freeze_line_b.py` 做最小兼容记录
> **Test 状态**：未访问 Test split / Test 索引 / Test 标签 / Test 概率；未训练、未重新推理、未重拟合阈值
> **FREEZE 状态**：本轮未生成 `FREEZE.json`（等待用户下一步指令）

---

## 1. 问题

B3 FREEZE 预检唯一失败项：

```text
FAILED  provenance.roots.sha  @ runs/LINE-B-ANALYSIS
```

- B3.5-A provenance 生成于 `2026-11-18T22:18:57+08:00`，当时 `runs/LINE-B-ANALYSIS` 仅 14 个 B3-A 正式文件，记录树摘要 `5940e063…26d9c8d`（14 文件）。
- 之后新增 `runs/LINE-B-ANALYSIS/validation_development/`（B3.5-C1/C2/D1/D2 后续 Validation-development 资产，共 9027 文件），父目录现有 9041 文件，当前全树摘要 `9a74b072…130984a18`。
- 已确认 14 个 `complementarity` 正式文件逐个 SHA256 与 provenance 一致，其余 5 个正式根摘要一致。**问题是 provenance 根目录范围过宽，而非正式资产被篡改。**

## 2. 修复方案（严格限定）

对 `runs/LINE-B-ANALYSIS` 不再做“当前整目录树摘要 == 历史整目录树摘要”的比较，改为：

1. 从权威 provenance 记录 `formal_assets.b3a_files` 读取原始文件集合（必须恰为 14 条且与 `after_file_count` 一致，否则 FAILED）；
2. 逐文件校验：项目相对路径解析、位于该根内、存在、非空、SHA256 等于记录值；
3. 用**与 `_tree_digest` 完全相同的相对路径语义**（根相对、Windows 原生分隔符、`rel\0sha\n` 排序拼接）从这 14 条记录**重建**历史树摘要，必须等于 `after_tree_sha256`；
4. 独立识别并排除 `validation_development/`：当前树中除 14 个原始文件外的所有文件必须全部位于该子树，否则 FAILED（不得静默忽略其他新增文件）；
5. 任一缺失、空文件、路径不明、哈希不符、摘要无法重建、出现子树外新增文件，一律 FAILED，绝不降级为 PASS。

其余 5 个正式根仍按原逻辑做整根树摘要比对；`provenance.roots` 数量检查（6）保持不变。

## 3. 代码变更

`training/line_b/verify_line_b_manifests.py`

- 新增常量：`B3A_ANALYSIS_ROOT_SUFFIX`、`B3A_ADDITIVE_SUBTREE`。
- 新增 `_verify_b3a_analysis_root(root, record, assets, checks)`：逐文件校验 + 历史摘要重建 + 新增子树识别/排除。
- `_verify_provenance_assets()` 的根目录循环：当根规范化路径以 `runs/LINE-B-ANALYSIS` 结尾时改走上述函数（`provenance.b3a_root`），其余根逻辑不变；返回值新增 `b3a_analysis_root` 明细。
- `verify_project()` 输出新增 `b3a_analysis_root` 与 `verifier_scope` 两个记录块。

`training/line_b/freeze_line_b.py`

- FREEZE 的 `verification` 块记录 `verifier_scope` 与 `b3a_analysis_root_scope`（便于 B4 追溯范围语义）。
- `assert_preflight_ready()` 增加两条守卫：权威文件数必须为 14、重建摘要必须等于记录摘要。

## 4. 验证结果（完整 `verify_project()`）

**PASS：10 / FAILED：0 / ACCEPTED_HISTORICAL_LIMITATION：4（共 14 项检查）**

| 检查 | 结果 |
|---|---|
| `provenance.b3a_root` | **PASS** — 14/14 文件逐个一致；重建摘要 `5940e063…26d9c8d` == 记录值 |
| `provenance.roots` × 5（PREDICTIONS / METRICS / THRESHOLDS / B1 / B3） | PASS |
| 18/18 模型（9 Shared14 + 9 UVSPEC15） | PASS |
| 261 条阈值（126 + 135） | PASS |
| B3-A 6 + 6 正式输出 | PASS |
| `formal_assets.tree_unchanged` | PASS（核验前后全树摘要一致） |
| `no_test_access` | true |
| `future_missing` | 仅 `runs/LINE-B-MANIFESTS/FREEZE.json`（下游预期） |

新增子树识别：`validation_development/` = B3.5-C1 8 + C2 36 + D1 52 + D2 8931 = **9027 文件**，全部被判定为历史快照之后追加的资产；子树外无异常新增文件。

正式资产前后对比：6 个正式根在修复与核验前后摘要完全一致（`unchanged=true`）。14 个 B3-A 文件 recorded vs actual 全部一致。

- 测试 JSON：`runs/LINE-B-MANIFESTS/b3_verifier_scope_fix_verification_20261120T003232.json`
- 修复记录：本文件

## 5. 源码 SHA256

| 文件 | 修复前 | 修复后 |
|---|---|---|
| `training/line_b/verify_line_b_manifests.py` | `8cb2e067828d7ed355f0e518b5713eb01ec4f93f7c1416b45c59bc8defc1feb6` | `c310c5d4e429428ab254d7cabf010eeadf87e63542cde9d6e436d72a926069c8` |
| `training/line_b/freeze_line_b.py` | — | `e4b0a1e9139c179d5f6ac4d1d6e187e736b9f881db4a20a21dcff97b077903f8`（范围修复后）→ `0e02972c97b87208202acf54e7492017636faaa393829ba09864d29a1a7379e4`（预检输出迁出 B3 后，最终 FREEZE 版本） |

`code_provenance.json` 未被修改（其 `current_source_sha256` 不含 Verifier，因此不影响 `code_provenance.current_source_sha` 检查）。

## 6. 不变性声明

- 未迁移、未删除、未改写任何 C1/C2/D1/D2 资产；
- 未修改 `code_provenance.json`、B3-A 正式资产、模型、checkpoint、概率数组、阈值、标签；
- 未访问 Test，未启动 B4，未运行 Bootstrap；
- 本修复轮未生成 `FREEZE.json`；随后按 B0 v0.2 FINAL 指令完成 FREEZE（见第 8 节）。

## 7. 复现命令

```powershell
D:\miniconda3\envs\spectra-db\python.exe -X utf8 -m py_compile training/line_b/verify_line_b_manifests.py training/line_b/freeze_line_b.py
D:\miniconda3\envs\spectra-db\python.exe -X utf8 -c "import sys; sys.path.insert(0,'.'); from training.line_b.verify_line_b_manifests import verify_project; r=verify_project(); print(r['overall_status'], r['summary']); raise SystemExit(r['exit_code'])"
```

## 8. 结论与后续执行

修复后预检：`FAILED = 0`，READY。

随后按用户指令将预检报告输出位置从 `runs/LINE-B-MANIFESTS/B3/` 迁至 `runs/LINE-B-MANIFESTS/`（仅输出位置的最小修改，验证逻辑不变），并执行 FREEZE：

- 预检报告：`runs/LINE-B-MANIFESTS/freeze_preflight_verification_20261120T014410.json`
- FREEZE：`runs/LINE-B-MANIFESTS/FREEZE.json`，SHA256 `65a84959356a9a1f7a6e1d6434449f607c75a4c224a17450f3345ba25bb92a0c`
- 写入后复核 + 完整复验：`runs/LINE-B-MANIFESTS/final_freeze_verification_20261120T014514.json`，`PASS 10 / FAILED 0 / ACCEPTED 4`，212 项资产 0 缺失 0 哈希不符
- B3 目录保持 6 个文件，verifier 兼容摘要仍为 `c099ae978071e14886b6f2e66997ee93eb80f6e4ecbfd91f0383df89ecb96a44`（与 B3.5-A 记录一致，冻结未改动 B3）

```text
B0 v0.2: FINAL
B3 FREEZE: PASS
FREEZE.json: CREATED AND VERIFIED
```
