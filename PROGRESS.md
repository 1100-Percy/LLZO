# PROGRESS.md
- M-1 (human, pre-Codex): scaffold, verified data facts (tests/test_data_facts.py), chemistry utils (tests/test_chemistry.py). 14 tests green.

## 后台任务
（无）

## M0 完成
- 已运行原有测试及新增元数据测试，结果见 outputs/M0/validation.csv 与 reports/checkpoints/M0.md。
- 离线 .venv 复用系统依赖；运行 `make test`，或激活 `.venv` 后运行 `python -m pytest -q`。
- 原始文件及已核实模块、测试的校验值已保存。缺少 lightgbm、catboost，见 D-002。
- 下一个步骤：按本次启动指令继续 M1；不执行 M2。

## M1 完成
- 已新增 features.py、preprocess.py、scripts/run_m1.py 及特征、预处理、产物集成测试。
- `make m1` 可重跑数据产物、完整测试与 checkpoint；测试结果由脚本保存到 outputs/M1/test_validation.csv。
- 数值结果见 outputs/M1/dataset_summary.csv、validation.csv；报告与登记结论已从 CSV 重新读取核对。
- V_corr 仅执行 known_issues.csv 明确 correct 的项目；K12 被行过滤排除，只另行 transform 诊断，未放回训练集。
- 已核对原始数据、io.py、chemistry.py 及原有事实测试的 SHA-256 均与 M0 相同；协议唯一改动为 status=frozen。
- 冻结提交的 hash 按 D-010 在随后的文档提交回填，之后不得编辑 protocol_v1.yaml。

## 下一个会话从哪里开始
- 先读 HUMAN_FEEDBACK.md，再从 PLAN.md 的 M2 开始。当前会话没有训练任何 M2 模型。
- 注意当前运行时缺少 lightgbm、catboost；不联网、不安装、不替换指定模型，按 D-002 处理可用离线运行时或依赖阻塞。
- M1 的 A1 矩阵只适用于已保存的主阈值划分；M2 更换阈值、验证折或分组时须在相应训练行重新拟合插补器，防止泄漏。
- 没有后台任务。

## 本次会话摘要
- 完成 M0 与 M1 的实现、离线运行、验收、checkpoint、claims 登记及协议冻结。
- 所有关键数字由 outputs 中产物生成；优先查看 reports/checkpoints/M1.md、audit_log.csv 和 DECISIONS.md。
- 保留原始文件与已核实模块；没有联网、安装新包或执行 M2。
