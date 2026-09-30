# PROGRESS.md
- M-1 (human, pre-Codex): scaffold, verified data facts (tests/test_data_facts.py), chemistry utils (tests/test_chemistry.py). 14 tests green.

## 后台任务
- M2 完整计算正在运行；PID：`43699`。工作目录：`/Users/hanfengdexuexiji/.codex/worktrees/bba1/llzo_part1`。
- 启动：通过 Python Popen 的 start_new_session=True 执行 nohup；等价命令：`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nohup .venv/bin/python -u scripts/run_m2.py >> outputs/M2/run.log 2>&1 < /dev/null &`。
- 日志：`outputs/M2/run.log`；状态：`outputs/M2/run_state.json`；逐任务完成标记：`outputs/M2/jobs/*/complete.json`。启动快照见 `outputs/M2/launch.json`。
- 本次结束前进程核验：PID 存在且运行，状态快照显示已完成 25/330 个任务，当前为 full_RF_V_raw_A0_T_median_kept_seed1；以实时 run_state.json 为准。
- 预计约 38–68 分钟（来自 outputs/M2/budget.csv 的短计时估计）；协议总预算仍有效。运行中的日志和状态文件会继续变化，log.meta.json 在结束时刷新为最终校验值。
- 下一会话首先核对该 PID、状态和日志：仍运行则只更新进度并结束；失败则按 traceback 排查，保持配置不变后用同一命令续跑；完成则执行 `.venv/bin/python scripts/run_m2.py --verify`，更新 M2 完成状态并提交推送。不要启动 M3。

## M2 执行步骤
- [x] 新增 src/llzo_pdp/models.py 与 tests/test_models.py：确定性分层划分、固定网格、概率指标、模型重载验证；先运行新增测试确认缺失行为，再实现并通过。
- [x] 新增 scripts/run_m2.py 与 tests/test_m2.py：四阈值及分层/不分层诊断、耗时探测、按任务保存并校验续跑、最终汇总及 CSV 驱动 checkpoint。
- [x] 执行 `.venv/bin/python scripts/run_m2.py --prepare`，完成反馈要求的类别计数诊断并保存预算估计；完整测试通过。
- [x] 已按协议启动完整网格后台运行；命令、PID 与预计时长见上方。此项表示启动完成，全部计算和 M2 验收尚未完成。
- [ ] 下一会话收取结果：先查看后台日志；完成则验证每个模型重载、汇总所有产物与验收 checkpoint，再将 M2 标为 DONE。运行中则只记录进度后结束。

本研究选择与边界见本次新增 DECISIONS.md；协议保持冻结。整体预计不足三小时，不减少网格或变体。

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
- 冻结提交为 `015742b147be641d6e1c3de7a26a8ba5d53b4077`；hash 已按 D-010 回填至 DECISIONS.md，之后不得编辑 protocol_v1.yaml。

## 下一个会话从哪里开始
- 先读 HUMAN_FEEDBACK.md，再检查上方 M2 后台任务；严格按运行中、失败、完成三种状态处理。
- 模型依赖已按人类反馈授权补齐，包含 LightGBM 所需 libomp；当前验证结果见 outputs/feedback_dependencies/。D-002 为历史限制，新的授权与处理见 D-011。
- M1 的 A1 矩阵只适用于已保存的主阈值划分；M2 更换阈值、验证折或分组时须在相应训练行重新拟合插补器，防止泄漏。
- 当前后台任务以本文件开头的“后台任务”为准；下方旧会话摘要为历史记录。

## 启动会话摘要（M0/M1）
- 完成 M0 与 M1 的实现、离线运行、验收、checkpoint、claims 登记及协议冻结。
- 所有关键数字由 outputs 中产物生成；优先查看 reports/checkpoints/M1.md、audit_log.csv 和 DECISIONS.md。
- 保留原始文件与已核实模块；没有联网、安装新包或执行 M2。

## 本次会话摘要（人类反馈）
- 安装 lightgbm、catboost 及必要运行依赖，合成数据训练与完整测试通过；新增 scripts/check_model_dependencies.py 可重跑验收。
- 更新反馈处理记录、决策、环境产物、checkpoint 与 claims；保留 M0/M1 的历史检查记录，不改原始数据或冻结协议。
- 处理提交 `767cff3d9657c13cdd8ae10e7aa2fb65c1f34514` 已连同此前 M0/M1 提交推送至 GitHub main，远端 SHA 核验一致；HUMAN_FEEDBACK.md 已标记 [done]。此后的状态回填亦提交并推送。
- M2 保持 TODO；本会话到反馈处理完成为止。

## 本次会话摘要（M2 启动）
- 完成模型训练与持久化模块、可续跑 M2 入口、完整任务清单、阈值反馈诊断和阶段 checkpoint；已启动完整后台计算，M2 保持 IN_PROGRESS。
- 新增 D-013、D-014、D-015、D-016、D-017；没有新增 P-xxx，没有修改冻结协议或原始数据。
- 完整测试 37 项通过（outputs/M2/test_validation.csv）；依赖产生弃用警告，测试无失败或跳过。
- 阶段提交 `ae6a5b6460bf24f2daf8b74dd45525cbd41c2d9c` 已推送 GitHub main 并核对远端 SHA；新增反馈均已标记 [done]，随后将状态回填提交一并推送。
- 下一个会话先检查 PID 43699 与 outputs/M2/run.log，不做其他里程碑。后台产生的模型、日志和状态尚未收取提交；这不是 M2 完整验收。
