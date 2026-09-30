# PROGRESS.md
- M-1 (human, pre-Codex): scaffold, verified data facts (tests/test_data_facts.py), chemistry utils (tests/test_chemistry.py). 14 tests green.

## 后台任务
- 无运行中的后台计算。M3 已在本会话前台完成，`outputs/M3/run_state.json` 为 complete；日志见 `outputs/M3/run.log`。
- 完整命令：`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/llzo_m3_mpl .venv/bin/python -u scripts/run_m3.py > outputs/M3/run.log 2>&1`。
- 本次无需 nohup：耗时探测见 outputs/M3/budget.csv，实际完成时长见 run_state.json。M2 历史后台计算也已完成，勿重启。

## M3 完成
- 新增 pdp.py 与数值引擎测试：G1/G2、单变量 ICE、双变量虚拟输入、联动更新接口及冻结窗口规则。
- 新增 scripts/run_m3.py 与入口回归测试，全部概率由 M2 已保存的主设置模型生成；未重新训练模型。
- 已生成论文指定面板 46 个及模型自身前五面板 20 个，保存 1644048 条虚拟输入、50 条单变量窗口与 76 条对照记录（读取 outputs/M3/validation.csv）。
- 每张图有同名 CSV 和原始训练观测 rug；完整插补背景与紧凑虚拟输入共同支持重建。已从磁盘重读背景，核对曲线聚合、窗口边界和抽样预测。
- `.venv/bin/python scripts/run_m3.py --verify` 可复核产物并生成完整测试、checkpoint 与 claims；`--prepare` 仅更新耗时探测，不覆盖既有生成指纹。
- 完整测试 96 项通过；关键结果及需要人工查看的项目见 reports/checkpoints/M3.md。

## M2 执行步骤
- [x] 新增 src/llzo_pdp/models.py 与 tests/test_models.py：确定性分层划分、固定网格、概率指标、模型重载验证；先运行新增测试确认缺失行为，再实现并通过。
- [x] 新增 scripts/run_m2.py 与 tests/test_m2.py：四阈值及分层/不分层诊断、耗时探测、按任务保存并校验续跑、最终汇总及 CSV 驱动 checkpoint。
- [x] 执行 `.venv/bin/python scripts/run_m2.py --prepare`，完成反馈要求的类别计数诊断并保存预算估计；完整测试通过。
- [x] 已按协议启动完整网格后台运行；命令、PID 与预计时长见上方。此项表示启动完成，全部计算和 M2 验收尚未完成。
- [x] 已收取全部结果，复核每个模型重载、汇总产物与验收 checkpoint，M2 标为 DONE。

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
- 先读 AGENTS.md、HUMAN_FEEDBACK.md 与本文件；M3 已验收，无运行中后台任务。按 PLAN.md 开始 M4，仅执行该里程碑；本次没有启动 M4。
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

## 本次会话摘要（M2 完成验收）
- 收取 M2 后台结果，完成 330 个任务的模型重载、预测与指标一致性验证；完整测试 37 项通过，无失败、错误或跳过（数字读取 outputs/M2/completion_summary.csv 与 test_validation.csv）。
- 新增 scripts/finalize_m2.py 和 outputs/M2/review_summary.csv，从已有 CSV 生成完成 checkpoint 与 claims；更新 PLAN.md 为 DONE。
- 核对全部 M2 产物的配套元数据及 SHA-256，逐项重新读取 M2 claims；原始数据、受保护代码及事实测试、冻结协议均保持不变。
- 新增 D-/P- 编号：无。沿用现有研究选择；报告保留 A0 插补范围、单类别折 ROC-AUC 缺失及不作模型排名的边界。
- 提交定位：`git log --grep="M2: 完成模型验收与结果汇总"`。依 HUMAN_FEEDBACK.md 的持续授权，提交后推送 GitHub main，并在会话结束前核对远端 SHA；实际推送结果见会话最终回复。
- 下一个会话：先处理新反馈，然后从 M3（PDP 引擎与窗口提取）开始；不要重启已完成的 M2 训练。

### 本次提交与推送状态
- M2 完成提交：`6e42f831d265b2d4fa83755ada10de1caf1af0ea`，分支 `codex/m2-completion`；本地验收完成。
- 自动审批拒绝执行向 `https://github.com/1100-Percy/LLZO.git` 的推送及令牌读取命令，理由为可信授权未明确覆盖向该具体外部目的地上传项目数据。命令未执行，远端未核验，本次不绕过审批。
- 历史阻塞已解除：用户在本聊天明确回复“推送”，授权上述目标仓库；已完成非强制推送并核验远端 SHA。
- 下一会话先处理新反馈，再按计划进入 M3；M2 计算不需要重跑。

## 本次会话摘要（M2 推送完成）
- 已按用户明确授权将 M2 完成提交及模型产物推送至 `https://github.com/1100-Percy/LLZO.git` 的 `main`，未使用 force。
- 推送后远端与本地均为 `9cfbb0e29f6813c68654ac87f3a297b380e5da0c`，已通过 `git ls-remote` 核验；本次状态回填随后另作纯文档提交并同步。
- 新增 D-/P- 编号：无。本次仅推送及更新状态文档，未改代码、数据或协议，未重跑测试；沿用 M2 验收的完整测试结果，见 outputs/M2/test_validation.csv。
- 下一个会话从新反馈检查与 M3 开始；本次没有启动 M3。

## 本次会话摘要（M3 完成）
- 完成 M3 引擎、主设置全部面板、虚拟输入保存、窗口提取、论文一致性对照、中文 checkpoint 与 claims 登记；PLAN.md 中 M3 已标为 DONE。
- 新增 D-018、D-019、D-020、D-021；无 P-xxx。网格、背景、分段窗口和论文对照判据均在生成曲线前登记，未改变冻结协议或原始数据。
- 完整测试 96 项通过，无失败、错误或跳过（读取 outputs/M3/test_validation.csv）。新增回归测试确认耗时探测不会覆盖历史产物指纹。
- 已核对产物与配套元数据、当前源码指纹、逐条 M3 claims，以及历史受保护文件哈希；规范审查、代码复审和图像抽查通过。
- 本次只完成 M3；化学残差及支持标签仍留空，定性或含义不明确的论文目标仍为 not_comparable。
- 提交并按持续授权推送至 GitHub main；提交 hash 在下一次纯文档回填记录。下一个会话先处理新反馈，再开始 M4，不重跑 M2/M3。

### M3 提交与远端核验
- 完成提交：`14465a5b0b5cb2435175a2c0a0c90d2f14dc114a`；已推送至 `1100-Percy/LLZO` 的 `main`，`git ls-remote` 返回相同 SHA。
- 本次仅追加提交标识与远端核验记录；代码、产物及冻结协议不变。状态回填提交随后同步推送；无运行中后台任务。
- 下一会话从 M4 开始，并优先检查 HUMAN_FEEDBACK.md 中的新反馈。
