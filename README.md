# LLZO 机器学习设计解释的化学一致性 · 第一部分执行方案（1 周，Codex 全自动）

## 0. 这一周要回答的问题

> 原文用 PDP 给出的"有利窗口"，在计算过程中评价了哪些虚拟材料？
> 其中有多少违反声明的名义化学计量关系、或远离真实数据？这些输入是否改变了窗口？

这一周只做诊断，不提出新方法。输出是一张"窗口 → 变化 → 原因"的汇总表，作为第二部分的输入。

## 1. 启动前已经完成、并用测试锁定的内容

| 内容 | 位置 | 说明 |
|---|---|---|
| 数据加载器 | `src/llzo_pdp/io.py` | `record_id` 使用 Excel 行号；`source_id` 由首列向下填充 |
| 化学工具 | `src/llzo_pdp/chemistry.py` | 位点分配、电荷残差、位点和残差、联动更新 `linked_update` |
| 11 条数据事实 | `tests/test_data_facts.py` | 例如：227 条记录 / 53 个来源；204/218 条满足电荷平衡；Ga 单掺 55 条全部满足 Li = 7−3g；Li ≈ 7.2 只有 1 条且为低导；Zr_comp < 1 的 5 条中有 4 条是录入错误 |
| 3 条化学单元测试 | `tests/test_chemistry.py` | 在所有一致记录上，联动更新都保持电荷平衡 |
| 13 类已知问题 | `config/known_issues.csv` | 每类都区分 V_raw（照原文，不改）与 V_corr（修正）两种处理 |
| 掺杂元素性质 | `config/dopant_properties.csv` | Pauling 电负性、原子量、Shannon 半径（注明配位数） |
| 冻结协议 | `protocol_v1.yaml` | 阈值、特征、网格、窗口规则、支持判定、消融定义 |
| 原文目标值 | `paper_targets.yaml` | 只用于生成一致性对照表，不用于调参 |

另外还有一类数据问题：有 4 条记录的位点总量列与掺杂列互相矛盾（record 11、29、37、71）。这是在编写加载器时新发现的，已写入测试和 `known_issues`。

## 2. 协作方式：一个里程碑 = 一个会话

你和 Codex 之间只通过文件沟通，不在会话中途对话。

- **你 → Codex**：`HUMAN_FEEDBACK.md`（意见）、`protocol_v2.yaml`（仅当你采纳某个 P-xxx 提案时）
- **Codex → 你**：`reports/checkpoints/Mx.md`（每个里程碑的小结）、`DECISIONS.md`（它做的选择与提案）、`PROGRESS.md`（进度与下一步）

每个会话的固定循环：

```
你：开新会话 → 粘贴 prompts/01_continue.md → 离开
Codex：读规则 → 处理你的反馈 → 收取后台任务结果（如有）→ 做一个里程碑 → 测试 → checkpoint → 提交 → 结束
你：会话结束后，花 5–15 分钟读 checkpoint；有意见写进 HUMAN_FEEDBACK.md
→ 开下一个新会话，循环
```

## 3. 七天安排（约 12–15 个会话）

| 日 | 会话 | 你在会话之间做什么 |
|---|---|---|
| D0 晚 | 运行 `./setup.sh`，然后开会话，粘贴 `00_bootstrap.md`（执行 M0 + M1） | — |
| D1 | 读 M1 checkpoint 与 `audit_log.csv`；开会话跑 M2 | 保留行数、四个阈值的数值是否合理 |
| D2 | M2 收尾（如被拆分）；开会话跑 M3 | 看 `threshold_check.csv` 与 `group_cv.csv` |
| D3 | M3 收尾；**人工比对图像**；晚上开**新会话**跑 `03_audit.md` | 复现图与论文图 7、9、12 并排看，形状差别写进反馈 |
| D4 | M4 一个会话、M5 一个会话（或用 `02` 并行） | 看 `elnv_split_table.csv` 和 `focus_regions.csv` |
| D5 | M6 可能需要 2–3 个会话（E1–E6 按子步骤提交） | 重点看 E1（Ga 旗舰）和 E5（录入错误） |
| D6 | M7：会话 1 用 B=3 跑通并在后台启动 B=50 → 结束；会话 2（第二天或几小时后）收取结果 | 看 bootstrap 区间是否足够窄，能否支撑 M6 的结论 |
| D7 | M8 一个会话；再开**新会话**跑 `03_audit.md` | 通读报告，重点看"结论边界"一节 |

## 4. 什么时候需要你介入

| 你看到的情况 | 你要做的 |
|---|---|
| 状态表出现 BLOCKED | 读原因，判断后写进反馈 |
| DECISIONS.md 出现 P-xxx | 采纳：复制出 `protocol_v2.yaml`，并在反馈中写明需要重跑的里程碑；不采纳：在反馈中写"P-xxx 不采纳" |
| checkpoint 数字离谱 | 写反馈让它排查，不要自己改代码 |
| 审计给出 FAIL | 把审计文件路径写进反馈 |
| `git log -p tests/` 显示期望值被改 | `git revert`，并在反馈中强调规则 |
| 某个里程碑做坏了 | `git revert` 到上一个里程碑的提交，在 PLAN.md 把它改回 TODO，开新会话重做 |

## 5. 风险与应对

| 风险 | 应对 |
|---|---|
| 沙箱中无法联网安装依赖 | D0 先运行 `setup.sh`。Codex Cloud 用户把 `setup.sh` 内容填进环境的 setup script |
| 复现结果与原文差别大 | 这是预期内的（原文没有公开代码）。对照表如实记录，不调参；差别本身也是结果 |
| 会话上下文用完 | 代理每个里程碑都会提交并写 `PROGRESS.md`，下次用 `01_continue.md` 接着跑即可 |
| 代理为了通过测试修改期望值 | `AGENTS.md` 明确禁止；D3 和 D7 的审计会检查 `git log -p tests/` |
| M7 运行时间过长 | 协议规定超时后只把双变量 PDP 的网格降到 50，其他设置不变，并记录在案 |

## 6. 目录

```
AGENTS.md            Codex 常驻规则（Codex 会自动读取）
PLAN.md              里程碑、产出、验收标准（Codex 维护状态表）
protocol_v1.yaml     冻结参数
paper_targets.yaml   原文数值（仅作对照）
config/              已知问题、元素性质
src/llzo_pdp/        io.py、chemistry.py（已测试）+ 代理新增模块
tests/               已锁定的事实测试 + 代理新增测试
prompts/             00 启动 / 01 续跑 / 02 并行 / 03 审计
data/raw/            原始 xlsx 与论文 PDF（只读）
```
