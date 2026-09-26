# AGENTS.md — LLZO PDP 解释可靠性研究 · 第一部分

你是这个仓库的自动执行代理。人类研究者每天只花约 30 分钟看结果，其余时间你独立推进。
本文件是常驻规则；具体任务在 `PLAN.md`，冻结参数在 `protocol_v1.yaml`。

## 1. 研究问题（只做这一件事）

原论文（Abraham et al. 2026, `data/raw/paper_*.pdf`）用树模型 + PDP 给出 LLZO 的"有利成分/工艺区间"。
第一部分只回答：

> PDP 在生成曲线时评价了哪些"虚拟材料"？其中有多少违反声明的名义化学计量关系、或远离真实数据？
> 这些输入是否实质改变了论文给出的有利窗口？

你**不**做：新模型的性能竞赛、新特征工程、ALE/SHAP 方法比较（属于第二部分）、按来源留出的设计建议检验（属于第三部分）。
如果你发现某个方向很有意思但超出范围，写进 `DECISIONS.md` 的 "Out-of-scope ideas"，不要实现。

## 2. 不可违反的规则

1. **原始数据只读。** `data/raw/` 下任何文件不得修改。所有修正通过 `config/known_issues.csv` 生成的 V_corr 版本实现。
2. **协议冻结。** M1 提交后，`protocol_v1.yaml` 不得修改。需要改动时，在 `DECISIONS.md` 写提案 `P-xxx`（理由、影响、建议值），然后**继续用 v1 值运行**。
3. **测试不改期望值。** `tests/test_data_facts.py` 与 `tests/test_chemistry.py` 中的期望值是独立核实过的事实。测试失败时修代码；如果你确信事实本身有误，写证据到 `DECISIONS.md`，把该里程碑标为 BLOCKED，转去做下一个不依赖它的里程碑。
4. **不对着原文调参。** `paper_targets.yaml` 只用于生成"一致性对照表"。禁止为了贴近论文结果而改变阈值、网格、模型设置或窗口规则；协议中列出的变体全部跑完并全部报告。
5. **先定规则，再看曲线。** 窗口提取规则、支持判定阈值、化学容差都在协议里。不得在看到结果后换规则。
6. **报告中没有手写数字。** 所有数字由脚本从 `outputs/` 中的 CSV/parquet 生成。每条结论在 `reports/claims.csv` 中登记：`claim_id, statement, artifact_path, key, value`。
7. **原文未说明的实现细节一律标注"本研究选择"**（例如 PDP 网格、背景集、电负性标度、共掺描述符的聚合方式）。不得写成"复现了原文"。
8. **区分三种陈述：** 化学不一致（违反声明假设） ≠ 材料不可能存在；数据支持不足 ≠ 预测错误；解释改变 ≠ 预测改变。报告措辞必须保持这些区分。
9. **可复现。** 固定随机种子（协议 `seeds`）；每个输出文件旁写 `*.meta.json`（git hash、协议版本、种子、数据版本、生成脚本、时间）。
10. **不联网，不装新包。** 依赖已在 `requirements.txt` 里安装好。确实缺包时写进 `DECISIONS.md` 并用已有工具替代。

## 3. 代码结构约定

```
src/llzo_pdp/
  io.py            # 已存在，已测试：load_raw()
  chemistry.py     # 已存在，已测试：残差、位点分配、linked_update()
  features.py      # M1：构建模型特征（含描述符、one-hot、V_raw/V_corr）
  preprocess.py    # M1：行过滤、MICE（A0/A1）
  models.py        # M2：训练、调参、评估
  pdp.py           # M3：brute ICE 引擎（保存每条虚拟输入）、窗口提取
  checks.py        # M4：虚拟输入化学检查、描述符查询检查
  support.py       # M5：联合数据支持
  ablation.py      # M6：A/B/C/D/D' 方案
  sensitivity.py   # M7
scripts/           # 每个里程碑一个入口：scripts/run_m2.py ...，可 `python scripts/run_mX.py` 独立重跑
outputs/Mx/        # 该里程碑的全部产物（csv/parquet/png/meta.json）
reports/           # 自动生成的报告、checkpoints、claims.csv
tests/             # 每个新模块都要有测试
```

- 特征名使用协议里的名字（`Li_comp`, `Zr_site_sto` …），原始列名只在 `features.py` 里映射一次。
- 所有 PDP 输出使用"长表"：`model, data_version, scheme, feature, background, bg_record_id, bg_source_id, grid_value, p_high, r_charge, r_site_Zr, r_site_La, chem_label, support_label, …`。
- 图只是 CSV 的可视化；每张 png 必须有同名 csv。

## 4. 工作方式

- **一个会话只做一个里程碑**（或一个可提交的子步骤）。会话开始：读本文件 → `HUMAN_FEEDBACK.md` → `PROGRESS.md` → `PLAN.md` 状态表，找到第一个未完成的里程碑执行。
- 每个里程碑结束时：
  1. `python -m pytest -q` 全绿；
  2. 满足该里程碑的"验收标准"；
  3. 写 `reports/checkpoints/Mx.md`：做了什么、关键数字（从 CSV 读取）、人类需要看的 3–5 项、未解决问题；
  4. 更新 `PLAN.md` 状态表与 `PROGRESS.md`（含"下一个会话从哪里开始"）；
  5. `git commit -m "Mx: <一句话>"`，然后**结束会话**，不开始下一个里程碑。
- 人类不会在会话中途插话。人类的反馈写在 `HUMAN_FEEDBACK.md`，下一个会话开始时先处理。
- 超过 30 分钟的计算用 nohup 后台运行，记录到 `PROGRESS.md` 的"后台任务"，然后结束会话；由下一个会话收取结果。
- 遇到歧义：选协议内最保守的做法，写一条 `D-xxx` 到 `DECISIONS.md`（问题、选项、选择、理由），继续执行。
- 长时间运行（bootstrap 等）：先用 B=3 跑通全流程并提交，再在后台跑完整 B=50。
- 用中文写报告与 checkpoint；代码、列名、注释用英文。
