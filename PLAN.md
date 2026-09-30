# PLAN.md — 第一部分执行计划（7 天，Codex 全自动）

## 状态表（代理维护）

| 里程碑 | 内容 | 计划日 | 状态 | 提交 |
|---|---|---|---|---|
| M0 | 环境、骨架、已有测试 | D1 上午 | DONE | ebca7d3 |
| M1 | 数据版本、特征构建、协议冻结 | D1 | DONE | 015742b |
| M2 | 基准模型复现（全数据 4 模型 + Ga 子集 2 模型） | D2 | DONE | 6e42f83（已推送 GitHub main） |
| M3 | PDP 引擎 + 原文图复现 + 窗口提取 | D3 | DONE | 14465a5（已推送 GitHub main） |
| M4 | 虚拟输入化学检查 | D4 | TODO | |
| M5 | 联合数据支持 | D4 | TODO | |
| M6 | 受控消融 A→B→C→D→D′ | D5 | TODO | |
| M7 | 敏感性与稳定性（长任务） | D6 | TODO | |
| M8 | 报告生成、claims 核对、自审 | D7 | TODO | |

状态取值：TODO / IN_PROGRESS / DONE / BLOCKED(原因)。M4 与 M5 互不依赖，可在两个分支并行。

---

## M0 环境与骨架（约 1 小时）

**任务**
1. 确认 `.venv` 可用，`python -m pytest -q` 全绿（14 个测试）。
2. 建立 `scripts/`、`outputs/`、`reports/checkpoints/`、`DECISIONS.md`、`PROGRESS.md`、`HUMAN_FEEDBACK.md`（空）、`reports/claims.csv`（仅表头）。
3. 写 `src/llzo_pdp/meta.py`：`write_meta(path, **kw)`，自动记录 git hash、协议版本、时间。
4. 写 `Makefile`：`make test`，`make m1` … `make m8`（调用 `scripts/run_mX.py`）。

**验收**：测试全绿；`make test` 可运行；提交。

---

## M1 数据版本、特征构建、协议冻结（D1）

**任务**
1. `features.py`
   - `build_features(df, version)`：version ∈ {V_raw, V_corr}；V_corr 按 `config/known_issues.csv` 中 `action_V_corr == correct` 修正，并记录每一处修改到 `outputs/M1/audit_log.csv`（record_id, field, old, new, issue_id）。
   - 组成特征：`Li_comp` 等，来自原始列（改名在此完成）。
   - 描述符：按 `chemistry.dopant_long()` 的位点分配，从 `config/dopant_properties.csv` 取值（按 element + site 匹配），同一位点多个掺杂元素时按 x 加权平均；未掺杂位点 = 0。输出 `outputs/M1/descriptor_table.csv`（record_id, site, elements, x, elnv, irad, A）。
   - 位点未分配的掺杂（record 8、11、188–193，`dopant_long` 中 site = unassigned）：V_raw 按原始位点列（为空）得到描述符 0；V_corr 仅对 K07（record 11，Al → Li 位）修正。写 D-xxx 记录。
   - one-hot：协议 `one_hot` 列，缺失为 `missing` 类别；列名形如 `Li_dop_Al`。
   - 同时返回 `site_qbar`（chemistry.site_charge_table）供后续残差计算。
2. `preprocess.py`
   - 行过滤（> 3 个缺失）→ 输出保留行数、被删行 record_id。
   - A0 / A1 两种 MICE（IterativeImputer）。
   - **插补化学检查**：对 K12 记录，插补后的成分用 `charge_residual_features` 计算残差，输出 `outputs/M1/imputed_chemistry.csv`。
3. 测试 `tests/test_features.py`：
   - Ga 单掺记录 `Li_site_elnv == 1.81`、`Zr_site_elnv == 0`；
   - record 30（Gd0.2）`Zr_site_elnv == 1.20`；
   - V_corr 中 record 178 的 `Zr_comp == 1.6`、`La_comp == 3`，V_raw 中仍为 0.4 / 0；
   - one-hot 中存在 `Li_dop_Al`。
4. 写 `outputs/M1/dataset_summary.csv`：保留行数、来源数、各阈值下高导比例（T_main、T_median_kept、T_1e4、T_ratio 的数值也在这里计算出来并写入）。
5. **冻结协议**：把 `protocol_v1.yaml` 的 `status` 改为 `frozen`（这是唯一一次允许的编辑），提交，并在 `DECISIONS.md` 记录提交 hash。

**验收**：新测试通过；audit_log 行数与 known_issues 中 correct 项一致；checkpoint 列出保留行数（期望约 220，论文测试集 33 条 ≈ 15%），若偏差 > 10 条写 D-xxx 说明原因。

---

## M2 基准模型复现（D2）

**任务**
1. 全数据：DT、RF、LGBM、CB，V_raw，A0，T_main，70/15/15 分层，seeds 0–4，GridSearchCV（协议网格）。
   - 输出每个 seed 的 test/val 指标（accuracy、weighted F1、ROC-AUC）与混淆矩阵 → `outputs/M2/metrics.csv`。
2. **阈值复现检查**：对四个阈值变体各跑 seed 0–4，记录测试集真实类别计数；标出哪一种得到 (12, 21) 或最接近 → `outputs/M2/threshold_check.csv`。只报告，不据此改变主阈值。
3. 分组验证：同一流水线做 GroupKFold(5) 与 LOSO（按 source_id），与分层随机 CV 并列报告 → `outputs/M2/group_cv.csv`。
4. 特征重要性：内置重要性前 15，与 `paper_targets.top5_features_fig7` 对照 → `outputs/M2/importance_vs_paper.csv`。
5. Ga 子集：CB、AB，同一流程 → `outputs/M2/ga_metrics.csv`。
6. 另训练 V_corr 版本（seed 0，四模型）供 M6 的 Zr_comp 实验使用。
7. 模型持久化：`outputs/M2/models/{model}_{version}_{imp}_{thr}_seed{s}.pkl`，并保存训练/验证/测试 record_id 列表。

**验收**：所有模型可加载并复现保存的指标；checkpoint 中说明"33 条测试样本，1 条 = 3.03 个百分点"，不对模型排名下结论。

---

## M3 PDP 引擎 + 原文图复现（D3）

**任务**
1. `pdp.py`
   - `ice(model, X_bg, feature, grid, update_fn=None) -> long DataFrame`，保存每条虚拟输入（完整特征向量可选存 parquet，至少保存被修改列与 bg_record_id）。
   - `update_fn` 接口：给定背景行与网格值，返回修改后的行（M6 的联动路径用）。
   - `pdp_2d(model, X_bg, f1, f2, grid1, grid2)`。
   - 网格 G1/G2（协议）。
   - `extract_window(pd_curve, rule)`：严格按协议 `window_rule`。
   - 测试：`ice(...)` 的均值与 `sklearn.inspection.partial_dependence(method="brute")` 在 RF 与 DT 上相差 < 1e-10；`extract_window` 对手工构造曲线（平坦、单峰、双峰）的输出正确。
2. 复现（主设置：seed 0、V_raw、A0、T_main、B_train、G1）：
   - 图 7 的 20 个面板（模型-特征对见 `paper_targets.yaml`，按你模型自己的前五重要特征另做一份）。
   - 图 9 的 16 个双变量面板（Li_comp × y）。
   - 图 12 的 10 个面板（Ga 子集 CB、AB）。
3. 窗口提取 → `outputs/M3/windows.csv`；一致性对照 → `outputs/M3/agreement_vs_paper.csv`（agree / partial / disagree / not_comparable，附判据）。
4. 每个面板的 png 同时标出训练数据的 rug（真实观测值位置）。

**验收**：引擎测试通过；46 个面板全部有 csv+png；对照表完整。

---

## M4 虚拟输入化学检查（D4，可与 M5 并行）

**对象**：M3 的全部单变量 ICE（B_train）+ 以下重点：
- Ga 子集模型在 B_Ga_all 上的 `Li_comp` 与 `Li_site_sto`（图 12a、12e）
- 全数据模型在 B_Ga_single、B_Ta_single 上的 `Li_comp`、`Li_site_sto`、`Zr_comp`、`Zr_site_sto`
- 双变量图 9c、9d、9k、9o

**任务**
1. 对每条虚拟输入计算 `r_charge`（feature-level，qbar 来自背景记录）、`r_site_Zr`、`r_site_La`，以及 Δr（相对背景记录自身残差）；按协议打标签 consistent / background_flagged / violation。
2. 对每个网格点汇总：三类比例、|r| 的中位数与 90 分位、以及 |r| 与 PD 的对应关系（PD 变化集中在 |r| 大还是小的网格段）。
3. **描述符查询检查**（`Zr_site_elnv`、`Zr_site_irad`、`Zr_site_A`、`La_site_A`、`Li_site_elnv`）：
   - 每个网格值 → 该位点数据中出现过的最近元素及距离；超出容差标 `no_element`。
   - 背景 `site_sto == 0` 而描述符 ≠ 0 → `dopantless_query`。
   - 输出电负性分组表：图 7 的 `Zr_site_elnv > 1.22` 阈值两侧各有哪些元素、多少记录、多少来源。
4. 列间一致性：若同一物理量出现在多列（如 `Li_site_sto` 与 one-hot `Li_dop_*`、`# of Dopants`），单列修改造成的矛盾计数。
5. 图：每个面板在 PDP 下方加三类比例的条带图。

**输出**：`outputs/M4/chem_labels.parquet`、`chem_summary_by_gridpoint.csv`、`descriptor_queries.csv`、`elnv_split_table.csv`。

**验收**：`tests/test_checks.py` 验证：Ga 单掺背景上改 `Li_comp` 时，只有等于背景自身 Li 的网格点为 consistent；改 `Li_site_sto` 时 r = 3·Δg。

---

## M5 联合数据支持（D4，可与 M4 并行）

**任务**
1. `support.py`：按协议子空间、稳健缩放、k=3、阈值 = 训练集 LOO 1-NN 距离的 95 分位；输出 d_k、半径内邻居数、邻居来源数、支持标签。
2. 对 M4 同样的对象打支持标签；按网格点汇总。
3. 透明列表（直接交叉表，不依赖距离）：
   - Ga 单掺：g × 烧结温度的记录数与来源数；
   - Ta 单掺：t × 烧结温度；
   - Li_comp 分箱（0.1）× Zr_site_elnv 所属元素；
4. **三个重点区域**，逐一列出最近的 5 条真实记录（record_id、来源、σ、标签）：
   - 图 9c/9d 中 Li_comp ≈ 7.2 的高 PD 区域；
   - Zr_comp < 1；
   - g ≥ 0.35 且烧结温度 ≠ 1100 °C。

**输出**：`outputs/M5/support_labels.parquet`、`support_summary_by_gridpoint.csv`、`crosstabs/*.csv`、`focus_regions.csv`。

**验收**：`tests/test_support.py`：训练集自身每条记录都应是 supported 或 single_source；构造的远点（如 Li_comp=8、烧结 1300 °C、g=0.4）应为 unsupported。

---

## M6 受控消融（D5）

同一模型、同一背景，一次只改一个因素。每个实验输出一张窗口比较表（协议 window_rule）+ 每个网格点的 n 与来源数。

| 实验 | 模型 | 背景 | 链条 |
|---|---|---|---|
| E1 Ga 旗舰 | Ga 子集 CB、AB | B_Ga_all | 12a(Li_comp) 与 12e(Li_site_sto) 的 B、C；D = 沿 Li 位总量联动（linked_update, site=Li）；D′ |
| E2 全数据 Li | DT、RF、LGBM、CB | A: B_train；B/C/D/D′: B_single | D = 通过背景自身掺杂量改变 Li：x_new = x_bg + (v − Li_bg)/li_coefficient；x_new < 0 或超出该元素观测最大值 → 标记 out_of_path，不计算 |
| E3 Ga 背景 × 全数据模型 | 四模型 | B_Ga_single | Li_comp / Li_site_sto：A→B→C→D→D′ |
| E4 Ta 背景 | 四模型 | B_Ta_single | Zr_site_sto / Zr_comp：A→B→C→D→D′（D: site=Zr） |
| E5 录入错误 | 四模型 × {V_raw, V_corr} | B_train | Zr_comp、Zr_site_sto 单变量 + 图 9k、9o |
| E6 双变量 | DT、RF | B_train | 图 9c、9d：每格化学标签比例、支持标签；"Li≈7.2 且 elnv>1"区域内的高导记录数 |

要点：
- E1 必须同时画三条曲线在同一坐标系（以 Li 位掺杂量为横轴，副轴标 Ga 单掺对应的 Li_comp）。
- 模型与背景不一致的比较（例如 Ga 子集模型 vs 全数据模型）只能并列展示，不能作为链条中的一步。
- 所有派生特征在 D 中重新计算：描述符因元素与比例不变而不变，但必须由代码重算并断言不变。

**输出**：`outputs/M6/E*/windows.csv`、`curves.csv`、`png`；汇总 `outputs/M6/ablation_summary.csv`（experiment, model, feature, step_from, step_to, Δlower, Δupper, Δpeak, Δamplitude, n_segments_change, attributed_factor）。

**验收**：`tests/test_ablation.py`：D 路径上所有虚拟输入的 |r_charge| < 1e-9（对 consistent 背景）；B 与 D 的背景 record_id 集合相同。

---

## M7 敏感性与稳定性（D6，长任务，先 B=3 跑通）

对 M6 的核心结果（E1、E2、E4、E5 中的主窗口），逐项单独改变：
1. 按来源 bootstrap（B=50，超参固定）→ 窗口上下界与峰位的 5–95% 区间；
2. 背景加权：记录等权 vs 来源等权；
3. 去掉来源 54（a: 只从背景去掉；b: 从训练与背景都去掉）；
4. 阈值变体 ×4；
5. 网格 G1 vs G2；
6. 划分种子 0–4；
7. 插补 A0 vs A1；
8. 重复标签组按中位数合并（K13）。

**输出**：`outputs/M7/sensitivity.csv`（target, factor, variant, lower, upper, peak, amplitude, n_segments），`outputs/M7/bootstrap_windows.csv`。

**验收**：每个 target × factor 都有结果；报告中每项因素单独列出，不合成为一个总的不确定性。

---

## M8 报告与自审（D7）

1. `scripts/make_report.py` 生成 `reports/part1_report.md`（中文），结构：
   1. 数据与实现：原文给定 vs 本研究选择（表）
   2. 基准复现：性能、阈值检查、分组 CV 差距、重要性对照
   3. 虚拟输入诊断：化学标签、描述符查询、支持
   4. 窗口是否改变：消融汇总表（每行：体系、特征、原窗口 → B/C/D/D′，主要影响因素）
   5. 稳定性：各因素单独列出
   6. 结论边界：哪些是"化学不一致"、哪些是"支持不足"、哪些结论只在特定模型/背景下成立
   7. 给第二部分的输入：哪些问题值得联动路径解决、哪些更像背景/数据问题
2. `reports/claims.csv` 中每一行都能被 `scripts/verify_claims.py` 从 artifact 重新读出并比对（容差 1e-9）；该脚本必须通过。
3. 自审清单 `reports/self_audit.md`：泄漏检查（A1 是否真的只用训练集）、协议是否被修改、是否存在手写数字、是否有未登记的"本研究选择"。

**验收**：`verify_claims.py` 通过；测试全绿；报告中不出现"证明""验证了"等超出证据的措辞（用 grep 检查并记录）。
