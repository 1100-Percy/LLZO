# DECISIONS.md

## 实现选择（D-xxx）
格式：D-001 | 里程碑 | 问题 | 选项 | 选择 | 理由

- D-001 | M0 | 当前 checkout 没有 .venv，python 命令不可用；系统 python3 可运行原有测试。选择 `python3 -m venv --system-site-packages --without-pip .venv` 离线复用依赖，Makefile 默认 `.venv/bin/python`。不执行会联网安装的 setup.sh。
- D-002 | M0 | 本研究环境缺少 lightgbm、catboost；不联网、不安装。M0/M1 使用已安装工具继续；M2 的指定模型不能用别的算法冒充，后续需找到已有离线运行时，否则只推进不依赖缺包的任务并明确标注限制。
- D-003 | M0/M1 | 本次启动指令明确要求同一会话完成 M0 与 M1，优先于常驻的每会话单里程碑约定；仍逐里程碑验收与提交。元数据记录生成时 HEAD 与 git_dirty，提交本身收纳生成代码及输出，避免将生成前的 HEAD 误称为完整生成代码版本。
- D-004 | M1 | 本研究选择：过滤仅统计协议选定的原始组成、计数、合成数值及类别列，派生描述符不重复计缺失，one-hot 前缺失类别仍计缺失。严格 >3 删除，不以论文保留数调规则。K12 全部被过滤，另用仅保留集拟合的插补器 transform 作排除记录诊断，不进入训练、阈值估计或有效化学检查。
- D-005 | M1 | 本研究选择：描述符仅依赖 dopant_long 的实际位点分配；未分配元素不按常识补位，V_raw 为零，V_corr 仅执行 K07。缺失 qbar 按已有 charge_residual_features 置零，K12 数值残差不能认作有完整掺杂信息的化学判定；输出 eligible_for_chemistry_checks=false 与 n_unassigned。
- D-006 | M1 | 审计按 record_id × field 展开：correct 的问题条目数不是修改单元格数。验收核对展开的精确键集合、前后值及其对应问题，flag_only 等不修正；audit_log 只保存 V_corr 的实际修改。
- D-007 | M1 | 本研究选择：one-hot 把原始类别整体编码，共掺字符串不拆成 multi-hot；0、异常原值和 missing 分开。使用全数据中可见的类别词表，仅数值插补有 A0/A1 拟合差别，不使用标签构建类别。测量温度按协议 rename 写为 Measurement Temperature；此命名与 synthesis_numeric 中的原始名通过同一映射连接。
- D-008 | M1 | 本研究选择：IterativeImputer 使用 BayesianRidge 默认估计器、max_iter=10、tol=1e-3、mean 初始化、ascending、sample_posterior=false、skip_complete=false；不缩放、不裁剪、不去离群点。记录收敛警告。A0 在全部保留行拟合，A1 在明确训练 ID 上拟合；目标和来源 ID 不进入插补器。M1 为 A1 诊断按主阈值和协议种子先切 test=0.15，再在余下行按 0.15/0.85 切 validation；M2 其他阈值或 CV 必须重新划分并在各自训练 fold 内拟合 A1，不能复用这里的 A1 矩阵。
- D-009 | M1 | T_ratio 在有限样本或标签并列时可能无法精确达到 21/33。遵守协议的比较目的，枚举观测阈值取高导比例误差绝对值最小者，平手取较高阈值（更少高导标签），报告实际比例、误差及是否精确；不改变 T_main，也不读取 paper_targets 的结果调参。
- D-010 | M1 | Git 提交不能在自身受跟踪文件中包含其最终 hash。冻结提交先包含协议 status=frozen 和全部验收产物，随后仅追加一次文档提交记录该冻结提交的完整 hash；不 amend 冻结提交、不再改协议，不开始 M2。

### 协议冻结记录

- M1 冻结提交：`015742b147be641d6e1c3de7a26a8ba5d53b4077`。
- 冻结文件：`protocol_v1.yaml`；SHA-256：`ac99cfc50a87bf818b33923af5510198f09ff437a8c8e1ad5cede337b9fc158f`。
- 此后仅允许在本文件写 P-xxx 提案，继续使用冻结参数。本记录由随后纯文档提交补全。

## 协议修改提案（P-xxx，只有人类可以采纳并生成 protocol_v2.yaml）
格式：P-001 | 里程碑 | 建议修改 | 理由 | 对结果的预期影响

## Out-of-scope ideas（不实现）
