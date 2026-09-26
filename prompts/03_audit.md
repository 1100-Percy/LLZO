# 独立审计（建议 D3 晚与 D7 各跑一次，用一个新的 Codex 会话，不要复用执行会话）

你是审计员，不是执行者。不要修复任何东西，只写 reports/audit_<日期>.md。逐项检查并给出证据
（文件路径、行号、命令输出）：

1. 协议：git log -p protocol_v1.yaml —— M1 冻结之后是否有改动？
2. 测试：tests/test_data_facts.py、tests/test_chemistry.py 的期望值是否被改过（git log -p）？全部测试是否通过？
3. 泄漏：A1 插补是否只在训练行上 fit？GridSearch 是否只用训练集？测试集 record_id 是否出现在任何调参步骤中？
4. PDP 引擎：随机抽 3 个面板，用 sklearn.inspection.partial_dependence(method="brute") 重算，差值 < 1e-10？
5. 联动路径：随机抽 20 条 D 方案虚拟输入，手算 Li/Zr/site_sto 是否满足 chemistry.linked_update 的公式？
6. 背景一致性：M6 每条链中 B/C/D/D′ 的背景 record_id 集合是否相同？C/D′ 中被置空的网格点是否符合 n≥5、来源≥2？
7. 报告：reports/part1_report.md 中是否有不在 claims.csv 里的数字？运行 scripts/verify_claims.py。
8. 措辞：grep "证明|验证了|证实|最优配方" 等超出证据的表述，逐条列出。
9. "本研究选择"：列出代码中所有未在协议或 DECISIONS.md 登记的实现选择。

最后给出 PASS / PASS_WITH_ISSUES / FAIL，并按严重程度排序问题清单。
执行会话下一次启动时会读到这份审计（请在 HUMAN_FEEDBACK.md 追加一行指向它）。
