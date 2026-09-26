# 第一次启动（只用一次，D0 晚上 setup.sh 跑完之后，粘贴给 Codex）

你在仓库 llzo_part1 中工作。请按顺序：

1. 完整阅读 AGENTS.md、PLAN.md、protocol_v1.yaml、paper_targets.yaml、config/known_issues.csv。
   论文 PDF 在 data/raw/，需要时可以读取正文；图中数值以 paper_targets.yaml 为准。
2. 运行 `python -m pytest -q`，确认 14 个测试通过。src/llzo_pdp/io.py 与 chemistry.py 已经过独立核实，
   不要重写它们；需要扩展时新增函数，并为新增部分写测试。
3. 只执行 M0 和 M1。M1 结束时按 PLAN.md 冻结协议。
4. 每个里程碑结束：测试全绿 → 写 reports/checkpoints/Mx.md → 更新 PLAN.md 状态表与 PROGRESS.md → git commit。
5. M1 提交后结束本次会话，不要开始 M2。在 PROGRESS.md 末尾写"本次会话摘要"。

遇到歧义时选协议内最保守的做法，写 D-xxx 到 DECISIONS.md，然后继续，不要停下来提问。
