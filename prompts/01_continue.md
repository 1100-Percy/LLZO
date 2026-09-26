# 续跑（每个里程碑开一个新会话，粘贴这段）

继续执行 llzo_part1 的第一部分计划。

1. 先读 AGENTS.md，再读 HUMAN_FEEDBACK.md。每条未标记 [done] 的反馈优先处理：
   - 若它要求修改冻结协议，不要改协议，写 P-xxx 提案到 DECISIONS.md 并在该反馈后注明；
   - 其余反馈直接处理，完成后在该条前加 [done] 并注明提交 hash。
2. 读 PROGRESS.md。如果上一次会话在后台启动了长计算（PROGRESS.md 中有"后台任务"记录），
   先检查它的日志：已完成 → 汇总结果并完成该里程碑的剩余步骤；失败 → 排查并重新启动；
   仍在运行 → 在 PROGRESS.md 记录当前进度后结束会话，不做别的事。
3. 读 PLAN.md 状态表，本次会话只执行"一个"里程碑（第一个 TODO / IN_PROGRESS 的）：
   完成 → 测试全绿 → 写 reports/checkpoints/Mx.md → 更新 PLAN.md 与 PROGRESS.md → 提交 → 结束会话。
   不要开始下一个里程碑。
   若该里程碑预计超过 3 小时，先把它拆成子步骤写进 PROGRESS.md，本次只做到一个可提交的子步骤为止。
4. 预计运行超过 30 分钟的计算：用 nohup 在后台启动，日志写到 outputs/Mx/run.log，
   把启动命令、PID、预计时长写进 PROGRESS.md 的"后台任务"一节，提交后结束会话。
5. 会话结束前在 PROGRESS.md 末尾追加"本次会话摘要"：完成了什么、新增 D-/P- 编号、
   测试数量、下一个会话应该从哪里开始。
