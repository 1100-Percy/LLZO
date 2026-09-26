# 可选：M3 完成后并行跑 M4 与 M5（两个独立的 Codex 任务）

## 任务 A（分支 m4-chem）
在分支 m4-chem 上只执行 PLAN.md 的 M4。不得修改 src/llzo_pdp/support.py 或 outputs/M5/。
完成后测试全绿、写 checkpoint、提交。不要更新 PLAN.md 中 M5 的状态。

## 任务 B（分支 m5-support）
在分支 m5-support 上只执行 PLAN.md 的 M5。不得修改 src/llzo_pdp/checks.py 或 outputs/M4/。
完成后测试全绿、写 checkpoint、提交。不要更新 PLAN.md 中 M4 的状态。

## 合并（主分支上的第三个任务）
依次合并 m4-chem 与 m5-support，解决 PLAN.md / PROGRESS.md 的冲突（保留两边状态），
运行全部测试，更新 PLAN.md 与 PROGRESS.md，提交后结束会话。M6 由下一个 01_continue 会话执行。
