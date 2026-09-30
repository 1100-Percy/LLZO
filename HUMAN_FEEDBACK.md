# HUMAN_FEEDBACK.md
每条反馈一行，格式：`- YYYY-MM-DD Mx: <内容>`。代理处理后会在前面加 [done] 与提交 hash。

- [done] 767cff3d9657c13cdd8ae10e7aa2fb65c1f34514 | 2026-09-30 M1 后续：如果你需要lightgbm 和catboost 你可以进行安装，并且完成后推送到github上面
  - 指定依赖及必要运行库已安装，合成数据检查与完整测试通过；上述提交已推送至 GitHub main 并核对远端 SHA 一致。详情见 reports/checkpoints/feedback_dependencies.md；未开始 M2。

- [done] ae6a5b6460bf24f2daf8b74dd45525cbd41c2d9c | 2026-09-30 M2: threshold_check 中请额外报告：在 T_main 下 5 个 seed 的测试集类别计数分布，以及按"分层"和"不分层"两种划分分别能否得到 (12, 21)。不分层只作诊断，不作为主设置。
  - 已生成 outputs/M2/threshold_check.csv、main_seed_class_distribution.csv 与阶段 checkpoint；只报告协议种子的观察结果，不改变主实验。
- [done] ae6a5b6460bf24f2daf8b74dd45525cbd41c2d9c | 2026-09-30 M2: 每个里程碑提交后推送到 GitHub。
  - 本阶段提交已推送 main，远端 SHA 已核验。此要求继续适用于后续里程碑提交。
