# 人类反馈处理：模型依赖

根据 HUMAN_FEEDBACK.md 的明确授权，安装指定模型库及所需运行依赖。

- lightgbm 4.6.0：导入、合成数据训练与概率输出检查通过。
- catboost 1.2.10：导入、合成数据训练与概率输出检查通过。
- 完整测试：25 项，失败 0，错误 0，跳过 0。

初次导入 LightGBM 缺少 libomp.dylib，安装 Homebrew libomp 后重新检查通过。
Python 包安装在本 checkout 的 .venv；libomp 安装于 Homebrew 管理目录，环境二进制不提交 Git。
这些检查仅使用合成数据；M2 尚未开始，冻结协议与研究代码保持不变。

重跑：`.venv/bin/python scripts/check_model_dependencies.py`。
GitHub 推送与反馈完成状态以 HUMAN_FEEDBACK.md、PROGRESS.md 为准。
