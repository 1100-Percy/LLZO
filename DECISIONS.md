# DECISIONS.md

## 实现选择（D-xxx）
格式：D-001 | 里程碑 | 问题 | 选项 | 选择 | 理由

- D-001 | M0 | 当前 checkout 没有 .venv，python 命令不可用；系统 python3 可运行原有测试。选择 `python3 -m venv --system-site-packages --without-pip .venv` 离线复用依赖，Makefile 默认 `.venv/bin/python`。不执行会联网安装的 setup.sh。
- D-002 | M0 | 本研究环境缺少 lightgbm、catboost；不联网、不安装。M0/M1 使用已安装工具继续；M2 的指定模型不能用别的算法冒充，后续需找到已有离线运行时，否则只推进不依赖缺包的任务并明确标注限制。
- D-003 | M0/M1 | 本次启动指令明确要求同一会话完成 M0 与 M1，优先于常驻的每会话单里程碑约定；仍逐里程碑验收与提交。元数据记录生成时 HEAD 与 git_dirty，提交本身收纳生成代码及输出，避免将生成前的 HEAD 误称为完整生成代码版本。

## 协议修改提案（P-xxx，只有人类可以采纳并生成 protocol_v2.yaml）
格式：P-001 | 里程碑 | 建议修改 | 理由 | 对结果的预期影响

## Out-of-scope ideas（不实现）
