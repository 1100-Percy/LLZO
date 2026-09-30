# PROGRESS.md
- M-1 (human, pre-Codex): scaffold, verified data facts (tests/test_data_facts.py), chemistry utils (tests/test_chemistry.py). 14 tests green.

## 后台任务
（无）

## M0 完成
- 已运行原有测试及新增元数据测试，结果见 outputs/M0/validation.csv 与 reports/checkpoints/M0.md。
- 离线 .venv 复用系统依赖；运行 `make test`，或激活 `.venv` 后运行 `python -m pytest -q`。
- 原始文件及已核实模块、测试的校验值已保存。缺少 lightgbm、catboost，见 D-002。
- 下一个步骤：按本次启动指令继续 M1；不执行 M2。
