# PTCG 关键实验图

本版保留六组实验，均为 CE 曲线：训练主任务（日志 EMA）、分布内验证、最新日期验证、最新日期高手子集验证。CE 越低越好。图注、run 链接、数值表与局限见[实验结果报告](../solution-experiment-results.md)。

| 图号 | 内容 | PNG | SVG |
|---|---|---|---|
| 01 | 数据规模：Tiny / Small，1% / 10% / 30% / 100% | [PNG](01_data_scaling.png) | [SVG](01_data_scaling.svg) |
| 02 | 模型规模：Tiny / Small / Base | [PNG](02_model_scaling.png) | [SVG](02_model_scaling.svg) |
| 03 | 三步动作历史：Off / Basic / Structural / Full | [PNG](03_action_history.png) | [SVG](03_action_history.svg) |
| 04 | GEGLU / GELU 与 dropout | [PNG](04_activation_dropout.png) | [SVG](04_activation_dropout.svg) |
| 05 | 多任务学习与纯 BC | [PNG](05_multitask.png) | [SVG](05_multitask.svg) |
| 07 | effect/contextCard 特征 | [PNG](07_effect_context.png) | [SVG](07_effect_context.svg) |

精确绘图数据位于 `outputs/solution_analysis/chart_data.json`；每张图提供 PNG 和 SVG。

## 模型结构与 EDA

最终模型结构分为三张图，可在[完整 write-up](../../kaggle_writeup.md)中按流程阅读：

| 流程 | PNG | SVG |
|---|---|---|
| 26 个状态 token 与三步历史 token | [PNG](model_state_tokens.png) | [SVG](model_state_tokens.svg) |
| 合法选项编码与候选动作 query | [PNG](model_action_tokens.png) | [SVG](model_action_tokens.svg) |
| Encoder、Decoder 与双模型集成 | [PNG](model_policy_ensemble.png) | [SVG](model_policy_ensemble.svg) |

图像由 [build_model_diagrams.py](build_model_diagrams.py) 生成。

EDA 图从 `imitation_learning/outputs/` 复制到此处，便于在 GitHub 上阅读；原始生成脚本位于 `imitation_learning/eda/` 和 `imitation_learning/eda/deck_eda.ipynb`。其中包括[精确卡组覆盖](eda/top_deck_coverage.png)、[同 archetype 变体覆盖](eda/top_deck_archetype_coverage.png)、[使用率及胜率时间趋势](eda/archetype_share_and_win_rate.png)、[高分段卡组切换](eda/archetype_sankey.png)、[对局关系矩阵](eda/archetype_matchup_matrix.png)和[队伍推理耗时](eda/score_filtered_startup_vs_action.png)。
