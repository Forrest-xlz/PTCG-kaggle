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
