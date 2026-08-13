# 对手 Exact Deck 分类选择 EDA 设计

## 目标

创建独立 notebook，从最近若干实际数据日期中选出 replay 覆盖量最大的 Exact Deck，作为后续“预测对手 Deck ID”辅助任务的分类集合。

## 输入与参数

Notebook 读取 `data/deck/*.decks.csv` 和现有卡牌表。顶部集中配置：

- `RECENT_DATE_COUNT`：选择最新多少个实际存在的日期。
- `MIN_UNIQUE_REPLAYS`：Exact Deck 至少出现于多少个不同 replay。
- `TOP_K`：过滤后按 replay 数选择前多少个 Exact Deck。
- `OUTPUT_NAME`：输出到 `imitation_learning/data/` 的 CSV 文件名。

## 分类与统计

分类单位是现有 `deck.analysis.exact_deck_identity` 生成的 Exact Deck ID。Archetype 不用于合并或筛选，只使用现有 `classify_deck` 的人工规则优先、named fallback、普通 fallback 逻辑作为解释列。

每个 Exact Deck 统计 unique replay、参与方行数、胜负、胜率、占比、可监督的对手视角样本数与逐日 replay 数。过滤 `unique_replays < m` 后按 replay 数降序、Deck ID 升序稳定排序并取前 `k`，`class_id` 按该顺序从 0 开始。

## 后续损失语义

CSV 内的 Deck 才是有效分类。对手 Deck 不在 CSV 中的训练样本后续设置 `valid_mask=false`，不设 `OTHER` 类，也不计算辅助损失。本 notebook 只输出标签集合和覆盖率，不修改缓存或训练代码。

## 展示与输出

- 数据日期范围和行数质量检查。
- 完整 eligible 表与最终 Top K 表。
- eligible Deck replay 数柱状图，标记 Top K。
- Top K 每日 replay 趋势图。
- Top K 覆盖的 unique replay、参与方行数和可监督对手视角样本比例。
- 保存 `class_id`、Deck ID、archetype、分类方法、统计量及 60 张 Card ID 到 `imitation_learning/data/<OUTPUT_NAME>`。

所有标题和图例使用英文，避免中文字体问题。
