# 三步动作历史 Token 设计

## 目标

在不改变当前策略 decoder 交互方式的前提下，将当前行动玩家最近三次已执行动作编码为一个新的 encoder token，使模型能够学习诸如“打牌 → 搜索 → 进化”和“进化 → 赋能 → 攻击”这样的短期动作序列。

历史表示提供三个可切换的信息粒度，用于区分收益来自抽象动作语法、结构化动作信息还是具体卡牌语义。历史模块拥有独立的可训练参数，不复用当前 decoder 的 embedding 或投影层。

## 配置接口

模型配置新增：

```yaml
model:
  history_encoding: basic       # off / basic / structural / full
  history_action_mlp_layers: 1
  history_sequence_mlp_layers: 2
```

- `off`：不构造历史特征，保持当前26个 encoder tokens。
- `basic`、`structural`、`full`：构造一个历史 token，encoder token 数变为27。
- 历史窗口固定为3步，baseline 阶段不增加可配置窗口长度。
- `history_action_mlp_layers` 可以为0；0表示单步动作 embedding 求和后不再变换。
- `history_sequence_mlp_layers` 在启用历史时必须至少为1。第一层为 `3*d_model -> d_model`，后续层均为 `d_model -> d_model`。

只实例化当前 `history_encoding` 对应的分支，避免未使用版本增加 checkpoint 大小。

## 历史范围

历史按 replay、玩家分别维护。当前样本只读取当前行动玩家此前最近三次决策动作，顺序为：

```text
[t-3, t-2, t-1]
```

对手历史暂不加入。切换 replay 或玩家历史尚不足三步时在左侧补空位置。一个 replay 内的历史必须来自真实时间顺序，不能从经过 shuffle 的训练样本反推。

## 通用动作聚合

一个引擎 action 是选中 option 的无序集合。每个被选 option 先独立编码，同一动作内的 option embedding 直接求和：

```python
raw_action = decision_embedding + sum(selected_option_embeddings)
```

重复选择相同 OptionType 时重复相加，因此组合动作的数量会反映在向量幅度中。

合法空选择使用独立的 `history_no_action_embedding`。不存在的历史位置使用零向量，并在单步 MLP 后乘 `history_valid` mask，防止 MLP bias 把 padding 变成非零历史。

## 版本一：basic

只表达动作语法：

```python
decision_embedding = (
    history_select_type_embedding[select_type]
    + history_select_context_embedding[select_context]
)

selected_option_embedding = (
    history_option_type_embedding[option_type]
)
```

该版本不包含区域、索引、Card ID、Attack ID 或动态状态。它可以学习 `PLAY -> ATTACH -> ATTACK` 等抽象模式。

## 版本二：structural

在 basic 的决策信息上，每个 option 增加：

```python
structural_option_embedding = (
    history_option_type_embedding[option_type]
    + history_source_area_embedding[source_area]
    + history_target_area_embedding[target_area]
    + history_source_relation_embedding[source_player_relation]
    + history_target_relation_embedding[target_player_relation]
    + history_number_embedding[number]
    + history_count_embedding[count]
    + history_special_condition_embedding[special_condition]
)
```

来源和目标区域必须先按 OptionType 规范化，不能只复制可能为空的原始字段。例如：

- `PLAY` 的来源规范化为我方手牌。
- `ATTACK` 的来源和目标规范化为我方 Active 与对方 Active。
- `RETREAT` 的来源规范化为我方 Active。
- `ATTACH`、`EVOLVE` 使用 `area` 与 `inPlayArea`。

source/target relation 使用两个独立的历史 embedding table，避免动作方向在求和后产生歧义。

该版本不包含 Card ID、Attack ID、静态卡牌特征、动态宝可梦特征或任何实例索引。

## 版本三：full

使用与当前 decoder option 相同的信息集合，但所有可训练 embedding 和投影层均属于历史模块：

```python
full_option_embedding = (
    history_option_type_embedding
    + history_select_context_embedding
    + history_candidate_embedding
    + history_target_embedding
    + history_attack_embedding
    + history_number_embedding
    + history_count_embedding
    + history_player_relation_embedding
    + history_area_embedding
    + history_in_play_area_embedding
    + history_special_condition_embedding
    + history_candidate_static_projection
    + history_target_static_projection
    + history_attack_static_projection
    + history_pokemon_dynamic_projection
    + history_attack_dynamic_projection
)
```

固定的54维 Card 特征表和14维 Attack 特征表可以作为非训练数据源共享；将它们映射到 `d_model` 的历史投影层不与 decoder 共享。

full 明确排除当前五维数值分支：

```text
option.index
toolIndex
energyIndex
inPlayIndex
option relative position
```

因此历史不会记录 option 列表排序或物理实例位置。过去状态中的索引只允许在特征提取阶段用于解析 candidate、target 和动态 Pokémon，解析结果中不保留索引数值。

`select.type` 仍以动作级历史 embedding 加一次；`select.context` 已包含在每个 full option 中，不在动作级重复加入。

## 两级 MLP

三个时间位置严格共享同一个单步动作 MLP：

```python
a_t3 = history_action_mlp(raw_t3) * valid_t3
a_t2 = history_action_mlp(raw_t2) * valid_t2
a_t1 = history_action_mlp(raw_t1) * valid_t1
```

该 MLP 的每层均为 `d_model -> d_model`。三个时间位置不能各自持有参数。

随后按固定时间顺序拼接：

```python
sequence = torch.cat([a_t3, a_t2, a_t1], dim=-1)
history_token = history_sequence_mlp(sequence)
```

concat 本身确定三个时间位置，因此不增加时间位置 embedding，也不使用 1D CNN。

## Encoder 集成

`history_token` 作为新的固定槽位加入主 encoder。至少存在一个有效历史动作时该 token 可见；三个位置均无历史时在 encoder padding mask 中屏蔽。

历史 token 与其他状态 token 一起通过主 Transformer encoder。当前 decoder 仍然让每个候选 action 独立 cross-attend 同一个 encoder 输出，候选动作之间不增加 self-attention。

## 数据与缓存

历史必须在 replay 的顺序处理中产生。训练记录或 cache 需要保存每个样本对应的三个历史动作原始字段；不能保存模型产生的 embedding。

- basic 只需保存三组 `select.type`、`select.context` 和已选 OptionType 列表。
- structural 额外保存规范化后的区域、玩家关系、number、count 和特殊状态。
- full 需要保存过去 option 的 categorical、Pokémon dynamic 和 attack dynamic 特征；静态 Card/Attack 表继续由模型构建。

历史 schema 应保存 full 所需的超集，使同一份 cache 可以切换 basic、structural、full，而不重复生成三套 cache。启用本设计需要提升 cache schema，并重新生成 feature cache；若现有 extracted JSONL 保持 replay 内严格时间顺序且包含完整 observation/selected，则无需重新从原始 replay 压缩包提取。

## 验证与消融

使用完全相同的数据划分和训练配置分别训练：

```text
off
basic
structural
full
```

比较 train、分布内验证、最新日期、deck isolation 和 card isolation 的 CE 与 Top-1/3/5。重点判断：

- basic 的提升表示抽象动作序列有效。
- structural 相对 basic 的提升表示区域与动作方向有效。
- full 相对 structural 的提升表示具体卡牌与动态状态有效。
- full 若只提高分布内指标并降低 card isolation，说明模型依赖 deck 特有连招记忆。

单元验证至少覆盖单选、组合动作、合法空选择、少于三步历史、跨 replay 清空、按玩家隔离历史，以及 full 中五个 numeric/index 字段确实没有进入模型。

## 非目标

本次不加入对手动作历史、日志序列、可变历史长度、1D CNN/RNN、历史 Card serial、历史 option 相对位置，也不改变当前 decoder 架构。
