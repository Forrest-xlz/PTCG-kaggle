# 多任务辅助训练设计

## 目标

在纯 BC 策略训练之外增加三个可独立开关和加权的辅助任务，共享同一次 encoder/decoder 前向并合并为一次反向传播：

1. 预测同一玩家下一次决策的 `select.type` 和 `select.context`。
2. 预测对手 Exact Deck 分类。
3. 预测当前玩家终局剩余奖励卡数（0 至 6）。

## 标签

Extract 按 replay 和玩家生成记录。每条记录增加：下一次同玩家决策的 type/context 与 valid mask、对手 60 张牌对应的稳定 Exact Deck ID、当前玩家最后一个有效 observation 中自身 prize 列表长度。下一决策不存在时 mask 无效；对手 Deck 是否有效在 cache 阶段按分类 CSV 判断；终局奖励卡标签对所有正常样本有效。

## 模型表示

下一决策的两个头读取 replay 真实动作在全部 decoder cross-attention 层之后、策略标量头之前的 `d_model` 表示。状态级任务读取可配置的 `global` 或 learned `CLS` encoder 表示。

当选择 `cls` 且至少一个状态任务启用时，在 encoder 序列最前面加入一个永不 mask 的可学习 token。动作 decoder 可以 cross-attention 到该 token。选择 `global` 时不增加 token，读取原 global token 经 encoder 后的输出。状态任务都关闭时不创建 CLS。

辅助分类头均为一层 Linear：type、context、对手 Deck 类别数和 7 类奖励卡。

## 配置与损失

`model.auxiliary` 包含 `state_representation`，以及 `next_decision`、`opponent_deck`、`final_own_prize` 三个 `{enabled, weight}`。对手 Deck 任务另含分类 CSV 路径。

总损失为策略 CE 加各个启用任务的加权 CE。下一决策 type/context 的 CE 先相加，再乘共同 weight。所有损失只调用一次 backward 和 optimizer step。空 valid mask 时该辅助项贡献零且不产生 NaN。

## 指标与推理

训练和每个现有验证命名空间分别记录策略指标及启用任务的 loss/accuracy/valid sample count。策略 CE 和 Top-1/3/5 定义不变。推理仍只使用策略 logits；辅助头不会改变动作选择接口。

## 兼容性

升级 extract schema、cache schema 和 feature signature。旧 JSONL 与旧 cache 明确拒绝，需重新 extract 和 cache。Checkpoint 的 ModelConfig 保存所有辅助架构参数，从而自动恢复对应头和 CLS/global 选择。三个任务全部关闭时，策略主干结构和损失行为保持原样。

## 验证

测试覆盖同玩家下一决策前瞻、终局自身 prize、对手 Deck ID、CSV 映射与 mask、cache 往返、CLS/global 路由、真实动作 token 选取、masked CE、总损失单次 backward 以及辅助关闭时的原行为。
