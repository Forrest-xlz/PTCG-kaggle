# Model architecture and features

The submitted policy scores legal actions with an encoder-decoder Transformer. It has no value head. The 3+3-layer and 4+4-layer models use the same features and differ in Transformer depth. See the [model diagrams](figures/README.md) for the full path from visible state to action score.

## State encoder

The base state has 26 tokens: eight Bench slots per player, one Active Pokémon per player, three player/global summaries, two discard piles, the player's hand, remaining deck, and Stadium. One token summarizing the previous three actions is appended, giving the final model 27 encoder tokens.

Card-bearing regions combine learned Card ID embeddings with static card features. The own-player, opponent-player, and global/select summaries have 69, 71, and 73 numeric values, each projected separately. Region-specific MLPs map all tokens to `d_model=512`. A shared three-state appearance embedding marks Pokémon that are absent, present from an earlier turn, or newly present. Empty Bench slots stay in the fixed layout but are masked from attention.

## Legal-action decoder

The engine supplies the current selection and legal options. Each option is encoded once from its categorical fields, card and attack IDs, five position/index values, static card and attack features, and relevant dynamic Pokémon or attack features. The categorical and numeric projections are summed into a 512-dimensional option token. Missing dynamic features contribute zero.

The model enumerates legal option combinations from the largest allowed selection to the smallest, capped at 64 candidate actions. Option vectors within a candidate are summed; an empty choice uses a learned vector. Each candidate is then a query for decoder cross-attention over the encoded state. The decoder does not use self-attention between candidates. A linear policy head returns one logit per candidate.

## Recent-action history

`model.history_encoding` selects how the previous three decisions are represented:

| Mode | Included information |
|---|---|
| `basic` | Selection type, context, and chosen option types |
| `structural` | Basic fields plus areas, player relations, counts, and conditions |
| `full` | Card, attack, static, and dynamic features, excluding option-position numerics |

Options within each past action are summed and projected. The three action vectors are concatenated in time order and mapped to one history token. Its trainable parameters are separate from the current-action decoder. The final models use `full` history.

## Transformer and compatibility

The final models use eight attention heads, PostNorm, GEGLU feed-forward blocks, and dropout probability 0.1. `model.transformer_dropout` sets the probability for independently enabled embedding, attention-probability, attention-output, and feed-forward-output dropout. Ordinary feature MLPs use ReLU.

`model.norm_mode`, `model.transformer_activation`, projection depths, and dropout switches are configured in [train_policy.yaml](../imitation_learning/cfg/train_policy.yaml). Changing width, depth, GEGLU, or projection depths can change checkpoint weight shapes. Changing the feature layout requires rebuilding the feature cache and keeping inference feature construction in sync. The standard feature cache uses schema 16; the auxiliary-task experiment uses schema 17 for extra labels.
