# Model Architecture and Features

This project uses a pure behavior-cloning policy without a value head. The settings below describe the current configurable architecture, not verified performance gains from every option. See [train_policy.yaml](../imitation_learning/cfg/train_policy.yaml) for complete defaults.

## Transformer, projections, and dropout

`model.norm_mode` accepts `postnorm` or `prenorm`. PreNorm applies
normalization before every encoder/decoder sublayer and adds a final encoder
LayerNorm; PostNorm preserves the original notebook residual ordering.
`model.transformer_activation` selects `relu`, tanh-approximate `gelu`, or
`geglu` for Transformer FFNs only; all ordinary model MLPs retain ReLU.
`model.transformer_dropout` supplies one probability to four independent
switches. `dropout_embedding` applies LayerNorm and dropout to completed
encoder tokens and decoder action queries. `dropout_attention_probs` applies
dropout after attention softmax, `dropout_attention_output` applies it after
the attention output projection and before the residual, and
`dropout_ffn_output` applies it after the second FFN linear and before the
residual. The custom encoder preserves the former TransformerEncoderLayer
PreNorm/PostNorm ordering. ReLU with probability zero and all switches false
is checkpoint-compatible with the old architecture; GEGLU changes the first
FFN weight shape. These settings do not require feature-cache rebuilding.
`model.summary_mlp_layers` and `model.card_mlp_layers` control the projection
depths for numeric-summary tokens and static-card features. The first layer
maps the input width to `d_model`; additional layers are
`ReLU -> Linear(d_model, d_model)`. `model.card_mlp_scope: shared` keeps one
static-card MLP for the whole model. `model.card_mlp_scope: region` gives each
semantic card region its own MLP while sharing it among Pokemon, Tools, and
Energy cards inside that region; decoder cards reuse the corresponding encoder
region MLP. Setting `card_mlp_layers` to zero disables static-card embeddings
while preserving all learned Card ID embeddings.

The encoder has a 26-token base layout: eight bench slots per player, two
active Pokémon, three dense summary tokens, separate discard tokens for both
players, the own hand, remaining-deck estimate, and stadium. The own-player
(69), opponent-player (71), and global/select (73) numeric summaries replace
the old sparse summaries through independent `Linear(n, d_model)` projections.
Missing bench slots remain in the fixed layout but are excluded from encoder
self-attention and decoder cross-attention by a boolean key-padding mask.
Prize counts, selection type, and selection context are one-hot encoded.
`pokemon_appear_embedding` adds one shared three-state embedding (absent,
present from an earlier turn, present this turn) to the 18 Bench/Active Pokemon
tokens. Five `*_token_mlp_layers` settings control eight independent post-token
MLPs: own/opponent Bench, Active, and discard plus own hand and own deck. The
two sides share configured depths but not weights. `region_token_mlp_residual`
selects `token + MLP(token)` or `MLP(token)` globally for these modules.
Feature-layout changes require rebuilding a compatible feature cache; the current cache schema is 16.

The decoder stores each raw engine option once using eleven categorical fields:
option type, selection context, candidate/target Card IDs, Attack ID, number,
Energy count, player relation, area, in-play area, and special condition. Two
routed Pokemon dynamic blocks (46 values total) and six attack-matchup values
are projected separately and masked to exact zero when absent. Five remaining
numeric values (index, Tool index, Energy index, in-play index, and relative
option position) are projected by `model.option_numeric_mlp_layers`. Learned
ID and categorical embeddings, static Card/Attack projections, and these
numeric/dynamic projections are summed in `d_model` space.
`model.option_token_mlp_layers: 0`
uses that sum directly; positive values apply the standard projection MLP to
each completed option token. Exact candidate action combinations are still
enumerated up to 64, and their option tokens are summed before the
cross-attention-only decoder. Rebuild the feature cache after feature-layout
changes.

`model.history_encoding` optionally appends one action-history token to the
encoder. `basic` uses the previous three decisions' select type, select
context, and selected option types; `structural` additionally uses normalized
source/target areas, player relations, number/count, and special condition;
`full` uses independent decoder-like Card, Attack, static, and dynamic
features, while deliberately excluding all five option-position numerics.
Historical options in a combination action are summed, one shared
`history_action_mlp_layers` projection is applied at each of `[t-3,t-2,t-1]`,
and their concatenation is mapped by `history_sequence_mlp_layers` to one
token. All trainable history parameters are independent from the current-action
decoder. Switching among `basic`, `structural`, and `full` reuses the same
schema-16 cache.


## Configuration and compatibility

Architecture is saved in checkpoint config and reconstructed by standalone validation. Changes to depth, width, or settings such as GEGLU may alter parameter shapes; do not assume old weights remain compatible. Feature-layout changes also require a cache rebuild and synchronized inference notebook. Changing training splits alone does not change the architecture.
