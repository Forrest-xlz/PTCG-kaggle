# Encoder Value Training Design

## Goal

Add an isolated value-training workflow that predicts the acting player's
eventual replay result from the current observation. Reuse a trained policy
checkpoint to initialize every encoder component, fine-tune the complete
encoder, and train a new scalar value head. Existing policy training, Beam
Search, extraction, and feature-cache behavior remain unchanged.

The supervised target is stored in the existing feature cache:

- win: `1.0`
- draw: `0.0`
- loss: `-1.0`

No replay extraction or feature-cache rebuild is required.

## Project Boundary

Add an independent `imitation_learning/value/` package and
`imitation_learning/cfg/value_train.yaml`. The value workflow may reuse
read-only utilities from `training/` for cache access, replay metadata,
expert cutoffs, isolation selections, scheduling, and mixed precision. It
must not change the semantics of `training/train.py` or `cfg/train.yaml`.

Suggested files:

- `value/model.py`: encoder backbone and scalar value model.
- `value/data.py`: value-specific replay split construction.
- `value/metrics.py`: streaming regression metrics.
- `value/train.py`: configuration, training, evaluation, checkpointing, and
  WandB integration.
- `cfg/value_train.yaml`: all value-training parameters.

## Model Architecture

Refactor the existing network implementation so policy and value models share
one `PTCGEncoderBackbone`. The backbone owns all modules needed before and
inside the main Transformer encoder:

- encoder card embeddings and static-card projections;
- own-player, opponent-player, and global-summary projections;
- Pokémon appearance and region-token transformations;
- historical-action encoding when enabled;
- encoder input normalization and dropout;
- Transformer encoder layers and padding-mask construction.

The policy model continues to call the same backbone before its existing
option encoder and cross-attention decoder. Its logits and checkpoint behavior
must remain numerically unchanged.

The value model contains only:

```text
PTCGEncoderBackbone
  -> post-Transformer global token at base encoder position 25
  -> configurable value MLP
  -> scalar
  -> tanh
```

The optional history token is appended after the 26 base tokens, so it does
not change the global-token index. The value MLP uses `d_model -> d_model` for
each hidden layer and `d_model -> 1` for its output. Its final linear layer is
initialized with zero weights and zero bias, giving an initial prediction of
zero instead of a random value. Hidden layers use the existing project MLP
activation convention. `value_model.head_layers` must be at least one; a value
of one means the direct `d_model -> 1` projection.

No action-option encoding, policy decoder, or policy-logit computation occurs
in value forward passes. Policy-only modules are neither moved to the GPU nor
included in the optimizer.

## Policy Checkpoint Initialization

`value_train.pretrained_checkpoint` is required. The loader reads the saved
model configuration from the policy checkpoint and treats it as the only
source of encoder architecture. The YAML does not repeat `d_model`, attention
heads, Transformer depth, feature-projection depth, or history mode.

The loader maps all policy encoder weights into `PTCGEncoderBackbone` and
strictly verifies that every expected backbone tensor is present with the
correct shape. Policy decoder tensors are intentionally ignored. Missing or
incompatible encoder weights are fatal errors rather than silently randomized
parameters. Only the new value-head hidden layers require normal initialization;
the final output layer is zero-initialized.

The whole encoder and value head are trainable. Value checkpoints save the
resolved encoder configuration, backbone and value-head weights, optimizer,
scheduler, precision state, global step, epoch, EMA state, and training
history so value training can be resumed independently.

## Value-Specific Replay Splits

The existing BC `build_splits()` cannot be reused directly because its
validation sets apply a winner-only mask. Value validation must contain all
outcomes. A value-specific split builder reuses the same replay allocation
rules while retaining every sample from each selected replay.

Split precedence is:

1. configured isolation-validation replays;
2. every replay from the latest cache date;
3. replay-hashed in-distribution validation from remaining dates;
4. all remaining replay samples for training.

The split is replay-level: both players and every step of one replay always
belong to the same partition. The training set excludes every validation
replay. Win, draw, and loss samples are retained in all partitions. Value
training therefore does not use BC loser augmentation.

The validation namespaces match policy training:

- `val_in_distribution`
- `val_in_distribution_expert`
- `val_in_distribution_deckN`
- `val_in_distribution_expert_deckN`
- `val_latest`
- `val_latest_expert`
- `val_latest_deckN`
- `val_latest_expert_deckN`
- `val_deck_isolation`
- `val_archetype_isolation`
- `val_top_deck_archetype_isolation`

`deckN` follows the order of `value_train.top_decks`. Expert replay selection,
top-Deck matching, and isolation CSV interpretation use the same utilities and
configuration semantics as policy training. Subgroups are masks over the two
base validation passes, so a base validation dataset is inferred once per
evaluation and all of its expert/Deck metrics are accumulated simultaneously.

## Objective and Metrics

Training minimizes sample-mean MSE between the tanh value and the target.
Validation reports:

```text
rmse = sqrt(mean((target - prediction)^2))
explained_variance = 1 - Var(target - prediction) / Var(target)
```

Metrics are accumulated from sums and squared sums over the complete dataset
or subgroup; batch metrics are not averaged. When `Var(target) == 0`, explained
variance is recorded as `NaN`, while RMSE remains valid. Each validation group
also records sample count, target mean, and prediction mean.

Training logs MSE/RMSE EMA, an explained-variance diagnostic, learning rate,
gradient norm, precision scale, throughput, and data-wait percentage. The EMA
persists across epoch boundaries, matching policy-training behavior.

## Optimization and Evaluation

Use AdamW, configurable beta values and weight decay, gradient clipping,
warmup steps followed by cosine decay, and configurable `fp32`, `fp16`, or
`bf16` mixed precision. The encoder and value head receive one joint backward
pass per batch.

`value_train.eval_every_steps` triggers evaluation by optimizer step. Step and
epoch checkpoint intervals follow `save_every_steps` and `save_every_epoch`.
Training and evaluation use the existing packed mmap cache and batch loader;
no observation parsing or feature construction occurs inside the training
loop.

## Configuration

The independent configuration has three layers:

```yaml
version_name: value_v1

value_train:
  cg_path: ../pokemon_tcg_ai_battle/sample_submission
  data: data/training_cache
  replay_episodes: ../replay_episodes
  pretrained_checkpoint: outputs/ver_1.6.0/checkpoints/epoch-005.pt
  output: outputs/${version_name}
  resume: false
  resume_checkpoint: null
  epochs: 3
  batch_size: 4096
  learning_rate: 0.0001
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.999
  warmup_steps: 100
  eval_every_steps: 200
  save_every_steps: 1000
  save_every_epoch: true
  validation_ratio: 0.05
  validation_seed: 42
  expert_validation_ratio: 0.05
  seed: 42
  device: cuda
  precision: bf16
  log_every_steps: 10
  grad_clip_norm: 1.0
  ema_alpha: 0.99
  isolation_validation: {}
  top_decks: []

value_model:
  head_layers: 2
  dropout: 0.0
  output_activation: tanh

wandb:
  enabled: true
  project: ptcg_value
  group: value_pretrain
  name: ${version_name}
  mode: online
```

The actual YAML carries the same isolation-selection and top-Deck entries as
the policy configuration. `output_activation` accepts only `tanh` in this
baseline so the value always remains in `[-1, 1]`.

## WandB and Outputs

WandB receives metrics only; it never uploads checkpoints. Each validation
namespace is separately defined against `optimizer_step`, with metrics such as
`val_latest/rmse` and `val_latest/explained_variance`. Value checkpoints and
the resolved YAML are written under `value_train.output`.

## Error Handling

Training stops with a descriptive error when:

- the policy checkpoint lacks a complete saved model configuration;
- any expected encoder tensor is missing or shape-incompatible;
- the cache signature is incompatible with that encoder configuration;
- a required validation set or configured Deck subgroup is empty;
- warmup steps are not smaller than total optimizer steps;
- mixed precision or CUDA configuration is unsupported.

An empty optional subgroup is reported explicitly during split construction;
configured required subgroups retain the current strict validation behavior.

## Verification

Tests cover:

- exact result-code to target mapping;
- replay-level, all-outcome split isolation;
- encoder checkpoint mapping and rejection of incomplete weights;
- policy output equivalence before and after backbone refactoring;
- decoder modules absent from value parameters and forward execution;
- global-token selection with history enabled and disabled;
- zero initial scalar output and `[-1, 1]` tanh bounds;
- exact RMSE and explained-variance accumulation, including zero variance;
- subgroup metrics from one validation pass;
- evaluation-step scheduling, checkpoint payload, and WandB namespaces.

A smoke test loads a real policy checkpoint and one real cache batch, performs
one value optimizer step, runs the latest and in-distribution evaluations, and
saves/reloads a value checkpoint.
