# Validation, EMA, PreNorm, and Training Simplification Design

## Goal

Extend the pure behavior-cloning baseline with replay-grouped validation,
time-based validation, cross-training EMA metrics, configurable transformer
normalization, step-based evaluation/checkpointing, and a simpler global-only
data loader.

## Configuration

`cfg/train.yaml` keeps the existing `version_name`, `train`, `model`, and
`wandb` hierarchy. The relevant fields become:

```yaml
version_name: pure-bc-e4-d4-v1

train:
  cg_path: ../pokemon_tcg_ai_battle/sample_submission
  data: data/training-cache
  output: outputs/${version_name}
  epochs: 3
  batch_size: 2048
  learning_rate: 0.0003
  weight_decay: 0.01
  beta1: 0.9
  beta2: 0.999
  warmup_steps: 100
  max_samples: null
  seed: 42
  device: cuda
  precision: bf16
  log_every_steps: 10
  eval_every_steps: 1000
  save_every_steps: 1000
  save_every_epoch: true
  ema_alpha: 0.99
  validation_ratio: 0.05
  validation_seed: 42
  grad_clip_norm: 1.0

model:
  d_model: 512
  ffn_multiplier: 4
  num_heads: 8
  encoder_layers: 3
  decoder_layers: 3
  norm_mode: postnorm

wandb:
  enabled: true
  project: ptcg_bc
  group: pure_bc
  name: ${version_name}
  mode: online
```

`shuffle_mode` and `warmup_ratio` are removed. `norm_mode` accepts only
`prenorm` or `postnorm`. `validation_ratio` must be strictly between zero and
one. Step intervals must be positive integers. `ema_alpha` must be in
`[0, 1)`.

## Cache Schema and Episode Identity

The packed feature-cache schema adds one `uint32 episode_key` per sample.
The key is a stable, process-independent hash of the replay episode ID.
Every sample extracted from the same replay receives the same key.

The cache shard metadata already records its source JSONL filename. Training
parses the filename stem as a numeric `(month, day)` tuple, so `7.19` sorts
after `7.5`, and `7.24` sorts after `6.30`. An unparseable source date is a
hard error rather than silently producing a bad temporal split.

This schema change invalidates existing feature caches. Replay JSONL extraction
does not need to be repeated because it already contains `episode_id` and
`date`.

The additional persistent cost is four bytes per sample, approximately 120 MB
for 30 million samples. It is memory-mapped with the other cache sections and
does not create Python objects per sample.

## Validation Split

Training opens all cache shards once and builds three global-index collections:

1. `val_latest`: every sample from every shard whose date equals the maximum
   parsed source date.
2. `val_in_distribution`: samples from all older dates whose mixed episode key
   falls below the configured `validation_ratio` threshold.
3. `train`: every remaining sample from older dates.

The split decision is:

```text
mixed_key = deterministic_mix(episode_key, validation_seed)
is_validation = mixed_key / 2**32 < validation_ratio
```

Because the decision uses an episode-level key, all samples from a replay stay
in one split. The ratio is statistically approximately 5%, not an exact replay
count. Changing `validation_ratio` or `validation_seed` does not require a
cache rebuild.

The latest date is excluded completely from training and from the
in-distribution validation pool. Training fails clearly if the split leaves no
training samples, no in-distribution validation samples, or no latest-date
validation samples.

`max_samples` applies only to the training index collection. Both validation
sets are always evaluated in full.

## Global-Only Loading

Shard-order shuffle is removed. `MmapFeatureDataset` exposes only global-index
batch iteration:

```python
iter_index_batches(indices, batch_size, seed, shuffle)
iter_batches(indices, batch_size, seed, shuffle)
```

Training indices are globally shuffled in place with `seed + epoch` and each
eligible sample is visited once per epoch, subject to `max_samples`.
Validation indices are traversed without shuffling. The implementation retains
compact `uint32` indices when the full cache fits in that range and uses
`uint64` only when necessary.

## Training Metrics and EMA

Each training batch computes masked cross-entropy plus Top-1, Top-3, and Top-5
accuracy over legal action candidates. For samples with fewer than `k` legal
actions, Top-k uses all legal actions.

The training display uses an exponential moving average that persists across
epoch boundaries:

```python
ema_0 = first_batch_value
ema_t = ema_alpha * ema_(t-1) + (1 - ema_alpha) * batch_value
```

There is no bias correction. EMA values update for every processed batch,
including a batch whose optimizer update is skipped because of mixed-precision
overflow; optimizer-step-based logging and triggers do not advance for a
skipped update.

WandB and console training metrics are:

```text
train/ema_loss
train/ema_top1_accuracy
train/ema_top3_accuracy
train/ema_top5_accuracy
train/learning_rate
train/grad_norm
train/grad_scale
train/samples_per_second
train/optimizer_step
```

Exact sample-weighted epoch aggregates remain available:

```text
epoch/loss
epoch/top1_accuracy
epoch/top3_accuracy
epoch/top5_accuracy
epoch/samples
epoch/seconds
epoch/samples_per_second
```

Epoch aggregates reset at each epoch; EMA state does not.

## Validation Evaluation

Every `eval_every_steps` successful optimizer updates, training pauses and
fully evaluates both validation collections using `model.eval()`,
`torch.inference_mode()`, the configured mixed-precision autocast context, and
the training batch size. Training mode is restored afterward.

Training completion always forces one final evaluation unless validation was
already run at the final optimizer step. Epoch boundaries do not independently
trigger validation.

Metrics are sample-weighted across each complete validation collection and use
separate WandB namespaces:

```text
val_in_distribution/loss
val_in_distribution/top1_accuracy
val_in_distribution/top3_accuracy
val_in_distribution/top5_accuracy
val_in_distribution/samples
val_in_distribution/seconds

val_latest/loss
val_latest/top1_accuracy
val_latest/top3_accuracy
val_latest/top5_accuracy
val_latest/samples
val_latest/seconds
```

All WandB metric namespaces use successful optimizer step as their explicit
horizontal axis.

## Learning-Rate Schedule

`warmup_steps` replaces `warmup_ratio`. It must be non-negative and smaller
than the total number of optimizer steps.

For positive `warmup_steps`, learning rate rises linearly from zero to the
configured target learning rate over exactly that many successful optimizer
updates. With zero warmup steps, training starts immediately at the target
rate. After warmup, cosine decay reaches zero on the final planned optimizer
step. Skipped mixed-precision updates do not advance the schedule.

## Checkpointing and Output Routing

`save_every_steps` saves a checkpoint after every configured number of
successful optimizer updates:

```text
checkpoints/step-00001000.pt
```

When `save_every_epoch` is true, every epoch also saves:

```text
checkpoints/epoch-001.pt
```

Step and epoch checkpoints can coexist even when both triggers occur after the
same batch. Checkpoints contain model state, optimizer state, scheduler state,
precision-scaler state where applicable, global step, epoch, resolved model
configuration, resolved experiment configuration, EMA state, and training
history. This makes the saved state internally complete even though resume
training is not added in this baseline change.

Output root selection is:

- WandB enabled: the local `wandb_run.dir`; checkpoints are not uploaded as
  WandB artifacts and WandB uploads metrics only.
- WandB disabled: resolved `train.output`.

The `checkpoints` directory, resolved configuration, and `history.json` all
live under that selected root. `train.output` remains required because it is
the disabled-WandB fallback.

## Transformer Normalization

`ModelConfig` records `norm_mode`, and both encoder and decoder follow it.

PostNorm preserves the current architecture:

```text
encoder sublayer: x = norm(x + sublayer(x))
decoder attention: x = norm1(x + cross_attention(x, encoder))
decoder FFN:       x = norm2(x + ffn(x))
```

PreNorm uses:

```text
encoder sublayer: x = x + sublayer(norm(x))
decoder attention: x = x + cross_attention(norm1(x), encoder)
decoder FFN:       x = x + ffn(norm2(x))
```

The PreNorm encoder stack has a final LayerNorm, and decoder cross-attention
therefore receives normalized encoder memory. Decoder layers continue to
cross-attend independently to the same encoder output and never perform
self-attention between candidate actions.

The Kaggle submission notebook mirrors both normalization modes and reads the
mode from checkpoint configuration. Checkpoints without `norm_mode` default to
`postnorm`.

## Baseline Simplification

The change removes:

- shard shuffle configuration, validation, loader implementation, tests, and
  README documentation;
- epoch-local `running_loss` and `running_accuracy`;
- `warmup_ratio`;
- the old six-argument network forward compatibility path that supplied
  explicit all-one decoder weights.

It retains mixed precision, gradient clipping, AdamW, global shuffling,
`max_samples`, epoch summaries, epoch checkpoints, and the current pure-policy
head.

## Testing

Tests use small real packed shards and tiny models. They cover:

1. cache round-trip preservation of `episode_key`;
2. deterministic, replay-grouped in-distribution splitting;
3. complete exclusion of the latest date from training;
4. numeric month/day ordering;
5. global shuffle visiting each selected training sample exactly once;
6. EMA initialization, update formula, and persistence across an epoch
   boundary;
7. masked CE and Top-1/3/5 calculations with fewer than five legal actions;
8. separate full-validation aggregates for both validation collections;
9. warmup and cosine values at schedule boundaries;
10. PreNorm and PostNorm encoder/decoder construction and forward shape;
11. step/epoch checkpoint trigger filenames;
12. WandB-enabled and WandB-disabled local output-root selection;
13. configuration rejection for invalid ratios, intervals, normalization
    modes, and warmup lengths.

The full test suite must pass before the change is considered complete.
