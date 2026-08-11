# Isolated Fine-Tuning Design

## Goal

Add an independent fine-tuning entry point that starts from a complete epoch
checkpoint, preserves the current training validation sets, and trains on a
replay-level random in-distribution subset plus expert winning samples from one
configured exact deck.

## Files and Isolation

Create `imitation_learning/finetune/finetune.py` and
`imitation_learning/finetune/__init__.py`. Configuration lives in
`imitation_learning/cfg/finetune.yaml`. The implementation may import stable
helpers and data structures from `training.train`, `training.feature_cache`,
and `training.expert_validation`, but it does not change the behavior of the
existing training entry point, feature extraction, cache schema, model, or
standalone validation command.

## Configuration

The fine-tuning YAML contains:

```yaml
version_name: finetune_deck1

finetune:
  train_config: cfg/train.yaml
  checkpoint: outputs/ver_x/checkpoints/epoch-005.pt
  output: outputs/${version_name}
  epochs: 2
  in_distribution_ratio: 0.1
  in_distribution_seed: 42
  expert_ratio: 0.05
  deck: [60 exact card IDs]
  learning_rate: 0.00003

wandb:
  enabled: false
  project: ""
  group: ""
  name: ${version_name}
  mode: online
```

`train_config` supplies all data paths, validation settings, top-deck groups,
model-independent runtime defaults, and unchanged training parameters. The
fine-tune section must specify checkpoint, output, epochs, replay sampling
ratio and seed, expert ratio, and one exact 60-card deck. `learning_rate` is an
optional override; when omitted it inherits the training learning rate. All
other optimization, precision, batching, evaluation, checkpoint, logging,
and seed settings inherit from the referenced training YAML.

Validate `in_distribution_ratio` and `expert_ratio` in `(0, 1]`, epochs as a
positive integer, the deck as exactly 60 non-negative integer card IDs, and
all paths before opening the model.

## Validation and Fine-Tune Data

Load isolation selections and daily validation expert sets using the referenced
training configuration. Build the full split after isolation, latest-date, and
in-distribution validation assignment with `train_replay_ratio=1.0`. These
validation arrays and masks are the ones evaluated during fine-tuning and must
match `training/train.py` for the same validation configuration.

Build a second split with identical validation inputs but
`train_replay_ratio=in_distribution_ratio` and
`train_replay_seed=in_distribution_seed`. Its training indices form the
generic in-distribution fine-tune component, sampled by replay rather than by
individual state/action samples.

For every date independently, use the existing expert-validation score
quantile calculation with `expert_ratio`. From the complete post-validation
training split, select samples satisfying all of:

- the replay is expert for its date;
- the acting player's exact `deck_key` matches the configured fine-tune deck;
- `player_result == WIN`.

Append all selected expert-deck indices to the sampled generic indices. Do not
deduplicate overlap: an expert-deck sample already present in the generic
component appears twice and therefore receives natural extra weight. Shuffle
the combined array once per epoch with `seed + epoch_index`. Validation indices
are never appended to training.

Fail before training if the generic component or expert-deck component is
empty. Print and log the full eligible training count, selected generic replay
and sample counts, each date's expert cutoff, expert-deck sample count, overlap
count, and combined samples per epoch.

## Checkpoint and Optimization

Require a complete epoch checkpoint containing model, optimizer, config,
scaler, epoch, and training state. Reconstruct the architecture exclusively
from `checkpoint["config"]`, then load model and AdamW optimizer states. Restore
the mixed-precision scaler when compatible.

Fine-tuning is a new run: reset global step, EMA values, epoch history, and
Wandb run. Create a new warmup-plus-cosine scheduler using the inherited
`warmup_steps`, fine-tune epoch count, batch size, and combined fine-tune sample
count. The optional fine-tune learning-rate override replaces the optimizer
parameter-group learning rate after loading its state and becomes the target
learning rate for the new scheduler.

## Training, Evaluation, and Saving

Reuse the current policy loss, Top-1/3/5 metrics, mixed precision, EMA logging,
evaluation cadence, validation subgroup names, step checkpoints, and epoch
checkpoints. Output only under the fine-tune output directory or the current
Wandb run's local-output directory, matching existing behavior. W&B receives
metrics and configuration only; checkpoints are not uploaded.

No extraction or cache rebuild is required.

## Tests

Test YAML inheritance and validation, replay-level generic sampling,
date-specific expert cutoffs, exact-deck and winner-only filtering, permitted
overlap duplication, validation-array equality between full and sampled split
construction, complete-checkpoint requirements, optimizer restoration with a
fresh scheduler, deterministic epoch shuffling, logging metadata, and source
isolation from `training/train.py`.
