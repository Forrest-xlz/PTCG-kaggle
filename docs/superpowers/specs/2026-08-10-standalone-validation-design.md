# Standalone Validation Evaluator Design

## Goal

Add a standalone `imitation_learning/validation/` evaluator that loads one
checkpoint, automatically reconstructs its architecture, rebuilds exactly the
same validation splits as training, and prints the same CE loss and Top-1/3/5
metrics. It must remain operationally separate from training and must not
change training behavior.

## Files and Entry Point

Create:

```text
imitation_learning/
├── validation/
│   ├── __init__.py
│   └── evaluate.py
└── cfg/
    └── validation.yaml
```

Run from the `imitation_learning` project root with:

```bash
python validation/evaluate.py
```

The evaluator has no command-line arguments.

## Configuration

`cfg/validation.yaml` contains runtime-only evaluation settings:

```yaml
validation:
  train_config: cfg/train.yaml
  checkpoint: outputs/ver_1.6.1/checkpoints/epoch-005.pt
  batch_size: 4096
  device: cuda
  precision: bf16
```

All relative paths resolve from the `imitation_learning` project root. The
referenced training YAML is the single source of truth for:

- `train.cg_path`;
- `train.data`;
- `train.replay_episodes`;
- `train.validation_ratio` and `train.validation_seed`;
- `train.expert_validation_ratio`;
- `train.isolation_validation`;
- `train.top_decks`.

The evaluator applies the training YAML's `${version_name}` interpolation
before reading those values. It ignores training-only settings such as epochs,
optimizer, scheduler, loser augmentation, resume, saving, and WandB.

Validate that batch size is positive, precision is one of `fp32`, `fp16`, or
`bf16`, device syntax is supported, all required train fields exist, and the
checkpoint path exists.

## Model Loading

Accept both full training checkpoints and inference-only checkpoints. Each
must contain mappings named `model` and `config`.

Reconstruct the model with:

```python
config = ModelConfig(**checkpoint["config"])
model = PTCGTransformer(config, card_feature_table, attack_feature_table)
model.load_state_dict(checkpoint["model"], strict=True)
```

Build the card and attack feature tables from the configured competition
engine. Set the model to evaluation mode and use `torch.inference_mode()`.
Dropout is therefore disabled. The configured precision uses the existing
mixed-precision semantics: FP32 without autocast, FP16 autocast where
supported, and BF16 autocast where supported.

The architecture is read explicitly from checkpoint metadata rather than
inferred from tensor shapes. A checkpoint produced by incompatible source code
or a cache with a different feature signature must fail clearly.

## Validation Split Parity

Rebuild splits from the configured cache using the same components and order
as training:

1. Resolve the three reviewed replay-level isolation validation sets.
2. Load per-date expert episode keys using `expert_validation_ratio`.
3. Reserve the latest cache date as latest-date validation.
4. Select older in-distribution validation replays using
   `validation_ratio/validation_seed`.
5. Build expert and ordered top-deck masks inside the two base validation
   sets.

The current cache's player-result field keeps every validation array
winner-only exactly as in training. Loser augmentation is irrelevant because
it changes only the training indices after validation has been fixed.

## Metric Parity and Forward Reuse

Match training evaluation semantics exactly:

- mask logits beyond each sample's legal `action_count`;
- calculate cross-entropy against the replay target;
- calculate Top-1, Top-3, and Top-5 accuracy from masked logits;
- accumulate loss weighted by sample count;
- run each base dataset once and derive subgroup metrics from the same logits.

Evaluate and print these namespaces:

- `val_deck_isolation`;
- `val_archetype_isolation`;
- `val_top_deck_archetype_isolation`;
- `val_in_distribution` and `val_in_distribution_expert`;
- ordered `val_in_distribution_deckN` and
  `val_in_distribution_expert_deckN` groups;
- `val_latest` and `val_latest_expert`;
- ordered `val_latest_deckN` and `val_latest_expert_deckN` groups.

Isolation union is forwarded once. Latest-date and in-distribution each use one
forward traversal, regardless of their number of expert/top-deck subgroups.

Print each metric group as:

```text
val_latest samples=12,345 loss=0.9876 top1=0.701 top3=0.882 top5=0.931 seconds=4.21
```

Subgroups derived from a base traversal may omit their own duration or print
the base traversal duration only once. Also print the checkpoint, device,
precision, cache sample count, split sample counts, and latest validation date
at startup.

## Decoupling and Side Effects

The standalone evaluator must not:

- call `training.train.main()`;
- instantiate an optimizer or scheduler;
- update model parameters;
- initialize WandB;
- write checkpoints, histories, metrics, or result files;
- modify cache, extracted replay data, or split-selection CSVs.

It may reuse read-only model, feature-cache, precision, expert-validation, and
isolation-validation modules. To avoid changing training behavior, standalone
metric orchestration remains in `validation/evaluate.py`. Tests compare its
metric calculations and subgroup construction against training behavior.

## Verification

Tests cover:

- validation YAML and referenced train YAML parsing/interpolation;
- automatic reconstruction from checkpoint `config`;
- rejection of malformed checkpoints and invalid runtime settings;
- validation namespace order and top-deck numbering;
- legal-action masking and CE/Top-1/3/5 parity with training;
- subgroup metrics sharing base forward calls;
- no optimizer, scheduler, WandB, or write paths in the evaluator.

Run focused unit tests and compile the new package. A smoke evaluation may use
an existing local checkpoint and cache when both are available; otherwise the
handoff must state the exact configured command to run in the training
environment.
