# Final Full-Training Design

## Goal

Create the final competition-training branch by applying the full-data training
behavior from `ver_final_0` to the current `ver_final_f` model and feature stack.

## Training data

- Every normal cached training sample is eligible for every epoch. No replay,
  date, deck, expert, or isolation validation subset is held out.
- Keep the `ver_final_0` expert-loser augmentation: qualifying loser samples
  from recent dates are added to the full normal dataset.
- `max_samples` and `train_replay_ratio` remain optional training-volume
  controls; neither is a validation split.

## Removed train-time behavior

- Remove in-distribution and latest-date validation construction.
- Remove expert, exact-deck, and isolation validation subsets.
- Remove periodic validation evaluation and validation W&B namespaces.
- Remove validation-only configuration from `cfg/train.yaml`.

The standalone `imitation_learning/validation/` package remains available and
is not imported by the final training loop.

## Preserved behavior

- Preserve all current `ver_1.16.0` model/features, including the known-deck
  token, setup summaries, Active Energy values, summary dimensions `86/84`,
  and cache schema `19`.
- Preserve optimizer, warmup/cosine schedule, mixed precision, EMA metrics,
  checkpoint/resume, epoch/step saves, W&B training metrics, and expert-loser
  reporting.
- Do not change replay extraction, feature caching, standalone validation, or
  the Kaggle submission notebook.

## Configuration

Remove from `train.yaml`: `eval_every_steps`, `validation_ratio`,
`validation_seed`, `expert_validation_ratio`, `isolation_validation`, and
`top_decks`. Retain `loser_augmentation`, `train_replay_ratio`, and
`train_replay_seed`.

## Verification

- Configuration loading rejects no missing validation-only fields.
- Full-data selection includes all normal samples and adds only configured
  expert-loser samples.
- `train.py` has no periodic validation path or validation-only imports.
- Current model/cache signature tests continue to pass.

