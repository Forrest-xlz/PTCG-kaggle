# Training and Validation

## Configuration and entrypoint

`python -m training.train` reads `cfg/train_policy.yaml`; hyperparameters are not CLI arguments. The `train` section controls data, training, and validation; `model` controls architecture; `wandb` controls logging. The root `version_name` can be interpolated as `${version_name}` in output paths and run names.

## Training modes

| Behavior | holdout | full_data |
|---|---|---|
| Dates | Latest date reserved for validation | All cache dates eligible for training selection |
| Isolation and in-distribution validation | Corresponding replays excluded | No such exclusions |
| Training-time evaluation | Enabled | Skipped |
| Loser augmentation dates | Exclude latest date | May include latest date |
| Model, optimizer, and checkpoint logic | Shared | Shared |

Full-data mode does not unconditionally use every action from both players. Result filtering, loser augmentation, and `train_replay_ratio` still apply. A full-data checkpoint evaluated on reconstructed holdout sets may have seen those samples during training; this is not unseen-data generalization performance.

## Holdout split order

1. Match either player's complete deck against enabled isolation CSVs and assign the entire replay to isolation validation.
2. Reserve the numerically latest month.day from the remaining data for latest-date validation.
3. Deterministically assign older remaining replays to training or in-distribution validation using `validation_ratio` and `validation_seed`.
4. Sample candidate training replays using `train_replay_ratio` and `train_replay_seed`, selecting actions according to result and loser-augmentation rules.

A replay cannot cross training and validation splits. Isolation groups may share replays; their union is removed before subsequent splits. Validation uses winner actions. Training uses winner actions and optionally eligible loser actions. Draws are excluded from this winner/loser training policy.

Replay sampling ratios need not equal action-sample ratios because games have different decision counts. The fixed training pool is reused across epochs; shuffling changes its order. `max_samples` limits per-epoch exposure, not the underlying training pool.

## Isolation selection

Run `python training/select_validation_decks.py`, configured by `cfg/select_validation_decks.yaml`.

- Deck isolation holds out selected exact decks to assess unseen deck combinations.
- Archetype isolation holds out selected archetypes to assess generalization to those groups.
- Top-deck archetype isolation selects other variants within a configured top deck's archetype; configured top decks themselves are excluded from variant selection.

The third group supports multiple named top decks, including entries sharing an archetype. Each entry specifies card_ids, archetype, candidate and total-replay constraints, and roll_id. Archetypes belonging to configured top decks or selected ordinary deck-isolation decks are excluded from archetype-isolation candidates.

Default outputs:

- `data/deck_isolation_selection.csv`
- `data/archetype_isolation_selection.csv`
- `data/top_deck_archetype_isolation_selection.csv`
- `outputs/select_validation_decks/selection_report.html`
- Candidate tables and `combined_audit.csv` under the report directory's `tables/`

The combined audit checks constraints, including card coverage, after all selected replays are removed together. A failure preserves the previous three selection CSVs. Review the HTML report and audit, then adjust roll_id or constraints instead of bypassing the audit.

Each selection path in the training YAML can be `null`. Disabling all isolation groups still retains latest-date and in-distribution validation. Selection CSVs are matched at training startup; no cache rebuild is needed.

## Expert episodes versus loser augmentation

`expert_validation_ratio` sets a daily high-score cutoff across all participants. A replay is expert if either participant reaches that cutoff. Ties are retained, so the expert-replay fraction need not equal the configured ratio, and both participants are not necessarily experts.

`loser_augmentation.expert_ratio` also defines a daily participant-score cutoff, but the replay's min_score must meet it. Both participants must qualify before loser actions become eligible.

`loser_augmentation.recent_dates` selects the newest eligible training dates. Holdout excludes the latest date; full_data does not. Requesting more dates than available raises an error. Expert validation and loser augmentation use independent ratios.

Startup fields `expert_date / cutoff / episodes / expert_episodes` report the date, expert threshold, total episodes, and expert episodes. They do not report validation counts for each top deck.

## Named top decks and metrics

Each `train.top_decks` entry contains `name` and a complete 60-card `card_ids` list. Order is ignored, but multiplicity must match. Names do not define archetype classification rules. Use consistent names for the same deck in training and selection YAMLs.

Each deck receives in-distribution, in-distribution expert, latest-date, and latest-date expert metrics, such as `val_latest_deck(grimmsnarl)`. Top-deck isolation also retains aggregate and named subgroup metrics. CE loss and Top-1/3/5 accuracy reuse the parent batch's logits.

Currently, configured top-deck validation subsets must be nonempty. The list following `configured top-deck validation subsets are empty` identifies missing groups. Check decks, dates, expert thresholds, isolation, and split settings. Nonzero daily expert_episodes does not guarantee expert validation samples for a particular deck.

## Optimizer, precision, and logging

AdamW supports learning_rate, weight_decay, beta1, and beta2. Learning rate uses linear warmup followed by cosine decay. `warmup_steps` must be smaller than the total optimizer-step count.

`log_every_steps`, `eval_every_steps`, and `save_every_steps` count successful optimizer updates. Training logs use a cross-epoch EMA controlled by `ema_alpha`.

- fp32: full precision.
- fp16: autocast with gradient scaling.
- bf16: autocast without a scaler; CUDA execution requires compatible hardware.
- Model parameters and training checkpoints remain FP32; export precision is configured separately.

With WandB enabled, checkpoints and history go to local-output beside the run's files directory, without automatically uploading model artifacts. Otherwise, train.output is used. Review the default online setting; disable WandB if no account is configured.

## Resume

Edit the existing fields in the training YAML:

```yaml
  resume: true
  resume_checkpoint: outputs/my_run/checkpoints/epoch-002.pt
  epochs: 5
```

This continues after completed epoch 2 through epoch 5; epochs is the final total. Resume restores the model, optimizer, scheduler, FP16 scaler, EMA, history, and optimizer step. New checkpoints also preserve RNG state. Older checkpoints without RNG state do not guarantee bit-for-bit identical random sequences.

Only epoch training checkpoints support resume. Step and inference-only checkpoints do not. Load only trusted checkpoints.

## Standalone validation and ensembles

`python -m validation.evaluate` reads `cfg/validate_policy.yaml`. Its `validation.train_config` reuses the training YAML's cache, dates, ratios, seed, isolation, and top decks. Architecture is reconstructed from checkpoint config, not duplicated in the validation YAML.

Example `validation.ensemble` configuration; replace the path with an existing file:

```yaml
  ensemble:
    enabled: false
    checkpoints:
      - outputs/my_run/checkpoints/epoch-005.pt
```

For an ensemble, enable it and provide at least two distinct checkpoints. Epoch or inference checkpoints containing model/config are accepted. Architectures may differ, but feature signatures and action spaces must match. Each model masks invalid actions, computes FP32 softmax probabilities, and contributes equally to the averaged legal-action probabilities used for metrics. All models remain on the device, increasing memory use.

Results are printed to the terminal; the evaluator does not create remote experiments or result files.

## Troubleshooting

- Missing selection CSV: run the selector or disable that isolation group.
- Empty top-deck subset: inspect the named group in the error, then check data and configuration.
- recent_dates exceeds available training dates: reduce it or disable loser augmentation.
- warmup_steps exceeds the total step count: reduce warmup or increase trial size.
- Incompatible cache: rebuild as described in the [data pipeline](data-pipeline.md); do not mix old schemas.
- Missing CUDA/BF16 support: adjust device, precision, and batch_size.
