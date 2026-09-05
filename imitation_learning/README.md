# PTCG Imitation Learning

A behavior-cloning project built from Pokémon TCG competition replays: extract observations and actions, train a Transformer policy, evaluate checkpoints, analyze data coverage, and package Kaggle inference submissions.

The policy predicts legal actions through supervised learning; it has no value head or reinforcement-learning training loop. Its encoder and policy architecture are adapted from the competition MCTS example notebook. The current implementation supports both **holdout training** and **full-data training**. This guide describes the current workflow, not historical development iterations.

## Layout and documentation

```text
imitation_learning/
├─ cfg/          Extraction, training, validation, and EDA configurations
├─ extraction/   Replay extraction entrypoints
├─ training/     Cache construction, validation selection, training, and export
├─ validation/   Standalone validation and ensemble evaluation
├─ eda/          EDA scripts and the Deck EDA notebook
├─ analysis/     Reusable statistics and deck-selection logic
├─ model/        Feature encoding and policy architecture
├─ data/         Generated datasets, caches, and selection CSVs
└─ outputs/      Analysis results and training output when WandB is disabled
```

Detailed guides live in the repository-level [docs](../docs/README.md) directory: [data preparation](../docs/data-pipeline.md), [training and validation](../docs/training-validation.md), [model architecture](../docs/model.md), [EDA](../docs/eda.md), and [inference submission](../docs/submission.md).

## 1. Prepare the environment and data

Python 3.10+ is required. A CUDA GPU is recommended for full training; CPU execution is useful for small checks. Runtime and memory requirements depend on model size and batch size. Use a PyTorch installation compatible with your local CUDA environment.

Activate your existing `orbit_wars` environment, or create one:

```bash
conda create -n orbit_wars python=3.11
conda activate orbit_wars
cd imitation_learning
python -m pip install -r requirements.txt
```

**Run all commands below from `imitation_learning/`.** Relative data paths in the YAML configurations generally resolve from this directory as well.

Prepare the competition resources using the default layout:

```text
repository/
├─ replay_episodes/                # Dated replay ZIP archives; supply separately
├─ pokemon_tcg_ai_battle/
│  ├─ EN_Card_Data.csv             # Card table for deck analysis
│  └─ sample_submission/cg/        # Competition engine Python package
├─ imitation_learning/
└─ docs/
```

These commands do not download replays, the competition engine, or pretrained checkpoints. Each dated archive should contain its replays; expert filtering also requires one `manifest.csv` per ZIP. See [data preparation](../docs/data-pipeline.md).

Review these configurations before running; the defaults may not suit your machine:

| Configuration | Check first |
|---|---|
| `cfg/extract_deck_lists.yaml` | Replay input, deck output, workers |
| `cfg/extract_training_samples.yaml` | Replay input, sample output, workers |
| `cfg/build_feature_cache.yaml` | cg_path, sample input, cache output, workers |
| `cfg/train_policy.yaml` | cg_path, data, replay_episodes, device, precision, batch_size, wandb |

Start with fewer workers and a smaller batch size. Do not retain `precision: bf16` without compatible hardware. For CPU checks, use `device: cpu` and `precision: fp32`. Set `wandb.enabled: false` if you do not want WandB logging.

## 2. Extract data and build the cache

Run in order:

```bash
python -m extraction.deck_lists
python -m extraction.training_samples
python -m training.build_feature_cache
```

These produce `data/deck/`, `data/training/`, and `data/training_cache/`, respectively. Large datasets require substantial time and disk space. Extraction and cache construction reuse completed, compatible shards.

For an extraction smoke test, set a small `limit_members` in both extraction YAML files. Use separate output directories for test data. Restore `null` for full extraction, and do not mistake a small test cache for the complete dataset. See [data preparation](../docs/data-pipeline.md).

## 3. Choose a training mode and train

Edit the existing fields in `cfg/train_policy.yaml`:

- `data_selection_mode: holdout`: reserves isolation, latest-date, and in-distribution validation sets; use it to compare models and settings.
- `data_selection_mode: full_data`: includes all cache dates in training selection and skips training-time validation; use it for final training after selecting a configuration. Result filtering, loser augmentation, and replay sampling still apply.

### Holdout: prepare validation selections first

Configure `cfg/select_validation_decks.yaml`, then run:

```bash
python training/select_validation_decks.py
```

Review `outputs/select_validation_decks/selection_report.html`. The three selection CSVs under `data/` are updated only after selection and the combined audit succeed.

To run without isolation validation, set all three paths to `null` in the training YAML. This does not disable latest-date or in-distribution validation:

```yaml
  isolation_validation:
    deck_data: data/deck
    selections:
      deck_isolation: null
      archetype_isolation: null
      top_deck_archetype_isolation: null
```

`train.top_decks` contains named, complete 60-card lists for subgroup metrics. **All four validation subsets for each configured deck must be nonempty.** With limited data, remove unsuitable top-deck entries instead of treating an empty subset as a valid evaluation.

### Start training

```bash
python -m training.train
```

The training entrypoint reads YAML rather than command-line hyperparameters. Start with a small model, a small batch, and a bounded `max_samples` trial. Keep `warmup_steps` below the total number of optimizer steps. For small datasets, disable loser augmentation or reduce its `recent_dates` to the available date range.

With WandB disabled, checkpoints go to `train.output`. With WandB enabled, they go to the local run's `local-output/` directory. Resume supports complete epoch checkpoints only, not step or inference-only checkpoints.

See [training and validation](../docs/training-validation.md) for the full rules and troubleshooting.

## 4. Run standalone validation

Edit `cfg/validate_policy.yaml`:

- Point `validation.train_config` to the training YAML that defines the validation sets.
- Fill `validation.ensemble.checkpoints` with actual checkpoint paths; the repository default list is not populated.
- For one model, set `enabled: false` and provide one path. For an ensemble, set `true` and provide at least two distinct paths.

```bash
python -m validation.evaluate
```

The command prints CE loss and Top-1/3/5 accuracy. Standalone validation reconstructs holdout sets. Evaluating a full-data model on data it already trained on is not an independent generalization test.

## 5. Analyze the data (optional)

| Analysis | Workflow | Configuration / output |
|---|---|---|
| Deck coverage | Open `eda/deck_eda.ipynb` and run all cells | Reuses `cfg/train_policy.yaml`; writes `outputs/deck_eda/` |
| Deck trends | Run `python -m extraction.deck_trend_data`, then `python eda/deck_trends_eda.py` | Separate extraction and EDA YAML files; writes `outputs/deck_trends/` |
| Replay timing | Run `python -m extraction.replay_timing`, then `python eda/replay_timing_eda.py` | Separate extraction and EDA YAML files; writes `outputs/replay_timing/` |

Deck EDA needs deck CSVs, the training cache, replay manifests, and any selection CSVs enabled by the current holdout configuration. Trends and timing analysis do not require a trained model. Output filenames are fixed; rerunning replaces the corresponding results. See [EDA](../docs/eda.md) for metric definitions.

## 6. Export and submit

Edit `CHECKPOINT_PATH`, `OUTPUT_PATH`, and `PRECISION` at the top of `training/export_inference.py`, then run:

```bash
python training/export_inference.py
```

Attach the inference checkpoint and competition `cg` package as Kaggle Datasets to the [submission notebook](kaggle_submission_imitation_agent.ipynb). Set the model path, engine path, and 60-card deck, then run it to generate `submission.tar.gz`. See [inference submission](../docs/submission.md) for details and limitations.

## Interpreting results

Validation action accuracy is not the same as game win rate, and data-volume statistics alone cannot establish a model bottleneck. This guide does not claim a final competition score or ablation outcome; interpret results using the corresponding checkpoint, configuration, and experiment records.
