# Pokémon TCG AI Battle Challenge Simulation

My solution for [The Pokémon Company – PTCG AI Battle Challenge Simulation](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/) on Kaggle. **Final rank: 59 / 6,807 teams**, as recorded in my [competition notes](https://app.notion.com/p/39f888c4bb6180d1a714f6a930465490).

**TL;DR.** I extracted decisions from game replays and trained an encoder-decoder Transformer to score legal actions by behavior cloning. Replay-level, later-date, expert-subset, and deck/archetype holdouts guided development. The final agent averages the action probabilities of two full-data models with **3+3** and **4+4** Transformer layers, then selects an eligible action. The full account, with experiment curves and limitations, is in [kaggle_writeup.md](kaggle_writeup.md).

---

### Environment setup

The training code lives in `imitation_learning/` and requires Python 3.10+ and PyTorch. A CUDA GPU is recommended for full training. Install a PyTorch build appropriate for your GPU, then install the project requirements:

```bash
conda create -n orbit_wars python=3.11
conda activate orbit_wars
cd imitation_learning
python -m pip install -r requirements.txt
```

The commands below run from `imitation_learning/`. Check `cfg/train_policy.yaml` before training: its default CUDA device, BF16 precision, batch size, data paths, and W&B setting may need to be changed for your machine. See the [detailed quick start](imitation_learning/README.md).

---

### Data and model files

The training pipeline expects dated replay ZIPs under repository-level `replay_episodes/` and the competition `cg` package under `pokemon_tcg_ai_battle/sample_submission/`. Deck analysis also uses `pokemon_tcg_ai_battle/EN_Card_Data.csv`. The default layout and required replay manifests are described in [data-pipeline.md](docs/data-pipeline.md).

Generated extraction shards, feature caches, and model files are not a substitute for the source replays. The final submission uses `final_d2_33.pt` from the local `archive (3).zip` and `final_d2_44.pt` from `archive (2).zip`; these are separate submission assets, not files required for ordinary code inspection. For a Kaggle run, attach the checkpoint and `cg` datasets at the paths configured in [submission.ipynb](submission.ipynb).

---

### Build replay features and train

Review the YAML paths in `imitation_learning/cfg/`, then run from `imitation_learning/`:

```bash
python -m extraction.deck_lists
python -m extraction.training_samples
python -m training.build_feature_cache
python training/select_validation_decks.py
python -m training.train
```

The extractors create deck and decision records; cache construction prepares model features. The selection step builds deck and archetype isolation sets and should be reviewed before holdout training. `train.data_selection_mode: holdout` keeps validation groups out of training; `full_data` makes all cached dates eligible for final training and skips training-time validation. Winner filtering, replay sampling, and qualified loser augmentation still apply in full-data mode. Training settings are read from `cfg/train_policy.yaml`, not command-line flags. See [training and validation](docs/training-validation.md) for split order, thresholds, precision, logging, and checkpoint behavior.

---

### Validate and submit

Configure `cfg/validate_policy.yaml` with the intended checkpoint and holdout training config, then run:

```bash
python -m validation.evaluate
```

This prints offline action CE and Top-k accuracy. It can evaluate a compatible multi-checkpoint probability ensemble. A checkpoint trained with `full_data` may have seen reconstructed holdout samples; such an evaluation is **not** an independent generalization result.

The actual final artifact is [submission.ipynb](submission.ipynb). Its parameter cell names the two checkpoint paths, the `cg` package, the 60-card Festival Lead deck, and the enabled ensemble and damage-counter KO mask. Running it with those Kaggle datasets attached creates `submission.tar.gz`. The repository also contains [kaggle_submission_imitation_agent.ipynb](imitation_learning/kaggle_submission_imitation_agent.ipynb) as a configurable submission template. [Submission documentation](docs/submission.md) covers checkpoint export and packaging.

---

### Experiments and results

The [write-up](kaggle_writeup.md) follows the competition reasoning from literature and replay EDA through data selection, validation, model design, experiments, and the submitted agent. Three [model diagrams](docs/figures/README.md) explain state tokens, action queries, and the two-model ensemble; six EDA figures cover deck coverage, changing archetypes, matchups, and replay timing. The [experiment report](docs/solution-experiment-results.md) provides W&B run links, exact comparison values, and six figure groups for data scale, model scale, action history, activation/dropout, multitask learning, and effect/context-card features. Each experiment figure shows **training CE, in-distribution CE, latest-date CE, and latest-date expert CE**. PNG and SVG versions are indexed in [docs/figures](docs/figures/README.md).

The strongest recovered history comparison was full three-action history versus no history: at their last common evaluation, latest-date CE was lower by **0.0328**. Data scaling exposed a clear failure mode: a Small model trained on 1% of replays reached low training CE while its latest-date CE rose. Other comparisons are mixed or small; the write-up reports their limits. **Offline action imitation CE is not battle win rate.** The two final full-data checkpoints do not have a verified like-for-like single-model-versus-ensemble win-rate result in this repository.

---

### Repository contents

| Path | Purpose |
|---|---|
| [kaggle_writeup.md](kaggle_writeup.md) | Full solution and experiment narrative |
| [submission.ipynb](submission.ipynb) | Final two-model Kaggle submission notebook |
| [imitation_learning/](imitation_learning/) | Extraction, feature caching, training, validation, analysis, and model code |
| [imitation_learning/cfg/](imitation_learning/cfg/) | YAML settings for the project workflows |
| [docs/](docs/README.md) | Technical guides, experiment report, and figures |
| [Experiment branches](docs/experiment-branches.md) | Kept branch snapshots and their representative W&B runs |
| [export_wandb_runs.py](export_wandb_runs.py) | Optional W&B run metadata export |

The [project documentation index](docs/README.md) is the starting point for implementation details. Branch names record development history; use the linked W&B run configuration and checkpoint when interpreting an experiment.
