# Post-Competition Repository Reorganization Design

## Purpose

Reorganize the imitation-learning project after the competition so that its
directory structure, runnable commands, configuration files, notebooks, and
documentation communicate their responsibilities directly. Preserve model
and training behavior while making the final repository easier to understand
and reproduce.

This is a clean-break migration. Old module names, commands, and configuration
filenames will not remain as compatibility aliases.

## Scope

The reorganization will:

- group all replay and deck extraction entrypoints under `extraction/`;
- group reusable analysis logic under `analysis/`;
- place research notebooks under `notebooks/`;
- separate deck exploration from validation-deck selection;
- move isolation-validation logic into the validation domain;
- replace ambiguous configuration filenames with purpose-specific names;
- update all imports, fixed configuration paths, tests, and documentation;
- retain the current locations of generated isolation-selection CSV files.

The reorganization will not:

- change model architecture or feature semantics;
- split `training/train.py` into additional internal modules;
- change checkpoint payloads or resume behavior;
- add command-line configuration flags;
- move caches, checkpoints, generated datasets, or training outputs;
- preserve old import paths or executable module names.

## Target Structure

```text
imitation_learning/
|-- analysis/
|   |-- __init__.py
|   |-- deck_selection.py
|   |-- deck_statistics.py
|   `-- deck_trends.py
|-- cfg/
|   |-- analyze_deck_trends.yaml
|   |-- build_feature_cache.yaml
|   |-- extract_deck_lists.yaml
|   |-- extract_training_samples.yaml
|   |-- select_validation_decks.yaml
|   |-- train_policy.yaml
|   `-- validate_policy.yaml
|-- extraction/
|   |-- __init__.py
|   |-- deck_lists.py
|   |-- deck_trend_data.py
|   `-- training_samples.py
|-- model/
|-- notebooks/
|   |-- deck_eda.ipynb
|   |-- deck_trends.ipynb
|   |-- replay_timing.ipynb
|   `-- select_validation_decks.ipynb
|-- training/
|   |-- __init__.py
|   |-- build_feature_cache.py
|   |-- expert_replays.py
|   |-- export_inference.py
|   |-- feature_cache.py
|   |-- precision.py
|   `-- train.py
|-- validation/
|   |-- __init__.py
|   |-- config.py
|   |-- evaluate.py
|   |-- isolation.py
|   `-- metrics.py
|-- tests/
|-- kaggle_submission_imitation_agent.ipynb
|-- README.md
`-- requirements.txt
```

No general-purpose `utils/` directory will be introduced. Modules that are not
directly executable remain in the domain that owns their behavior.

## File Migration

### Extraction

| Current path | Target path | Responsibility |
| --- | --- | --- |
| `training/extract.py` | `extraction/training_samples.py` | Extract policy-training samples from replay archives. |
| `deck/extract.py` | `extraction/deck_lists.py` | Extract both players' deck lists from replay archives. |
| `deck/trend_extract.py` | `extraction/deck_trend_data.py` | Prepare source data used by deck-trend analysis. |

The resulting commands are:

```bash
python -m extraction.training_samples
python -m extraction.deck_lists
python -m extraction.deck_trend_data
```

### Training

| Current path | Target path |
| --- | --- |
| `training/cache_features.py` | `training/build_feature_cache.py` |
| `training/expert_validation.py` | `training/expert_replays.py` |

`training/train.py`, `training/feature_cache.py`, `training/precision.py`, and
`training/export_inference.py` retain their locations and responsibilities.
In particular, `feature_cache.py` remains a training-domain library even
though it is called indirectly.

### Validation

| Current path | Target path |
| --- | --- |
| `training/isolation_validation.py` | `validation/isolation.py` |

The independent evaluator remains `validation/evaluate.py`. Holdout training
and standalone validation continue to obtain validation ratios, seeds,
isolation selections, and top-deck definitions from the training YAML so they
construct the same validation sets.

An isolation-selection path may remain a string or be `null`. A null entry
disables that validation group; all-null selections disable the isolation
union without disabling latest-date or in-distribution validation.

### Analysis

| Current path or source | Target path | Responsibility |
| --- | --- | --- |
| `deck/analysis.py` | `analysis/deck_statistics.py` | Deck census, similarity, and descriptive statistics. |
| `deck/trend.py` | `analysis/deck_trends.py` | Reusable deck-trend calculations and plotting inputs. |
| Selection functions currently embedded in `deck/deck_eda.ipynb` | `analysis/deck_selection.py` | Deterministic isolation-deck candidate construction, rolling, auditing, and CSV-ready results. |

`analysis/deck_selection.py` is the single implementation used by the
validation-selection notebook. It must not contain notebook-specific display
state.

### Notebooks

| Current path | Target path | Responsibility |
| --- | --- | --- |
| `deck/deck_eda.ipynb` | `notebooks/deck_eda.ipynb` | Deck census, similarity, visual exploration, and conclusions only. |
| Selection portion of `deck/deck_eda.ipynb` | `notebooks/select_validation_decks.ipynb` | Candidate inspection, selected-deck display, audit, and CSV export. |
| `eda/deck_trend.ipynb` | `notebooks/deck_trends.ipynb` | Deck usage, matchup, and team-switching trends. |
| `eda/replay_timing.ipynb` | `notebooks/replay_timing.ipynb` | Replay timing analysis. |

The validation-selection notebook continues to display candidate and selected
deck tables interactively. It exports the same files to the same locations:

```text
data/deck_isolation_selection.csv
data/archetype_isolation_selection.csv
data/top_deck_archetype_isolation_selection.csv
```

## Configuration Migration

Each executable continues to read a fixed default YAML. No `--config` option
will be added.

| Current filename | Target filename | Consumer |
| --- | --- | --- |
| `cfg/extract.yaml` | `cfg/extract_training_samples.yaml` | `extraction.training_samples` |
| `cfg/deck_extract.yaml` | `cfg/extract_deck_lists.yaml` | `extraction.deck_lists` |
| `cfg/deck_trend.yaml` | `cfg/analyze_deck_trends.yaml` | Deck-trend extraction and analysis |
| `cfg/cache.yaml` | `cfg/build_feature_cache.yaml` | `training.build_feature_cache` |
| `cfg/train.yaml` | `cfg/train_policy.yaml` | `training.train` and validation definitions |
| `cfg/validation.yaml` | `cfg/validate_policy.yaml` | `validation.evaluate` |
| Notebook-local selection parameters | `cfg/select_validation_decks.yaml` | `notebooks/select_validation_decks.ipynb` |

`cfg/analyze_deck_trends.yaml` retains both its extraction and analysis
sections so that one workflow does not require multiple small configuration
files. `cfg/validate_policy.yaml` continues to point to
`cfg/train_policy.yaml`.

## Training and Checkpoint Stability

`training/train.py` will not be decomposed during this migration. Its current
holdout/full-data behavior remains unchanged:

- `holdout` uses isolation, latest-date, and in-distribution validation and may
  evaluate during training;
- `full_data` trains on all eligible dates and skips training-time validation.

Checkpoint payloads and resume behavior remain unchanged. Resume continues to
accept completed `epoch-*.pt` checkpoints only. Supporting exact mid-epoch
step resume is outside this reorganization.

## Documentation and User Flow

The README will start with the canonical pipeline:

```bash
python -m extraction.deck_lists
python -m extraction.training_samples
python -m training.build_feature_cache
python -m training.train
python -m validation.evaluate
```

It will separately document the notebooks:

```bash
jupyter notebook notebooks/deck_eda.ipynb
jupyter notebook notebooks/select_validation_decks.ipynb
jupyter notebook notebooks/deck_trends.ipynb
jupyter notebook notebooks/replay_timing.ipynb
```

For every command or notebook, the README will identify its fixed YAML,
principal inputs, outputs, and place in the reproduction sequence. It will
also explain holdout versus full-data training and the generation of the three
isolation-selection CSV files.

## Migration Method

1. Record the pre-migration state with a Git commit or annotated tag.
2. Use `git mv` for tracked files so history remains discoverable.
3. Move Python modules and update imports before renaming configurations.
4. Rename YAML files and update every fixed configuration path and reference.
5. Extract reusable validation-deck selection logic from the existing
   notebook into `analysis/deck_selection.py`.
6. Produce the focused deck EDA and validation-selection notebooks.
7. Update tests and README references.
8. Search the tracked repository for all obsolete module and configuration
   names.
9. Run focused tests after each migration stage, then the full suite and
   notebook validation.

The migration will not modify or delete cache data, model checkpoints,
generated output directories, or isolation-selection CSV files.

## Verification

Verification must cover:

- every canonical `python -m` entrypoint imports successfully;
- each entrypoint resolves its renamed fixed YAML;
- all internal imports use the new packages;
- extraction and analysis unit tests pass under their new module paths;
- holdout and full-data selection tests still pass;
- standalone validation constructs the same holdout definitions from
  `cfg/train_policy.yaml`;
- null isolation selections still work;
- both split notebooks are valid notebook documents;
- the validation-selection notebook contains display and export steps;
- the deck EDA notebook contains no validation CSV export responsibility;
- generated CSV destinations remain unchanged;
- repository-wide search finds no unintended references to old paths;
- `git diff --check` passes for the migration changes.

Any unrelated pre-existing worktree changes will be preserved and excluded
from migration commits.
