# Post-Competition Repository Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the imitation-learning project into explicit extraction, analysis, training, validation, notebook, and configuration domains without changing training, validation, checkpoint, or generated-data behavior.

**Architecture:** Perform a clean-break migration with history-preserving file moves, then update imports and fixed YAML paths in small testable stages. Separate reusable validation-deck selection functions from general deck statistics, while retaining notebook display/export behavior and the existing three CSV destinations.

**Tech Stack:** Python 3.10+, pytest, PyYAML, pandas, Jupyter Notebook JSON, Git

**Spec:** `docs/superpowers/specs/2026-09-03-post-competition-repository-reorganization-design.md`

## Global Constraints

- Work on the existing `end` feature branch or an explicitly approved isolated worktree; never implement on `main` or `master`.
- Preserve the user's pre-existing uncommitted change to `imitation_learning/cfg/train.yaml`; its content must become `imitation_learning/cfg/train_policy.yaml` during the rename.
- Use a clean-break migration: do not retain old import paths, executable module names, notebook paths, or YAML filenames as aliases.
- Do not change model architecture, feature semantics, training selection behavior, checkpoint payloads, or epoch-only resume behavior.
- Do not split `training/train.py` or introduce a general-purpose `utils/` package.
- Do not add a `--config` command-line option; executable modules continue to use fixed YAML paths.
- Do not move or delete caches, checkpoints, generated datasets, training outputs, or the three isolation-selection CSV files.
- Use the `orbit_wars` interpreter at `C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe` for validation.
- Because this Windows/Torch environment may hang after pytest reports 100%, preserve the complete pytest output and terminate only after the result line is visible.

---

### Task 1: Establish migration contract tests

**Files:**
- Create: `imitation_learning/tests/test_repository_layout.py`
- Modify: `imitation_learning/tests/test_artifacts.py`

**Interfaces:**
- Consumes: the target tree and canonical commands from the design spec.
- Produces: tests that reject obsolete paths/references and validate the two focused notebooks' responsibilities.

- [ ] **Step 1: Add a failing target-layout test**

Create assertions that require the new packages, modules, YAML files, and notebooks and reject the old files. The test must define explicit tuples rather than deriving expectations from the filesystem:

```python
EXPECTED = (
    "analysis/deck_selection.py",
    "analysis/deck_statistics.py",
    "analysis/deck_trends.py",
    "extraction/deck_lists.py",
    "extraction/deck_trend_data.py",
    "extraction/training_samples.py",
    "training/build_feature_cache.py",
    "training/expert_replays.py",
    "validation/isolation.py",
    "cfg/analyze_deck_trends.yaml",
    "cfg/build_feature_cache.yaml",
    "cfg/extract_deck_lists.yaml",
    "cfg/extract_training_samples.yaml",
    "cfg/select_validation_decks.yaml",
    "cfg/train_policy.yaml",
    "cfg/validate_policy.yaml",
    "notebooks/deck_eda.ipynb",
    "notebooks/deck_trends.ipynb",
    "notebooks/replay_timing.ipynb",
    "notebooks/select_validation_decks.ipynb",
)

OBSOLETE = (
    "deck/analysis.py",
    "deck/extract.py",
    "deck/trend.py",
    "deck/trend_extract.py",
    "training/extract.py",
    "training/cache_features.py",
    "training/expert_validation.py",
    "training/isolation_validation.py",
    "cfg/extract.yaml",
    "cfg/deck_extract.yaml",
    "cfg/deck_trend.yaml",
    "cfg/cache.yaml",
    "cfg/train.yaml",
    "cfg/validation.yaml",
)
```

Also load both new notebooks with `json.loads`, assert `nbformat == 4`, assert the selection notebook contains all three CSV names plus `display(`, and assert the deck EDA notebook contains none of those CSV names.

- [ ] **Step 2: Run the contract test and verify RED**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest tests/test_repository_layout.py -q
```

Expected: FAIL because the target files do not exist and old paths remain.

- [ ] **Step 3: Update artifact expectations to describe the final notebook split**

In `test_artifacts.py`, replace the old `deck/deck_eda.ipynb` path with the two new notebook paths. Keep submission-notebook checks untouched. Require selection/export symbols only in `select_validation_decks.ipynb`, and descriptive census/similarity symbols only in `deck_eda.ipynb`.

- [ ] **Step 4: Record the RED state without changing production files**

Run the two focused files and confirm failures are exclusively missing target paths or still-present obsolete paths:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest tests/test_repository_layout.py tests/test_artifacts.py -q
```

Expected: FAIL for the pre-migration layout.

### Task 2: Move extraction entrypoints and rename their configurations

**Files:**
- Create via move: `imitation_learning/extraction/__init__.py`
- Move: `imitation_learning/deck/extract.py` -> `imitation_learning/extraction/deck_lists.py`
- Move: `imitation_learning/deck/trend_extract.py` -> `imitation_learning/extraction/deck_trend_data.py`
- Move: `imitation_learning/training/extract.py` -> `imitation_learning/extraction/training_samples.py`
- Move: `imitation_learning/cfg/deck_extract.yaml` -> `imitation_learning/cfg/extract_deck_lists.yaml`
- Move: `imitation_learning/cfg/extract.yaml` -> `imitation_learning/cfg/extract_training_samples.yaml`
- Modify: `imitation_learning/tests/test_deck_extract.py`
- Modify: `imitation_learning/tests/test_deck_trend_extract.py`
- Modify: `imitation_learning/tests/test_extract_alignment.py`

**Interfaces:**
- Consumes: `extract_decks(payload: dict[str, Any]) -> list[list[int]]` from `extraction.deck_lists`.
- Produces: runnable modules `extraction.deck_lists`, `extraction.deck_trend_data`, and `extraction.training_samples` with unchanged data schemas and outputs.

- [ ] **Step 1: Change extraction tests to the desired imports**

Replace imports with:

```python
from extraction.deck_lists import load_settings
from extraction.deck_trend_data import extract_team_names, process_archive
from extraction.training_samples import _iter_player_records, _player_results
```

Preserve all existing behavioral assertions.

- [ ] **Step 2: Run extraction tests and verify RED**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest tests/test_deck_extract.py tests/test_deck_trend_extract.py tests/test_extract_alignment.py -q
```

Expected: collection errors because `extraction` does not yet exist.

- [ ] **Step 3: Move the modules and update fixed paths/imports**

Use `git mv` for tracked files. Add an empty `extraction/__init__.py`. Set:

```python
# extraction/deck_lists.py
CONFIG_PATH = PROJECT_ROOT / "cfg" / "extract_deck_lists.yaml"

# extraction/training_samples.py
from extraction.deck_lists import extract_decks
CONFIG_PATH = PROJECT_ROOT / "cfg" / "extract_training_samples.yaml"

# extraction/deck_trend_data.py
from extraction.deck_lists import extract_decks
CONFIG_PATH = PROJECT_ROOT / "cfg" / "analyze_deck_trends.yaml"
```

Update user-facing exception and rerun messages to name the new YAML and module names. Do not alter settings dataclasses, schema versions, archive iteration, or output formats.

- [ ] **Step 4: Run extraction tests and import smoke tests**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest tests/test_deck_extract.py tests/test_deck_trend_extract.py tests/test_extract_alignment.py -q
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -c "import extraction.deck_lists, extraction.deck_trend_data, extraction.training_samples"
```

Expected: all focused tests PASS and imports exit 0.

- [ ] **Step 5: Commit the extraction migration**

Stage only the extraction modules, their two renamed YAML files, and three focused tests. Commit message:

```text
refactor: group data extraction entrypoints
```

### Task 3: Separate deck statistics, selection, and trend analysis

**Files:**
- Move: `imitation_learning/deck/analysis.py` -> `imitation_learning/analysis/deck_statistics.py`
- Create: `imitation_learning/analysis/deck_selection.py`
- Move: `imitation_learning/deck/trend.py` -> `imitation_learning/analysis/deck_trends.py`
- Create: `imitation_learning/analysis/__init__.py`
- Modify: `imitation_learning/tests/test_deck_analysis.py`
- Modify: `imitation_learning/tests/test_deck_trend.py`

**Interfaces:**
- Produces from `analysis.deck_statistics`: `CardCatalog`, `ExactDeckIdentity`, `normalize_card_name`, `classify_deck`, `exact_deck_identity`, `validate_deck_rows`, `load_deck_rows`, `annotate_deck_facts`, `build_deck_census`, `changed_slots`, `weighted_jaccard`, and `build_similarity_pairs`.
- Produces from `analysis.deck_selection`: `SelectionRoll`, `build_deck_isolation_candidates`, `build_top_deck_archetype_candidates`, `archetype_core_card_ids`, `roll_deck_isolation_selection`, `roll_top_deck_archetype_selection`, `build_archetype_isolation_candidates`, `roll_archetype_isolation_selection`, `build_archetype_selection_details`, and `audit_deck_archetype_selection`.
- Produces from `analysis.deck_trends`: the existing public trend calculation and plotting functions with unchanged signatures.

- [ ] **Step 1: Change analysis tests to target the new package boundary**

Import census/similarity functions from `analysis.deck_statistics`, selection functions from `analysis.deck_selection`, and trend functions from `analysis.deck_trends`. Keep assertions unchanged so the tests enforce behavior preservation.

- [ ] **Step 2: Run analysis tests and verify RED**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest tests/test_deck_analysis.py tests/test_deck_trend.py -q
```

Expected: collection errors because `analysis` does not yet exist.

- [ ] **Step 3: Move statistics and trend modules**

Use `git mv`, add `analysis/__init__.py`, and change `analysis/deck_trends.py` to import shared classification/statistics symbols from `analysis.deck_statistics`.

- [ ] **Step 4: Extract selection code into its dedicated module**

Move `SelectionRoll` and the selection-only functions listed in Interfaces from `deck_statistics.py` into `deck_selection.py`. Move their private helpers only when used exclusively by selection. Import shared identities, annotated-fact validation, deck indexes, and archetype constants explicitly from `deck_statistics`; if a private helper is required across the boundary, rename it to a public, responsibility-revealing helper before importing it. Avoid wildcard imports and preserve every existing signature and deterministic random-seed behavior.

- [ ] **Step 5: Run analysis tests**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest tests/test_deck_analysis.py tests/test_deck_trend.py -q
```

Expected: all focused tests PASS.

- [ ] **Step 6: Commit the analysis separation**

Stage only `analysis/`, removal of the old analysis/trend modules, and the two focused tests. Commit message:

```text
refactor: separate deck analysis responsibilities
```

### Task 4: Move training helpers into their owning domains

**Files:**
- Move: `imitation_learning/training/cache_features.py` -> `imitation_learning/training/build_feature_cache.py`
- Move: `imitation_learning/training/expert_validation.py` -> `imitation_learning/training/expert_replays.py`
- Move: `imitation_learning/training/isolation_validation.py` -> `imitation_learning/validation/isolation.py`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/validation/evaluate.py`
- Modify affected tests importing the three old modules.

**Interfaces:**
- Produces: `training.build_feature_cache` as the cache-building entrypoint.
- Produces: unchanged expert replay APIs `load_expert_date_info`, `load_expert_loser_date_info`, `ExpertDateInfo`, and `ExpertLoserDateInfo` from `training.expert_replays`.
- Produces: unchanged `load_isolation_replay_sets(...)` from `validation.isolation`.

- [ ] **Step 1: Redirect tests to the desired modules**

Update imports in `test_cache_expert_labels.py`, `test_expert_signature.py`, `test_opponent_history_cache.py`, `test_feature_cache.py`, `test_expert_validation.py`, `test_training_metrics.py`, and `test_isolation_validation.py`. Add or update a test asserting:

```python
from training.build_feature_cache import CONFIG_PATH as cache_config
from validation.config import CONFIG_PATH as validation_config

assert cache_config.name == "build_feature_cache.yaml"
assert validation_config.name == "validate_policy.yaml"
```

- [ ] **Step 2: Run focused tests and verify RED**

Run the seven affected test modules plus `tests/test_validation_config.py`. Expected: collection or path assertion failures because moves have not happened.

- [ ] **Step 3: Move modules and update imports**

Use `git mv`, then change imports in `training/train.py` and `validation/evaluate.py`:

```python
from training.expert_replays import ...
from validation.isolation import load_isolation_replay_sets
```

Change the cache builder's fixed path to `cfg/build_feature_cache.yaml` and its error/rerun messages to `training.build_feature_cache` and `extraction.training_samples`. Do not alter cache schema or feature construction.

- [ ] **Step 4: Run focused training and validation tests**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest tests/test_cache_expert_labels.py tests/test_expert_signature.py tests/test_opponent_history_cache.py tests/test_feature_cache.py tests/test_expert_validation.py tests/test_training_metrics.py tests/test_isolation_validation.py tests/test_validation_config.py tests/test_validation_evaluate.py -q
```

Expected: migration-related tests PASS; record the known unrelated baseline failure in `test_validation_evaluate.py::test_feature_signature_matches_training_cache_contract` if it remains unchanged.

- [ ] **Step 5: Commit the domain moves**

Commit message:

```text
refactor: clarify training and validation modules
```

### Task 5: Rename remaining YAML files without losing local configuration

**Files:**
- Move: `imitation_learning/cfg/deck_trend.yaml` -> `imitation_learning/cfg/analyze_deck_trends.yaml`
- Move: `imitation_learning/cfg/cache.yaml` -> `imitation_learning/cfg/build_feature_cache.yaml`
- Move: `imitation_learning/cfg/train.yaml` -> `imitation_learning/cfg/train_policy.yaml`
- Move: `imitation_learning/cfg/validation.yaml` -> `imitation_learning/cfg/validate_policy.yaml`
- Create: `imitation_learning/cfg/select_validation_decks.yaml`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/validation/config.py`
- Modify config-referencing tests.

**Interfaces:**
- Produces: fixed training path `cfg/train_policy.yaml` shared by training and standalone validation definitions.
- Produces: fixed evaluator path `cfg/validate_policy.yaml`, whose `validation.train_config` equals `cfg/train_policy.yaml`.
- Produces: selection config sections for input data, exact-deck roll, archetype roll, top-deck roll, output CSV paths, and deterministic seeds copied from the current notebook constants.

- [ ] **Step 1: Add failing assertions for all fixed YAML paths**

Update config-path tests to require the seven target filenames. Add a YAML parse assertion that `validate_policy.yaml` points to `cfg/train_policy.yaml` and that the selection config outputs are exactly:

```yaml
deck_isolation: data/deck_isolation_selection.csv
archetype_isolation: data/archetype_isolation_selection.csv
top_deck_archetype_isolation: data/top_deck_archetype_isolation_selection.csv
```

- [ ] **Step 2: Run config tests and verify RED**

Run `test_repository_layout.py`, `test_validation_config.py`, `test_training_metrics.py`, and the extraction/cache config tests. Expected: missing target YAML failures.

- [ ] **Step 3: Rename configs and update consumers**

Use `git mv`. Preserve the working-tree content of `cfg/train.yaml` byte-for-byte under `cfg/train_policy.yaml` before making only necessary filename references. Set:

```python
# training/train.py
CONFIG_PATH = PROJECT_ROOT / "cfg" / "train_policy.yaml"

# validation/config.py
CONFIG_PATH = PROJECT_ROOT / "cfg" / "validate_policy.yaml"
```

Update `validate_policy.yaml` to use `train_config: cfg/train_policy.yaml`. Change exception messages accordingly.

- [ ] **Step 4: Create the selection YAML from existing notebook constants**

Transcribe every user-adjustable selection parameter currently embedded in the notebook: input deck glob/card catalog, training/latest date boundary, similarity bands, sample sizes, roll IDs/seeds, target top-deck definition, and output paths. Use descriptive nested section names and no undocumented placeholders.

- [ ] **Step 5: Verify YAML loading and behavior tests**

Run the focused config, holdout/full-data, standalone validation, and null-isolation tests. Expected: all migration-related tests PASS and the user's `data_selection_mode`/nullable selection content remains present in `train_policy.yaml`.

- [ ] **Step 6: Commit config migration separately**

Before staging, show `git diff -- imitation_learning/cfg/train_policy.yaml` and confirm changes are only the preserved user content plus intended path migration. Commit message:

```text
refactor: name workflow configurations explicitly
```

### Task 6: Split and relocate notebooks

**Files:**
- Move and edit: `imitation_learning/deck/deck_eda.ipynb` -> `imitation_learning/notebooks/deck_eda.ipynb`
- Create: `imitation_learning/notebooks/select_validation_decks.ipynb`
- Move and edit: `imitation_learning/eda/deck_trend.ipynb` -> `imitation_learning/notebooks/deck_trends.ipynb`
- Move: `imitation_learning/eda/replay_timing.ipynb` -> `imitation_learning/notebooks/replay_timing.ipynb`
- Modify: `imitation_learning/tests/test_artifacts.py`
- Modify: `imitation_learning/tests/test_deck_trend_notebook.py`

**Interfaces:**
- Consumes: `analysis.deck_statistics`, `analysis.deck_selection`, `analysis.deck_trends`, and `cfg/select_validation_decks.yaml`.
- Produces: one descriptive deck EDA notebook and one selection notebook that displays candidates/selections/audit and writes the same three CSV files.

- [ ] **Step 1: Strengthen notebook responsibility tests and verify RED**

Require both notebooks to discover the project root by checking `analysis/deck_statistics.py`. Require the selection notebook to load `cfg/select_validation_decks.yaml`, import from `analysis.deck_selection`, call `display` for candidates and selections, run `audit_deck_archetype_selection`, and export all three paths. Require deck EDA to import only descriptive APIs and contain no selection rolls or selection CSV writes.

- [ ] **Step 2: Build the selection notebook from existing cells**

Copy the setup/data-loading cells required by selection, then move the exact-deck, archetype, top-deck, display, export, and combined-audit cells into the new notebook in executable order. Replace notebook constants with values loaded from the selection YAML. Clear stale exception outputs and execution counts while preserving useful rendered charts/tables only where they remain valid.

- [ ] **Step 3: Reduce the deck EDA notebook to descriptive analysis**

Retain deck loading, classification, census, similarity, visual exploration, and conclusions. Remove selection candidate/roll/audit/export cells. Update imports to `analysis.deck_statistics` and root discovery to the target package.

- [ ] **Step 4: Move and update trend/timing notebooks**

Use `git mv`. In `deck_trends.ipynb`, change imports to `analysis.deck_statistics` and `analysis.deck_trends`, update root discovery, and reference `cfg/analyze_deck_trends.yaml`. Keep replay-timing computations unchanged apart from paths required by its new location.

- [ ] **Step 5: Validate notebook JSON and focused tests**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest tests/test_repository_layout.py tests/test_artifacts.py tests/test_deck_trend_notebook.py -q
```

Expected: all migration assertions PASS; if the unrelated Kaggle submission notebook validity assertion remains broken, record it without changing that notebook.

- [ ] **Step 6: Commit notebook organization**

Commit message:

```text
refactor: separate analysis and selection notebooks
```

### Task 7: Rewrite the README around the canonical workflow

**Files:**
- Modify: `imitation_learning/README.md`
- Modify: `imitation_learning/tests/test_repository_layout.py`

**Interfaces:**
- Consumes: every final command, fixed YAML filename, notebook responsibility, and output path.
- Produces: a reproducible post-competition guide with no obsolete commands or paths.

- [ ] **Step 1: Add a failing README contract test**

Assert the README contains these five commands in order:

```text
python -m extraction.deck_lists
python -m extraction.training_samples
python -m training.build_feature_cache
python -m training.train
python -m validation.evaluate
```

Assert it names all seven new YAML files and four notebook paths, explains `holdout` versus `full_data`, and contains none of the old command/config/notebook names.

- [ ] **Step 2: Run the README test and verify RED**

Run `pytest tests/test_repository_layout.py -q`. Expected: README assertions FAIL.

- [ ] **Step 3: Rewrite README layout and workflow sections**

Document, for each executable and notebook: its purpose, fixed YAML, primary inputs, outputs, and place in sequence. Preserve accurate technical documentation about schemas, validation subsets, precision, model architecture, and epoch-only resume, updating names only where the migration requires it.

- [ ] **Step 4: Run README and layout tests**

Expected: PASS.

- [ ] **Step 5: Commit documentation**

Commit message:

```text
docs: document the final project workflow
```

### Task 8: Repository-wide cleanup and verification

**Files:**
- Modify only files revealed by the obsolete-reference audit.
- Remove now-empty tracked package markers only when the directory has no remaining responsibility: `imitation_learning/deck/__init__.py` and/or old `eda/` contents.

**Interfaces:**
- Consumes: completed target tree.
- Produces: a clean repository with canonical imports, commands, and configuration references only.

- [ ] **Step 1: Search for obsolete references**

Run a tracked-source search excluding `.git`, caches, generated data, and outputs for old modules, configs, and notebook paths. Every hit must be classified as either an intentional historical statement in the design/plan or fixed in project code/docs/tests.

- [ ] **Step 2: Run syntax and import checks**

Run `compileall` over `analysis`, `extraction`, `model`, `training`, and `validation`. Import the five canonical entrypoint modules without invoking their `main()` functions. Expected: exit 0.

- [ ] **Step 3: Run focused migration suite**

Run all extraction, deck analysis/trend, cache, training-selection, validation-config, isolation, repository-layout, artifact, and notebook tests. Expected: all migration-related assertions PASS.

- [ ] **Step 4: Run the complete test suite**

Run:

```powershell
& 'C:\Users\Liuluotu\Anaconda\anaconda3\envs\orbit_wars\python.exe' -m pytest tests -q
```

Capture the final summary. Distinguish migration regressions from the two known unrelated baseline failures: `test_validation_evaluate.py::test_feature_signature_matches_training_cache_contract` and `test_artifacts.py::test_submission_notebook_is_valid_json`.

- [ ] **Step 5: Verify patch hygiene and preserved data**

Run `git diff --check`, list the final tree, confirm the three CSV destination strings are unchanged, and confirm no generated data/cache/output files are staged. Review `git status --short` and the cumulative diff.

- [ ] **Step 6: Commit final cleanup**

Stage only intentional migration files and commit:

```text
chore: complete repository reorganization
```

- [ ] **Step 7: Finish the development branch**

Invoke `superpowers:verification-before-completion`, then `superpowers:finishing-a-development-branch`. Report exact test results and known baseline failures, and let the user choose merge, push/PR, keep, or discard according to that skill.
