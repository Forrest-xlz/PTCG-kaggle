# Diverse Isolation Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed-count strict Card-isolation recommendations with independent, rerollable Deck-Isolation and rule-defined Archetype-Isolation samplers constrained by candidate size, minimum selection count, total deduplicated replay range, and diversity.

**Architecture:** `deck/analysis.py` will own replay-set indexing, candidate construction, bounded randomized-restart samplers, selection-detail construction, and combined auditing. `deck/deck_eda.ipynb` will own the two independent parameter cells, selected-result displays, fixed CSV writes beside the notebook, and manual reroll workflow. Existing census and similarity analysis remains intact.

**Tech Stack:** Python 3, pandas, matplotlib, Jupyter Notebook JSON, pytest.

## Global Constraints

- Deck Isolation selects exact deck IDs; no two selected exact decks may share `deck_archetype`.
- Deck similarity bands `high`, `moderate`, and `lower` have equal sampling weight when available.
- Archetype Isolation samples only archetypes matched by explicit `ARCHETYPE_RULES`.
- Archetype Isolation does not use strict core-card closure and does not guarantee core Card ID zero leakage.
- All validation-size constraints use unique `(date, episode_id)` replay unions, not summed `uses`.
- Deck and Archetype samplers have independent minimum replay, minimum count, total replay range, and `ROLL_ID` parameters.
- Failed rolls must not overwrite a previously accepted CSV.
- Selection CSVs are fixed at `imitation_learning/deck/deck_isolation_selection.csv` and `imitation_learning/deck/archetype_isolation_selection.csv`.
- This task does not split replay data or change training configuration.
- The user previously requested that automated tests and notebook execution not be run; implementation may add test cases, but only static checks are executed in this session.

---

## File Structure

- Modify `imitation_learning/deck/analysis.py`: candidate construction, independent samplers, detail tables, and combined audit.
- Modify `imitation_learning/deck/deck_eda.ipynb`: independent parameter/roll cells, displays, fixed CSV exports, and audit.
- Modify `imitation_learning/tests/test_deck_analysis.py`: synthetic behavior specifications for diversity, replay unions, archetype filtering, and audit.
- Modify `imitation_learning/tests/test_artifacts.py`: notebook headings, parameters, and fixed output-name assertions.

### Task 1: Replace strict Card-isolation data structures with generic rolls

**Files:**
- Modify: `imitation_learning/deck/analysis.py`
- Modify: `imitation_learning/tests/test_deck_analysis.py`

**Interfaces:**
- Produces `IsolationRoll(selected: tuple[str, ...], replay_count: int, attempts: int)`.
- Produces `IsolationSamplingError(message: str, diagnostics: dict[str, Any])`.
- Removes `IsolationRecommendation`, `build_card_isolation_candidates`, and `recommend_isolation_selection` after their consumers are migrated.

- [ ] **Step 1: Add behavior specifications for roll results and failures**

```python
def test_isolation_roll_records_replay_union_and_attempts() -> None:
    roll = IsolationRoll(selected=("A", "B"), replay_count=3, attempts=2)
    assert roll.selected == ("A", "B")
    assert roll.replay_count == 3
    assert roll.attempts == 2


def test_sampling_error_exposes_diagnostics() -> None:
    error = IsolationSamplingError("no feasible roll", {"eligible": 2})
    assert error.diagnostics == {"eligible": 2}
```

- [ ] **Step 2: Document the deferred test command**

Run later: `pytest imitation_learning/tests/test_deck_analysis.py -k "isolation_roll or sampling_error" -v`

Expected: FAIL before implementation because the two classes do not exist.

- [ ] **Step 3: Implement the data structures**

```python
@dataclass(frozen=True)
class IsolationRoll:
    selected: tuple[str, ...]
    replay_count: int
    attempts: int


class IsolationSamplingError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
```

- [ ] **Step 4: Record the passing command for later execution**

Run later: `pytest imitation_learning/tests/test_deck_analysis.py -k "isolation_roll or sampling_error" -v`

Expected: PASS.

### Task 2: Implement diverse Deck-Isolation rolling

**Files:**
- Modify: `imitation_learning/deck/analysis.py`
- Modify: `imitation_learning/tests/test_deck_analysis.py`

**Interfaces:**
- Consumes annotated facts and the existing `build_deck_isolation_candidates()` output.
- Produces:

```python
roll_deck_isolation_selection(
    annotated_facts: pd.DataFrame,
    candidates: pd.DataFrame,
    min_count: int,
    total_replays_min: int,
    total_replays_max: int,
    roll_id: int,
    max_attempts: int = 2000,
) -> IsolationRoll
```

- [ ] **Step 1: Add Deck-roll behavior specifications**

```python
def test_deck_roll_uses_unique_archetypes_and_replay_union() -> None:
    roll = roll_deck_isolation_selection(
        facts,
        candidates,
        min_count=2,
        total_replays_min=2,
        total_replays_max=4,
        roll_id=7,
    )
    selected = candidates.set_index("deck_id").loc[list(roll.selected)]
    assert selected["deck_archetype"].is_unique
    assert 2 <= roll.replay_count <= 4


def test_deck_roll_is_reproducible() -> None:
    first = roll_deck_isolation_selection(
        facts, candidates, 2, 2, 4, roll_id=9
    )
    second = roll_deck_isolation_selection(
        facts, candidates, 2, 2, 4, roll_id=9
    )
    assert first == second
```

- [ ] **Step 2: Document the deferred failing command**

Run later: `pytest imitation_learning/tests/test_deck_analysis.py -k "deck_roll" -v`

Expected: FAIL because `roll_deck_isolation_selection` does not exist.

- [ ] **Step 3: Implement validation and replay indexing**

Validate:

```python
if min_count < 1:
    raise ValueError("min_count must be >= 1")
if total_replays_min < 0 or total_replays_min > total_replays_max:
    raise ValueError("total replay range must satisfy 0 <= min <= max")
```

Create exact-deck replay sets from `_deck_indexes()`. Restrict candidates to
`eligible == True`, and collect diagnostics:

```python
{
    "eligible_decks": len(eligible),
    "available_archetypes": eligible["deck_archetype"].nunique(),
    "band_counts": eligible["similarity_band"].value_counts().to_dict(),
}
```

- [ ] **Step 4: Implement bounded randomized restarts**

For every attempt, derive a deterministic random stream from `roll_id`, start
empty, and repeatedly:

```python
available = eligible[
    ~eligible["deck_archetype"].isin(selected_archetypes)
    & ~eligible["deck_id"].isin(selected_ids)
]
feasible = available[
    available["deck_id"].map(
        lambda deck_id: len(
            replay_union | replays_by_deck[str(deck_id)]
        ) <= total_replays_max
    )
]
bands = sorted(feasible["similarity_band"].unique())
band = rng.choice(bands)
band_rows = feasible[feasible["similarity_band"] == band]
archetype = rng.choice(sorted(band_rows["deck_archetype"].unique()))
deck_id = rng.choice(
    sorted(
        band_rows.loc[
            band_rows["deck_archetype"] == archetype,
            "deck_id",
        ].astype(str)
    )
)
```

Return immediately when count and replay-range conditions both pass. If all
attempts fail, raise `IsolationSamplingError` with the diagnostics and no
selection.

- [ ] **Step 5: Record the deferred passing command**

Run later: `pytest imitation_learning/tests/test_deck_analysis.py -k "deck_roll" -v`

Expected: PASS.

### Task 3: Build and roll rule-defined Archetype-Isolation candidates

**Files:**
- Modify: `imitation_learning/deck/analysis.py`
- Modify: `imitation_learning/tests/test_deck_analysis.py`

**Interfaces:**
- Produces:

```python
build_archetype_isolation_candidates(
    annotated_facts: pd.DataFrame,
    catalog: CardCatalog,
    min_validation_replays: int,
) -> pd.DataFrame
```

- Produces:

```python
roll_archetype_isolation_selection(
    annotated_facts: pd.DataFrame,
    candidates: pd.DataFrame,
    min_count: int,
    total_replays_min: int,
    total_replays_max: int,
    roll_id: int,
    max_attempts: int = 2000,
) -> IsolationRoll
```

- Produces:

```python
build_archetype_selection_details(
    annotated_facts: pd.DataFrame,
    selected_archetypes: Sequence[str],
) -> pd.DataFrame
```

- [ ] **Step 1: Add Archetype candidate and roll specifications**

```python
def test_archetype_candidates_exclude_fallback_from_eligibility() -> None:
    candidates = build_archetype_isolation_candidates(
        facts, catalog(), min_validation_replays=1
    ).set_index("deck_archetype")
    assert candidates.loc["Great Tusk / Crustle", "eligible"]
    assert not candidates.loc["Mega Sharpedo", "eligible"]


def test_archetype_roll_is_uniform_unit_and_replay_bounded() -> None:
    roll = roll_archetype_isolation_selection(
        facts,
        candidates,
        min_count=2,
        total_replays_min=2,
        total_replays_max=4,
        roll_id=3,
    )
    assert len(set(roll.selected)) == len(roll.selected)
    assert 2 <= roll.replay_count <= 4
```

- [ ] **Step 2: Document the deferred failing command**

Run later: `pytest imitation_learning/tests/test_deck_analysis.py -k "archetype_candidates or archetype_roll" -v`

Expected: FAIL because the new Archetype-Isolation functions do not exist.

- [ ] **Step 3: Implement candidate aggregation**

Group annotated facts by `deck_archetype`. For each group, report:

```text
deck_archetype
classification_method
core_card_ids_json
core_card_names
replays
uses
wins
win_rate
unique_exact_decks
core_card_other_archetype_decks
core_card_other_archetype_replays
meets_min_replays
eligible
```

`eligible` requires:

```python
classification_method == "rule" and replays >= min_validation_replays
```

Core-card overlap fields are descriptive only.

- [ ] **Step 4: Implement uniform archetype rolling**

Use bounded randomized restarts. At each step, uniformly choose one remaining
eligible archetype whose addition keeps the deduplicated replay union at or
below the maximum. Stop when the minimum count and replay range both pass.
Raise `IsolationSamplingError` with candidate diagnostics on failure.

- [ ] **Step 5: Implement exact-deck detail expansion**

For selected archetypes, group by exact `deck_id` and produce:

```text
deck_archetype
deck_id
replays
uses
wins
win_rate
card_ids
```

Serialize `card_ids` as the compact sorted 60-card JSON list.

- [ ] **Step 6: Record the deferred passing command**

Run later: `pytest imitation_learning/tests/test_deck_analysis.py -k "archetype_candidates or archetype_roll or archetype_selection_details" -v`

Expected: PASS.

### Task 4: Replace the strict-core combined audit

**Files:**
- Modify: `imitation_learning/deck/analysis.py`
- Modify: `imitation_learning/tests/test_deck_analysis.py`

**Interfaces:**
- Produces:

```python
audit_deck_archetype_selection(
    annotated_facts: pd.DataFrame,
    selected_deck_ids: Sequence[str],
    selected_archetypes: Sequence[str],
) -> dict[str, Any]
```

- [ ] **Step 1: Add a combined audit specification**

```python
def test_combined_audit_checks_labels_not_core_closure() -> None:
    audit = audit_deck_archetype_selection(
        facts,
        selected_deck_ids=(deck_id,),
        selected_archetypes=("Marnie Grimmsnarl",),
    )
    assert audit["exact_deck_train_uses"] == 0
    assert audit["archetype_train_uses"] == 0
    assert "core_card_train_uses" not in audit
```

- [ ] **Step 2: Document the deferred failing command**

Run later: `pytest imitation_learning/tests/test_deck_analysis.py -k "combined_audit" -v`

Expected: FAIL because `audit_deck_archetype_selection` does not exist.

- [ ] **Step 3: Implement the label-based audit**

Build:

```python
deck_replays = union(replays_by_deck[deck_id] for selected deck IDs)
archetype_replays = set(
    facts.loc[
        facts["deck_archetype"].isin(selected_archetypes),
        "replay_key",
    ]
)
validation_replays = deck_replays | archetype_replays
```

After removing the union, report replay counts, overlap, exact-deck leakage,
missing Deck-Isolation Card IDs, missing remaining same-archetype coverage,
remaining selected-archetype uses, cross-selection archetype conflicts, and
overall validity. Do not compute strict core-card leakage.

- [ ] **Step 4: Remove superseded strict-core APIs**

After notebook and tests use the new APIs, delete:

```text
IsolationRecommendation
build_card_isolation_candidates
audit_isolation_selection
_sample_deck_ids
recommend_isolation_selection
```

Keep `archetype_core_card_ids()` because candidate tables still display core
definitions and overlap diagnostics.

- [ ] **Step 5: Record the deferred passing command**

Run later: `pytest imitation_learning/tests/test_deck_analysis.py -k "combined_audit" -v`

Expected: PASS.

### Task 5: Rebuild the notebook selection workflow and fixed CSV exports

**Files:**
- Modify: `imitation_learning/deck/deck_eda.ipynb`
- Modify: `imitation_learning/tests/test_artifacts.py`

**Interfaces:**
- Consumes all Task 2–4 APIs.
- Writes:
  - `imitation_learning/deck/deck_isolation_selection.csv`
  - `imitation_learning/deck/archetype_isolation_selection.csv`

- [ ] **Step 1: Add notebook structure specifications**

```python
for text in (
    "DECK_MIN_REPLAYS = 100",
    "DECK_MIN_COUNT = 3",
    "DECK_TOTAL_REPLAYS_MIN = 1000",
    "DECK_TOTAL_REPLAYS_MAX = 3000",
    "DECK_ROLL_ID = 0",
    "ARCHETYPE_MIN_REPLAYS = 100",
    "ARCHETYPE_MIN_COUNT = 3",
    "ARCHETYPE_TOTAL_REPLAYS_MIN = 1000",
    "ARCHETYPE_TOTAL_REPLAYS_MAX = 3000",
    "ARCHETYPE_ROLL_ID = 0",
    "deck_isolation_selection.csv",
    "archetype_isolation_selection.csv",
):
    assert text in source
assert "CARD_ISOLATION_ARCHETYPE_COUNT" not in source
assert "isolation_recommendation.json" not in source
```

- [ ] **Step 2: Document the deferred failing command**

Run later: `pytest imitation_learning/tests/test_artifacts.py::test_deck_eda_notebook_starts_with_census_and_similarity -v`

Expected: FAIL because the notebook still contains the old fixed-count strict
Card-isolation workflow.

- [ ] **Step 3: Replace the global isolation parameter cell**

Move Deck parameters directly into the Deck selection code cell and Archetype
parameters directly into the Archetype selection code cell. Keep similarity
thresholds beside Deck candidate construction:

```python
DECK_MIN_REPLAYS = 100
DECK_MIN_COUNT = 3
DECK_TOTAL_REPLAYS_MIN = 1000
DECK_TOTAL_REPLAYS_MAX = 3000
DECK_ROLL_ID = 0
```

and:

```python
ARCHETYPE_MIN_REPLAYS = 100
ARCHETYPE_MIN_COUNT = 3
ARCHETYPE_TOTAL_REPLAYS_MIN = 1000
ARCHETYPE_TOTAL_REPLAYS_MAX = 3000
ARCHETYPE_ROLL_ID = 0
```

- [ ] **Step 4: Add the Deck reroll and selected-result cell**

Build candidates with `DECK_MIN_REPLAYS`, call
`roll_deck_isolation_selection`, join selected rows to
`deck_summary["deck_card_ids_json"]`, rename that field to `card_ids`, display
all selected fields, and only then write:

```python
DECK_SELECTION_PATH = PROJECT_ROOT / "deck" / "deck_isolation_selection.csv"
selected_decks.to_csv(DECK_SELECTION_PATH, index=False, encoding="utf-8-sig")
```

Catch `IsolationSamplingError`, display its diagnostics, and do not call
`to_csv`.

- [ ] **Step 5: Replace Card-Isolation cells with Archetype-Isolation cells**

Rename headings and charts, build rule-defined candidates, display the full
candidate table, roll selected archetypes, display their summary, construct
the exact-deck detail table, and only then write:

```python
ARCHETYPE_SELECTION_PATH = (
    PROJECT_ROOT / "deck" / "archetype_isolation_selection.csv"
)
selected_archetype_decks.to_csv(
    ARCHETYPE_SELECTION_PATH,
    index=False,
    encoding="utf-8-sig",
)
```

- [ ] **Step 6: Replace recommendation JSON with the combined audit**

Delete `isolation_recommendation.json` output. When both sampler cells have
successful rolls, call `audit_deck_archetype_selection()` and display its
one-row DataFrame. If either roll failed, display that the combined audit is
unavailable without overwriting either CSV.

- [ ] **Step 7: Perform static notebook validation**

Run:

```powershell
$null = Get-Content -Raw -LiteralPath 'imitation_learning/deck/deck_eda.ipynb' | ConvertFrom-Json
```

Expected: exit code 0.

Run: `git diff --check -- imitation_learning/deck/deck_eda.ipynb`

Expected: exit code 0, allowing only line-ending warnings.

### Task 6: Final scope and artifact review

**Files:**
- Review: `imitation_learning/deck/analysis.py`
- Review: `imitation_learning/deck/deck_eda.ipynb`
- Review: `imitation_learning/tests/test_deck_analysis.py`
- Review: `imitation_learning/tests/test_artifacts.py`

- [ ] **Step 1: Inspect notebook outline**

List the notebook's markdown/code cell outline and confirm:

```text
Deck Census
Pairwise Similarity
Deck Isolation Candidates
Deck Isolation Roll
Archetype Isolation Candidates
Archetype Isolation Roll
Combined Audit
```

- [ ] **Step 2: Check fixed output paths and removed artifacts**

Search the notebook and confirm the two fixed CSV names are present while
`isolation_recommendation.json` and `CARD_ISOLATION_ARCHETYPE_COUNT` are
absent.

- [ ] **Step 3: Check repository scope**

Run: `git status --short`

Confirm no replay data, training cache, `train.yaml`, or generated selection
CSV was created during static implementation.

- [ ] **Step 4: Report verification limits**

State explicitly that pytest and notebook execution were not run under the
user's instruction. Report only the JSON, structure, and diff checks that were
actually executed.
