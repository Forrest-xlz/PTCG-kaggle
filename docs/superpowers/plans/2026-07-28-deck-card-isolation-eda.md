# Deck and Card Isolation EDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing deck EDA with constrained Deck-isolation and Card-isolation candidate tables, reproducible recommendations, and a hypothetical joint leakage audit.

**Architecture:** `deck/analysis.py` owns reusable replay annotation, candidate simulation, sampling, and audit logic. `deck/deck_eda.ipynb` owns exploration parameters, concise tables/charts, and CSV/JSON exports. The implementation evaluates hypothetical replay-level removal but does not create a dataset split or change training configuration.

**Tech Stack:** Python 3, pandas, matplotlib, seaborn, Jupyter Notebook JSON, pytest.

## Global Constraints

- Deck isolation selects exact deck IDs; Card isolation selects archetypes.
- `MIN_VALIDATION_REPLAYS` defaults to `100`.
- Deck recommendations are stratified by similarity and prefer distinct archetypes.
- Card isolation uses strict global zero leakage for the selected core Card IDs.
- Replay matching is symmetric across the two players and deduplicated by `(date, episode_id)`.
- This phase must not alter replay data, training caches, or training YAML.
- Notebook chart labels remain English to avoid missing Chinese glyphs.
- Automated tests are written in the plan, but the current execution must not run them because the user explicitly requested no test run.

---

## File Structure

- Modify `imitation_learning/deck/analysis.py`: reusable isolation analysis and recommendation functions.
- Modify `imitation_learning/deck/deck_eda.ipynb`: parameter controls, candidate exploration, charts, recommendation, audit, and exports.
- Modify `imitation_learning/tests/test_deck_analysis.py`: focused synthetic coverage for isolation semantics.
- Modify `imitation_learning/tests/test_artifacts.py`: notebook structure and export-name assertions.

### Task 1: Annotate replay-level deck facts

**Files:**
- Modify: `imitation_learning/deck/analysis.py`
- Test: `imitation_learning/tests/test_deck_analysis.py`

**Interfaces:**
- Consumes: raw deck rows accepted by `validate_deck_rows()` and a `CardCatalog`.
- Produces: `annotate_deck_facts(rows: pd.DataFrame, catalog: CardCatalog) -> pd.DataFrame`.
- Produces columns: `replay_key`, `deck_id`, `deck_archetype`, `classification_method`, `representative_card_id`, `deck_card_ids`, and the original fact columns.

- [ ] **Step 1: Add a failing annotation test**

```python
def test_annotate_deck_facts_is_replay_symmetric() -> None:
    rows = isolation_rows()
    facts = annotate_deck_facts(rows, isolation_catalog())
    assert facts.groupby("replay_key").size().eq(2).all()
    assert facts["deck_id"].notna().all()
    assert facts["deck_archetype"].notna().all()
```

- [ ] **Step 2: Run the focused test and observe the missing import**

Run: `pytest imitation_learning/tests/test_deck_analysis.py::test_annotate_deck_facts_is_replay_symmetric -v`

Expected: FAIL because `annotate_deck_facts` does not exist.

- [ ] **Step 3: Implement annotation**

```python
def annotate_deck_facts(
    rows: pd.DataFrame,
    catalog: CardCatalog,
) -> pd.DataFrame:
    facts = validate_deck_rows(rows)
    identities = facts["deck"].map(exact_deck_identity)
    classifications = facts["deck"].map(
        lambda deck: classify_deck(deck, catalog)
    )
    facts["replay_key"] = list(zip(facts["date"], facts["episode_id"]))
    facts["deck_id"] = identities.map(lambda item: item.deck_id)
    facts["deck_card_ids"] = identities.map(lambda item: item.card_ids)
    facts["deck_archetype"] = classifications.map(
        lambda item: item.archetype
    )
    facts["classification_method"] = classifications.map(
        lambda item: item.method
    )
    facts["representative_card_id"] = classifications.map(
        lambda item: item.representative_card_id
    )
    return facts
```

- [ ] **Step 4: Run the focused test**

Run: `pytest imitation_learning/tests/test_deck_analysis.py::test_annotate_deck_facts_is_replay_symmetric -v`

Expected: PASS.

- [ ] **Step 5: Commit the annotation unit**

```bash
git add -f imitation_learning/deck/analysis.py imitation_learning/tests/test_deck_analysis.py
git commit -m "feat: annotate replay deck facts for isolation analysis"
```

### Task 2: Build Deck-isolation candidates

**Files:**
- Modify: `imitation_learning/deck/analysis.py`
- Test: `imitation_learning/tests/test_deck_analysis.py`

**Interfaces:**
- Consumes: annotated facts, deck census, and similarity thresholds.
- Produces: `build_deck_isolation_candidates(facts, min_validation_replays, high_similarity_max_changed_slots, moderate_similarity_max_changed_slots) -> pd.DataFrame`.
- Candidate columns include `deck_id`, `deck_archetype`, `replays`, `uses`, `wins`, `win_rate`, `remaining_archetype_decks`, `all_cards_seen_in_train`, `unseen_card_ids_json`, `nearest_train_deck_id`, `nearest_changed_slots`, `nearest_weighted_jaccard`, `similarity_band`, `meets_min_replays`, and `eligible`.

- [ ] **Step 1: Add failing exact-deck isolation tests**

```python
def test_deck_isolation_requires_card_and_archetype_coverage() -> None:
    facts = annotate_deck_facts(isolation_rows(), isolation_catalog())
    candidates = build_deck_isolation_candidates(
        facts,
        min_validation_replays=1,
        high_similarity_max_changed_slots=5,
        moderate_similarity_max_changed_slots=15,
    ).set_index("deck_id")
    candidate = candidates.loc[exact_deck_identity(deck(1, 2)).deck_id]
    assert candidate["exact_deck_train_uses"] == 0
    assert candidate["remaining_archetype_decks"] >= 1
    assert candidate["all_cards_seen_in_train"]
    assert candidate["eligible"]
```

- [ ] **Step 2: Run the focused test**

Run: `pytest imitation_learning/tests/test_deck_analysis.py::test_deck_isolation_requires_card_and_archetype_coverage -v`

Expected: FAIL because `build_deck_isolation_candidates` does not exist.

- [ ] **Step 3: Implement replay-removal simulation and nearest-deck lookup**

Use replay-key sets per exact deck, exclude the union from training, collect
remaining Card IDs and same-archetype exact decks, then compute:

```python
distance_rows = [
    (
        train_deck_id,
        changed_slots(candidate_cards, train_cards),
        weighted_jaccard(candidate_cards, train_cards),
    )
    for train_deck_id, train_cards in same_archetype_cards.items()
]
nearest_id, nearest_slots, nearest_jaccard = min(
    distance_rows,
    key=lambda row: (row[1], -row[2], row[0]),
)
```

Set `eligible` only when the replay threshold, same-archetype coverage, Card ID
coverage, and exact-deck zero-leakage checks all pass.

- [ ] **Step 4: Run Deck-isolation tests**

Run: `pytest imitation_learning/tests/test_deck_analysis.py -k deck_isolation -v`

Expected: PASS.

- [ ] **Step 5: Commit Deck-isolation candidates**

```bash
git add -f imitation_learning/deck/analysis.py imitation_learning/tests/test_deck_analysis.py
git commit -m "feat: build constrained deck isolation candidates"
```

### Task 3: Build strict Card-isolation candidates

**Files:**
- Modify: `imitation_learning/deck/analysis.py`
- Test: `imitation_learning/tests/test_deck_analysis.py`

**Interfaces:**
- Consumes: annotated facts and `CardCatalog`.
- Produces: `build_card_isolation_candidates(facts, catalog, min_validation_replays) -> pd.DataFrame`.
- Produces helper: `archetype_core_card_ids(facts, catalog, archetype) -> tuple[int, ...]`.
- Candidate columns include `deck_archetype`, `core_card_ids_json`, `core_card_names`, `archetype_unique_decks`, `closure_unique_decks`, `closure_extra_decks`, `closure_extra_archetypes`, `validation_replays`, `remaining_training_replays`, `core_card_train_uses`, `archetype_train_uses`, `meets_min_replays`, and `eligible`.

- [ ] **Step 1: Add a failing strict-closure test**

```python
def test_card_isolation_closure_removes_other_archetypes_with_core_card() -> None:
    facts = annotate_deck_facts(card_closure_rows(), isolation_catalog())
    candidates = build_card_isolation_candidates(
        facts, isolation_catalog(), min_validation_replays=1
    ).set_index("deck_archetype")
    candidate = candidates.loc["Crustle Wall"]
    assert candidate["closure_extra_decks"] >= 1
    assert candidate["core_card_train_uses"] == 0
    assert candidate["archetype_train_uses"] == 0
    assert candidate["eligible"]
```

- [ ] **Step 2: Run the focused test**

Run: `pytest imitation_learning/tests/test_deck_analysis.py::test_card_isolation_closure_removes_other_archetypes_with_core_card -v`

Expected: FAIL because strict Card-isolation functions do not exist.

- [ ] **Step 3: Implement core resolution and strict closure**

Resolve rule markers by normalized name and retain only marker Card IDs present
in decks classified into the archetype. For fallback classifications, use the
representative Card IDs observed in that archetype. Build the closure from all
exact decks containing any resolved core ID, remove every replay containing a
closure deck, and audit both core-card and archetype training occurrences.

- [ ] **Step 4: Run Card-isolation tests**

Run: `pytest imitation_learning/tests/test_deck_analysis.py -k card_isolation -v`

Expected: PASS.

- [ ] **Step 5: Commit Card-isolation candidates**

```bash
git add -f imitation_learning/deck/analysis.py imitation_learning/tests/test_deck_analysis.py
git commit -m "feat: build strict card isolation candidates"
```

### Task 4: Add constrained recommendations and joint audit

**Files:**
- Modify: `imitation_learning/deck/analysis.py`
- Test: `imitation_learning/tests/test_deck_analysis.py`

**Interfaces:**
- Produces dataclass `IsolationRecommendation`.
- Produces `audit_isolation_selection(facts, catalog, selected_deck_ids, selected_card_archetypes) -> dict[str, Any]`.
- Produces `recommend_isolation_selection(facts, catalog, deck_candidates, card_candidates, deck_count, card_archetype_count, random_seed, max_attempts=1000) -> IsolationRecommendation`.

- [ ] **Step 1: Add failing deterministic and joint-validity tests**

```python
def test_recommendation_is_deterministic_and_jointly_valid() -> None:
    first = recommend_isolation_selection(
        facts,
        catalog,
        deck_candidates,
        card_candidates,
        deck_count=2,
        card_archetype_count=1,
        random_seed=42,
    )
    second = recommend_isolation_selection(
        facts,
        catalog,
        deck_candidates,
        card_candidates,
        deck_count=2,
        card_archetype_count=1,
        random_seed=42,
    )
    assert first == second
    assert first.audit["valid"]
```

- [ ] **Step 2: Run the focused test**

Run: `pytest imitation_learning/tests/test_deck_analysis.py -k recommendation -v`

Expected: FAIL because recommendation functions do not exist.

- [ ] **Step 3: Implement stratified seeded sampling**

Use `random.Random(random_seed)`. Fill Deck-isolation slots from high and
moderate similarity pools first, prefer unused archetypes, then fill remaining
slots from all eligible candidates. Sample Card-isolation archetypes from the
eligible archetype pool. Audit the combined hypothetical split and retry
invalid combinations up to `max_attempts`.

The audit must report component replay counts, their overlap, remaining
training replay count, exact-deck leakage, missing Deck-isolation Card IDs,
remaining selected Deck-isolation archetype coverage, Card-isolation
archetype leakage, core-card leakage, and `valid`.

- [ ] **Step 4: Run recommendation tests**

Run: `pytest imitation_learning/tests/test_deck_analysis.py -k "recommendation or audit" -v`

Expected: PASS.

- [ ] **Step 5: Commit recommendations**

```bash
git add -f imitation_learning/deck/analysis.py imitation_learning/tests/test_deck_analysis.py
git commit -m "feat: recommend and audit isolation selections"
```

### Task 5: Extend the notebook and artifacts

**Files:**
- Modify: `imitation_learning/deck/deck_eda.ipynb`
- Modify: `imitation_learning/tests/test_artifacts.py`

**Interfaces:**
- Consumes the Task 1–4 analysis APIs.
- Writes `deck_isolation_candidates.csv`, `card_isolation_candidates.csv`, and `isolation_recommendation.json` under `data/deck_analysis`.

- [ ] **Step 1: Add failing notebook artifact assertions**

```python
for heading in (
    "## Isolation Parameters",
    "## Deck Isolation Candidates",
    "## Card Isolation Candidates",
    "## Recommended Isolation Selection",
):
    assert heading in source
for artifact in (
    "deck_isolation_candidates.csv",
    "card_isolation_candidates.csv",
    "isolation_recommendation.json",
):
    assert artifact in source
```

- [ ] **Step 2: Run the notebook structure test**

Run: `pytest imitation_learning/tests/test_artifacts.py::test_deck_eda_notebook_starts_with_census_and_similarity -v`

Expected: FAIL because the new headings and artifacts are absent.

- [ ] **Step 3: Add notebook parameter and analysis cells**

Add a parameter cell with the exact defaults from the design, annotate facts
once, build both candidate tables, and display full tables with concise
eligibility summaries.

- [ ] **Step 4: Add concise English charts**

Create:

- horizontal bar chart of unique exact decks by archetype;
- Deck-isolation scatter plot of nearest changed slots versus replay count,
  colored by similarity band;
- horizontal bar chart of Card-isolation validation replay count by eligible
  archetype.

Use explicit palettes and label only high-value points where labels materially
help.

- [ ] **Step 5: Add recommendation, joint audit, and exports**

Call `recommend_isolation_selection`, display the selected exact-deck rows,
selected archetype rows, and a one-row audit table. Export candidates and:

```python
payload = {
    "parameters": isolation_parameters,
    "selected_deck_ids": list(recommendation.deck_ids),
    "selected_card_archetypes": list(recommendation.card_archetypes),
    "attempts": recommendation.attempts,
    "audit": recommendation.audit,
}
```

Write the JSON with UTF-8 and indentation. Do not write replay split files.

- [ ] **Step 6: Validate notebook JSON without executing cells**

Run: `python -m json.tool imitation_learning/deck/deck_eda.ipynb`

Expected: exit code 0.

- [ ] **Step 7: Commit notebook exploration**

```bash
git add -f imitation_learning/deck/deck_eda.ipynb imitation_learning/tests/test_artifacts.py
git commit -m "feat: explore deck and card isolation candidates"
```

### Task 6: Final static review

**Files:**
- Review: `imitation_learning/deck/analysis.py`
- Review: `imitation_learning/deck/deck_eda.ipynb`
- Review: `imitation_learning/tests/test_deck_analysis.py`
- Review: `imitation_learning/tests/test_artifacts.py`

- [ ] **Step 1: Confirm scope**

Run: `git status --short`

Expected: only intended existing worktree changes and the isolation EDA files
are present; no replay, cache, output, or training YAML files were created or
modified.

- [ ] **Step 2: Inspect notebook headings and code-cell order**

Read the notebook cell outline and confirm the original census/similarity
sections remain before the new isolation sections.

- [ ] **Step 3: Record the deferred runtime validation**

Do not execute pytest or the notebook in this session. Report that runtime
validation remains for the user, while noting that JSON/static inspection was
performed.
