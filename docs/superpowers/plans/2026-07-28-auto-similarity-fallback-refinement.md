# Automatic Similarity and Fallback Eligibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Deck-Isolation similarity bands data-driven and expand Archetype-Isolation sampling from explicit rules to curated named fallback archetypes.

**Architecture:** `deck/analysis.py` will retain nearest-deck diagnostics but choose the highest-training-use same-archetype deck as a reference for automatic tie-preserving empirical tertiles. Archetype eligibility will accept `rule` and `named_fallback_main_pokemon` while keeping raw fallback and unknown labels visible but ineligible. The notebook will remove manual similarity thresholds and display calculated cuts and band counts.

**Tech Stack:** Python 3, pandas, matplotlib, Jupyter Notebook JSON, pytest.

## Global Constraints

- Equal reference distances must never be split across similarity bands.
- Fewer than three supported bands are valid; the existing sampler renormalizes over available bands.
- Deck Isolation remains open to rule and fallback archetype labels.
- Archetype Isolation accepts `rule` and `named_fallback_main_pokemon`.
- Raw `fallback_main_pokemon`, `Unknown`, and `no_pokemon_found` remain visible but ineligible.
- The user requested no pytest or notebook execution; only static checks run in this session.

---

### Task 1: Add dominant-reference similarity metrics and automatic bands

**Files:**
- Modify: `imitation_learning/deck/analysis.py`
- Modify: `imitation_learning/tests/test_deck_analysis.py`

**Interfaces:**
- Changes `build_deck_isolation_candidates()` to remove manual high/moderate threshold arguments.
- Adds candidate columns `reference_train_deck_id`, `reference_train_uses`, `reference_changed_slots`, and `reference_weighted_jaccard`.
- Stores calculated `q33`, `q67`, and band counts in `result.attrs["similarity_bands"]`.

- [ ] **Step 1: Add behavior specifications**

```python
def test_deck_similarity_bands_use_dominant_training_reference() -> None:
    candidates = build_deck_isolation_candidates(
        annotate_deck_facts(isolation_rows(), catalog()),
        min_validation_replays=1,
    )
    assert candidates["reference_train_deck_id"].notna().any()
    assert candidates["reference_changed_slots"].notna().any()
    assert {"q33", "q67", "counts"} <= set(
        candidates.attrs["similarity_bands"]
    )


def test_equal_reference_distances_share_one_band() -> None:
    eligible = candidates[candidates["eligible"]]
    assert (
        eligible.groupby("reference_changed_slots")["similarity_band"]
        .nunique()
        .le(1)
        .all()
    )
```

- [ ] **Step 2: Document the deferred test command**

Run later: `pytest imitation_learning/tests/test_deck_analysis.py -k "similarity_bands or reference_distances" -v`

Expected before implementation: FAIL because reference columns and attrs do not exist.

- [ ] **Step 3: Compute the dominant training reference**

For every candidate, count remaining training-side uses for each same-archetype
exact deck. Select:

```python
reference_id = min(
    same_archetype_ids,
    key=lambda deck_id: (-training_uses[deck_id], deck_id),
)
```

Keep the existing nearest-deck columns independently.

- [ ] **Step 4: Assign tie-preserving empirical tertiles**

Calculate eligible reference-distance quantiles with nearest interpolation:

```python
q33 = distances.quantile(1 / 3, interpolation="nearest")
q67 = distances.quantile(2 / 3, interpolation="nearest")
```

Assign entire distance values:

```python
if distance <= q33:
    band = "high"
elif q33 < q67 and distance <= q67:
    band = "moderate"
else:
    band = "lower"
```

When all distances are equal, all candidates remain `high`. Store cut points
and actual counts in DataFrame attrs so the notebook reports the honest
distribution.

- [ ] **Step 5: Record the deferred passing command**

Run later: `pytest imitation_learning/tests/test_deck_analysis.py -k "similarity_bands or reference_distances" -v`

Expected: PASS.

### Task 2: Expand curated Archetype-Isolation eligibility

**Files:**
- Modify: `imitation_learning/deck/analysis.py`
- Modify: `imitation_learning/tests/test_deck_analysis.py`

**Interfaces:**
- `build_archetype_isolation_candidates()` adds `sampling_method_allowed`.
- `eligible` becomes `meets_min_replays and sampling_method_allowed`.

- [ ] **Step 1: Add a named-fallback eligibility specification**

```python
def test_named_fallback_is_eligible_but_raw_fallback_is_not() -> None:
    candidates = build_archetype_isolation_candidates(
        facts, catalog(), min_validation_replays=1
    ).set_index("deck_archetype")
    assert candidates.loc["Mega Sharpedo", "sampling_method_allowed"]
    assert candidates.loc["Mega Sharpedo", "eligible"]
```

- [ ] **Step 2: Document the deferred failing command**

Run later: `pytest imitation_learning/tests/test_deck_analysis.py -k "named_fallback" -v`

Expected before implementation: FAIL because named fallback is currently ineligible.

- [ ] **Step 3: Implement method eligibility**

```python
allowed_methods = {"rule", "named_fallback_main_pokemon"}
sampling_method_allowed = bool(methods) and set(methods).issubset(
    allowed_methods
)
eligible = sampling_method_allowed and meets_min
```

Keep `rule_defined` as a descriptive column and display the actual
`classification_method`.

- [ ] **Step 4: Record the deferred passing command**

Run later: `pytest imitation_learning/tests/test_deck_analysis.py -k "named_fallback" -v`

Expected: PASS.

### Task 3: Update the notebook controls and displays

**Files:**
- Modify: `imitation_learning/deck/deck_eda.ipynb`
- Modify: `imitation_learning/tests/test_artifacts.py`

**Interfaces:**
- Consumes automatic band attrs and new reference columns.
- Removes `HIGH_SIM_MAX_CHANGED_SLOTS` and `MODERATE_SIM_MAX_CHANGED_SLOTS`.

- [ ] **Step 1: Update notebook structure specifications**

```python
assert "HIGH_SIM_MAX_CHANGED_SLOTS" not in source
assert "MODERATE_SIM_MAX_CHANGED_SLOTS" not in source
for text in (
    "reference_train_deck_id",
    "reference_changed_slots",
    "Automatic similarity cut points",
    "sampling_method_allowed",
):
    assert text in source
```

- [ ] **Step 2: Remove manual thresholds and display calculated bands**

Build Deck candidates with only `DECK_MIN_REPLAYS`. Display:

```python
band_info = deck_isolation_candidates.attrs["similarity_bands"]
display(pd.DataFrame([{
    "q33": band_info["q33"],
    "q67": band_info["q67"],
    **band_info["counts"],
}]))
```

Replace nearest-distance x-axis and threshold lines with
`reference_changed_slots`. Retain nearest metrics in the selected table as
secondary diagnostics.

- [ ] **Step 3: Update Archetype candidate text and columns**

Explain that explicit rules and curated named fallback labels are eligible.
Display `classification_method`, `sampling_method_allowed`, `rule_defined`,
and `eligible`.

- [ ] **Step 4: Perform static checks**

Parse the notebook with PowerShell `ConvertFrom-Json`, search for removed and
added identifiers, and run `git diff --check`. Do not execute notebook cells.
