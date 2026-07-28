# Diverse Deck and Archetype Isolation Sampling Design

## Goal

Replace the fixed-count isolation recommendation in `deck_eda.ipynb` with two
independent, rerollable samplers:

- **Deck Isolation:** select unseen exact decks while retaining their
  archetype and Card IDs on the hypothetical training side.
- **Archetype Isolation:** select complete rule-defined or curated named
  fallback archetypes and remove all replays containing a deck whose primary
  label is one of those archetypes.

The EDA is a selection aid. It displays every selected object for manual
review and writes three selected-deck CSV files that a later training-data task
can consume. It does not create validation caches or modify training
configuration.

This design supersedes strict core-card closure sampling. Archetype Isolation
does not guarantee that an archetype's core Card IDs are absent from decks
with other primary archetype labels.

## Common Replay Semantics

Each replay has two deck rows, one per player. A replay matches a selected
exact deck or archetype when either player matches it.

Replay totals are deduplicated by `(date, episode_id)`. If two selected decks
play against each other, that replay contributes one—not two—to the selected
validation size.

`uses` remains a separate descriptive metric and counts player-side deck
occurrences.

## Top-Deck Definition

The notebook owns two editable values for the model's final deck:

```python
TOP_DECK_ARCHETYPE = "..."
TOP_DECK_CARD_IDS = [...]
```

`TOP_DECK_CARD_IDS` must contain exactly 60 non-negative integer Card IDs. Its
exact-deck identity is excluded from every isolation selection and CSV, but a
replay is not excluded merely because the top deck appears as the opponent.

The ordinary Deck-Isolation and Archetype-Isolation samplers exclude
`TOP_DECK_ARCHETYPE`. A dedicated selector handles variants within that
archetype without selecting the model's exact top deck as a validation deck.

## Deck-Isolation Parameters

The Deck Isolation notebook cell owns its parameters:

```python
DECK_MIN_REPLAYS = 100
DECK_MIN_COUNT = 3
DECK_TOTAL_REPLAYS_MIN = 1000
DECK_TOTAL_REPLAYS_MAX = 3000
DECK_ROLL_ID = 0
```

- `DECK_MIN_REPLAYS`: minimum unique replay count for each exact-deck
  candidate.
- `DECK_MIN_COUNT`: minimum number of selected exact decks.
- `DECK_TOTAL_REPLAYS_MIN/MAX`: allowed inclusive range for the union of
  selected replay IDs.
- `DECK_ROLL_ID`: changes the deterministic random stream. Incrementing it
  rerolls the selection while preserving reproducibility.

The parameter cell validates:

```text
DECK_MIN_REPLAYS >= 1
DECK_MIN_COUNT >= 1
0 <= DECK_TOTAL_REPLAYS_MIN <= DECK_TOTAL_REPLAYS_MAX
```

## Deck-Isolation Candidate Constraints

Before random selection, each exact deck is evaluated by hypothetically moving
every replay containing that deck to validation.

An eligible exact deck must:

1. Have at least `DECK_MIN_REPLAYS` unique replays.
2. Have zero remaining exact-deck occurrences on the hypothetical training
   side.
3. Leave at least one other exact deck with the same primary archetype on the
   training side.
4. Leave every Card ID from the selected deck visible somewhere on the
   training side.
5. Leave at least one training replay.

The nearest remaining same-archetype exact deck remains visible as:

- `nearest_changed_slots`
- `nearest_weighted_jaccard`

It is not used for similarity bands because nearest-deck distance often
collapses to one changed slot when many small variants exist.

For banding, the hypothetical training-side exact deck with the highest usage
within the same archetype becomes the candidate's reference deck. Ties are
resolved by stable `deck_id`. The analysis records:

- `reference_train_deck_id`
- `reference_changed_slots`
- `reference_weighted_jaccard`

Candidates are sorted by `reference_changed_slots` and divided automatically
using the 1/3 and 2/3 empirical quantiles:

- `high`: closest third to the dominant reference build
- `moderate`: middle third
- `lower`: farthest third

Equal `reference_changed_slots` values are never split across bands. If ties
make fewer than three non-empty bands possible, only the bands supported by
the data are created, and sampling renormalizes equally across those bands.
The notebook displays the calculated cut points and candidate counts for every
resulting band.

## Deck-Isolation Sampling

The sampler uses bounded randomized restarts.

Within one attempt:

1. Start with an empty selection and empty replay union.
2. Choose among currently available `high`, `moderate`, and `lower` bands with
   equal probability. If one or more bands have no feasible candidates,
   renormalize uniformly across the remaining bands.
3. Within the chosen band, choose uniformly among primary archetypes not
   already selected.
4. Within the chosen `(band, archetype)` group, choose one exact deck
   uniformly.
5. Never select two exact decks with the same primary archetype.
6. Add the candidate only if the deduplicated replay union does not exceed
   `DECK_TOTAL_REPLAYS_MAX`.
7. Stop when both conditions hold:
   - at least `DECK_MIN_COUNT` decks are selected;
   - deduplicated replay total is within the configured inclusive range.

If the attempt gets stuck before satisfying both conditions, restart from an
empty selection. The retry count is bounded. Failure returns diagnostics
instead of an invalid selection, including eligible candidate count, available
archetype count, band counts, and reachable replay context.

Deck usage volume never increases sampling weight.

## Archetype-Isolation Parameters

The Archetype Isolation notebook cell owns a separate parameter set:

```python
ARCHETYPE_MIN_REPLAYS = 100
ARCHETYPE_MIN_COUNT = 3
ARCHETYPE_TOTAL_REPLAYS_MIN = 1000
ARCHETYPE_TOTAL_REPLAYS_MAX = 3000
ARCHETYPE_ROLL_ID = 0
```

- `ARCHETYPE_MIN_REPLAYS`: minimum unique replay count for the whole primary
  archetype.
- `ARCHETYPE_MIN_COUNT`: minimum number of selected archetypes.
- `ARCHETYPE_TOTAL_REPLAYS_MIN/MAX`: allowed inclusive range for the union of
  selected archetype replay IDs.
- `ARCHETYPE_ROLL_ID`: deterministic reroll control.

Archetypes classified by either of these methods are eligible:

- `rule`
- `named_fallback_main_pokemon`

The named fallback method uses a representative main Pokémon whose display
name has an explicit curated entry in `FALLBACK_ARCHETYPE_NAMES`.

Raw `fallback_main_pokemon`, `Unknown`, and `no_pokemon_found` classifications
remain visible in the complete EDA tables but cannot be randomly selected for
Archetype Isolation.

The minimum replay condition applies to the archetype as a whole. Individual
exact-deck variants inside that archetype do not need to satisfy the threshold.

## Archetype-Isolation Sampling

The sampler uses bounded randomized restarts:

1. Filter to rule-defined or curated named-fallback archetypes with at least
   `ARCHETYPE_MIN_REPLAYS` unique replays.
2. Sample eligible archetypes uniformly without replacement.
3. Add an archetype only if the deduplicated replay union does not exceed
   `ARCHETYPE_TOTAL_REPLAYS_MAX`.
4. Stop after selecting at least `ARCHETYPE_MIN_COUNT` archetypes and reaching
   the configured replay range.
5. Restart when an attempt cannot reach the constraints.

Every eligible archetype has equal sampling weight. Archetype replay volume and
the number of exact-deck variants do not affect its probability.

Archetype Isolation removes only replays containing decks whose
`deck_archetype` equals a selected value. Core-card overlap with other
`deck_archetype` labels is reported for interpretation but is not a hard
constraint.

## Notebook Layout

The notebook keeps the existing census and similarity exploration, then
presents:

1. Top-deck definition and dedicated same-archetype variant selector.
2. Deck Isolation candidate table and plots.
3. A Deck Isolation parameter and reroll cell.
4. The selected Deck Isolation table.
5. Archetype Isolation candidate tables and plots.
6. An Archetype Isolation parameter and reroll cell.
7. The selected archetype summary and selected exact-deck detail table.
8. A combined hypothetical audit.

All chart and column labels remain English.

## Deck-Isolation Output

The selected table contains one row per selected exact deck:

```text
deck_id
deck_archetype
replays
uses
win_rate
similarity_band
nearest_train_deck_id
nearest_changed_slots
nearest_weighted_jaccard
card_ids
```

It is written, with no configurable output directory, to:

```text
imitation_learning/data/deck_isolation_selection.csv
```

`card_ids` stores the complete sorted 60-card list as compact JSON.

## Archetype-Isolation Output

The notebook first displays one summary row per selected archetype:

```text
deck_archetype
core_card_names
replays
uses
win_rate
unique_exact_decks
```

It then displays one detail row per exact deck belonging to the selected
archetypes:

```text
deck_archetype
deck_id
replays
uses
win_rate
card_ids
```

The exact-deck detail table is written to:

```text
imitation_learning/data/archetype_isolation_selection.csv
```

This expanded mapping lets later training-data code identify validation
replays using exact 60-card multisets without reimplementing archetype
classification.

Rerunning a sampler with a different `ROLL_ID` overwrites only that sampler's
selection CSV. The current CSV represents the latest manually reviewed roll.

## Top-Deck Archetype Deck Isolation

A third notebook cell selects exact-deck variants from the configured
`TOP_DECK_ARCHETYPE`:

```python
TOP_DECK_MIN_REPLAYS = 100
TOP_DECK_MIN_COUNT = 3
TOP_DECK_TOTAL_REPLAYS_MIN = 1000
TOP_DECK_TOTAL_REPLAYS_MAX = 3000
TOP_DECK_ROLL_ID = 0
```

Candidates must satisfy the ordinary Deck-Isolation replay, exact-deck,
training-card, and remaining-training constraints. The configured exact top
deck is never eligible. Multiple variants from the target archetype may be
selected because archetype uniqueness is intentionally not enforced here.

Similarity is measured directly against `TOP_DECK_CARD_IDS` using
`top_deck_changed_slots` and `top_deck_weighted_jaccard`. Eligible candidates
are divided into tie-preserving empirical thirds, and the sampler chooses
uniformly among available bands before choosing a deck uniformly within the
band. Replay totals use the deduplicated union.

The selected table contains the target archetype, exact deck ID, replay and
usage statistics, direct top-deck similarity, band, and complete Card ID list.
It is written to:

```text
imitation_learning/data/top_deck_archetype_isolation_selection.csv
```

## Combined Audit

The final notebook cell simulates removing the union of all three selected replay
sets and reports:

- Deck Isolation unique replay count.
- Top-deck archetype Deck Isolation unique replay count.
- Archetype Isolation unique replay count.
- Replay overlap among the three selections.
- Combined validation and remaining-training replay counts.
- Exact-deck leakage for Deck Isolation.
- Missing training-side Card IDs for Deck Isolation.
- Missing training-side same-archetype coverage for Deck Isolation.
- Remaining selected Archetype Isolation occurrences on the training side.
- Any primary archetype selected by both isolation methods.
- Whether the exact configured top deck was accidentally selected.
- Overall audit validity and diagnostics.

The exact top deck may still occur as an opponent in a selected replay. This is
intentional and is not an audit failure.

The combined audit does not require core Card IDs to disappear globally.

The samplers are independent selection cells. The combined audit is the final
manual acceptance guard: a user can change either `ROLL_ID` and rerun until
the displayed combination is acceptable.

## Failure Behavior

A sampler must not write a new selection CSV when no valid combination is
found. It displays:

- the violated parameter range;
- eligible candidate count;
- available unique archetype count;
- similarity-band counts where applicable;
- a concise suggestion to reduce the minimum count, lower the replay minimum,
  or widen the total replay interval.

Previously accepted CSV files are not silently replaced by an invalid or empty
selection.

## Scope Exclusions

This EDA change does not:

- split replay data;
- modify `train.yaml`;
- alter the latest-date or random replay validation sets;
- create training caches;
- guarantee core Card ID zero leakage for Archetype Isolation;
- automatically accept a sampled selection on the user's behalf.
