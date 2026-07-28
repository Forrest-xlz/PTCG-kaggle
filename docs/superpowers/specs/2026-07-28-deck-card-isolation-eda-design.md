# Deck and Card Isolation EDA Design

## Goal

Extend the deck EDA so it can identify good candidates for two future
generalization validation sets:

- **Deck isolation:** the validation exact deck is unseen in training, while its
  archetype and every Card ID in that deck remain seen in training.
- **Card isolation:** the validation archetype and its core Card IDs are unseen
  in training. Generic cards may remain seen.

This phase only explores and selects candidate exact decks and archetypes. It
does not split replay data, create validation caches, or change training
configuration.

## Units of Selection

- Deck isolation selects an **exact deck ID**.
- Card isolation selects an **archetype**. Selecting an archetype includes all
  of its exact decks and applies a strict core-card closure.

The notebook exposes:

```python
MIN_VALIDATION_REPLAYS = 100
DECK_ISOLATION_COUNT = 3
CARD_ISOLATION_ARCHETYPE_COUNT = 3
HIGH_SIM_MAX_CHANGED_SLOTS = 5
MODERATE_SIM_MAX_CHANGED_SLOTS = 15
RANDOM_SEED = 42
```

These parameters control exploration and reproducible recommendations only.

## Replay-Level Isolation

Isolation is defined at replay level. If either player's deck matches an
isolated exact deck, or contains a core Card ID covered by a card-isolation
selection, the whole replay is treated as validation for the hypothetical
split.

The player's side does not matter. Replay IDs are deduplicated before counts
are reported.

## Deck-Isolation Candidate Constraints

For each exact deck, simulate removing every replay in which that deck appears.
The deck is eligible only when:

1. It appears in at least `MIN_VALIDATION_REPLAYS` unique replays.
2. At least one other exact deck with the same archetype remains on the
   hypothetical training side.
3. Every Card ID in the isolated deck still appears in at least one training
   deck.
4. The isolated exact deck has no remaining training occurrence.

For each eligible candidate, compare it with the remaining exact decks of the
same archetype. Record the nearest deck according to:

- `changed_slots = 0.5 * sum(abs(count_a - count_b))`
- weighted Jaccard:
  `sum(min(count_a, count_b)) / sum(max(count_a, count_b))`

Candidates are assigned to similarity bands:

- high similarity: `changed_slots <= HIGH_SIM_MAX_CHANGED_SLOTS`
- moderate similarity:
  `HIGH_SIM_MAX_CHANGED_SLOTS < changed_slots <= MODERATE_SIM_MAX_CHANGED_SLOTS`
- lower similarity: above the moderate threshold

## Card-Isolation Candidate Constraints

Core cards are derived from the archetype rules used by the current analysis.
Rule marker names are resolved to their Card IDs. For fallback archetypes, the
representative main Pokémon defines the core marker.

Card isolation uses strict global zero leakage:

1. Start from one archetype and its core Card IDs.
2. Find every exact deck containing any of those core Card IDs, including decks
   classified under other archetypes.
3. Treat every replay containing any such deck on either side as validation.
4. Verify that the hypothetical training side contains zero occurrences of all
   selected core Card IDs.

The candidate table first reports the number of unique exact decks in each
archetype. A card-isolation archetype is eligible only when its strict closure
contains at least `MIN_VALIDATION_REPLAYS` unique replays and the zero-leakage
check succeeds.

The table also reports how much the strict closure expands beyond the original
archetype, so broad or ambiguous core definitions remain visible to the user.

## Reproducible Recommendation

The notebook retains the complete candidate tables and additionally produces a
reproducible recommendation using `RANDOM_SEED`.

Deck isolation randomly selects `DECK_ISOLATION_COUNT` exact decks after
constraint filtering. Sampling is stratified across high and moderate
similarity when both groups are available and prefers distinct archetypes for
diversity. Remaining slots are filled from the full eligible pool.

Card isolation randomly selects
`CARD_ISOLATION_ARCHETYPE_COUNT` eligible archetypes. Selecting one archetype
means selecting all exact decks and strict-closure replays associated with its
core cards.

After sampling, a joint hypothetical audit recomputes the constraints for the
combined selection. If a combination becomes invalid because candidates
overlap or jointly remove required training coverage, the sampler retries with
the same deterministic random stream. If no valid combination is found within
a bounded number of attempts, it reports the reason instead of silently
returning an invalid recommendation.

## Notebook Outputs

### Deck isolation

- Full candidate table containing exact deck ID, archetype, replay count, uses,
  win rate, nearest remaining same-archetype deck, changed slots, weighted
  Jaccard, similarity band, and remaining same-archetype deck count.
- Scatter plot with changed slots on the x-axis, replay count on the y-axis,
  and similarity band as color.
- Reproducibly sampled recommendation table.

### Card isolation

- Bar chart of unique exact-deck count by archetype.
- Full candidate table containing archetype, core cards, original exact-deck
  count, strict-closure exact-deck count, validation replay count, closure
  expansion into other decks/archetypes, and remaining training replay count.
- Bar chart of validation replay count by eligible archetype.
- Reproducibly sampled recommendation table.

### Joint audit

The final table reports:

- total hypothetical validation replay count;
- overlap between deck- and card-isolation replay sets;
- exact-deck leakage for deck isolation;
- training-side Card ID coverage for deck isolation;
- archetype and core-card leakage for card isolation;
- an overall validity flag and diagnostics.

## Saved Artifacts

The EDA saves:

- `deck_isolation_candidates.csv`
- `card_isolation_candidates.csv`
- a small JSON containing parameters, recommended exact deck IDs, recommended
  card-isolation archetypes, and joint-audit results

These artifacts are selection aids for the later validation-splitting work.
They do not themselves alter the replay or training datasets.
