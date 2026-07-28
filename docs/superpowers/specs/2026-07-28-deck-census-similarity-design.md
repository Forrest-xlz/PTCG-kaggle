# Deck Census and Similarity Exploration Design

## Scope

This phase replaces the existing deck EDA notebook with a small,
reproducible exploration focused on:

1. YAML-driven extraction of both decks from every replay;
2. rule-based deck archetypes based on the supplied reference notebook;
3. stable exact-deck identities and usage/win-rate statistics;
4. a complete pairwise similarity table for all unique exact decks.

Deck-isolated and card-isolated model validation sets are future work. The
analysis functions are kept reusable so that later validation construction
can use the same definitions without copying notebook code.

## YAML-Driven Extraction

`deck/extract.py` no longer accepts command-line configuration. It reads
`cfg/deck_extract.yaml`:

```yaml
extract:
  input: ../replay_episodes
  output: data/deck
  workers: 16
  limit_members: null
  force: false
```

Relative paths resolve from the `imitation_learning` project directory.
Configuration validation requires:

- an existing ZIP file or directory containing ZIP files;
- `workers >= 1`;
- `limit_members` equal to `null` or an integer greater than zero;
- a boolean `force`.

Extraction remains parallel by ZIP and resumable. One replay produces exactly
two rows, one for each player. The existing fact-only schema remains:

- `date`;
- `episode_id`;
- `player`;
- sorted complete 60-card `deck`;
- `reward`;
- `result`.

Archetypes, exact-deck IDs, and similarity are deliberately not stored by the
extractor. Changing analysis rules therefore never requires replay extraction.

## Reusable Deck Analysis

Deck logic lives in importable Python modules under `deck/`, while the notebook
orchestrates and displays results.

### Card Metadata

The analysis loads `EN_Card_Data.csv` and constructs:

- Card ID to English card name;
- Card ID to card kind;
- a predicate identifying Pokémon cards.

Names are normalized with Unicode NFKC, curly-to-straight apostrophe
conversion, whitespace normalization, and case folding.

### Archetype Classification

The ordered marker-card rules and fallback aliases come from
`ptcg-ai-battle-1100-plus-exact-deck-meta-v3-score-ranked.ipynb`.

Rules support:

- `all`: every named marker must occur;
- `any`: at least one named marker must occur;
- both keys: both conditions must pass.

The first matching rule wins. Classification returns:

- `deck_archetype`;
- `classification_method`;
- `classification_evidence`;
- representative card ID and name.

When no explicit rule matches, the fallback:

1. counts only Pokémon cards by normalized name;
2. chooses the most frequent Pokémon whose name ends with `ex`, if present;
3. otherwise chooses the most frequent Pokémon;
4. resolves count ties by normalized name;
5. applies a known fallback alias where available;
6. otherwise labels the deck `Other / <Pokémon name>`.

Decks without 60 cards are invalid and do not enter the analysis.

### Exact Deck Identity

An exact deck is the multiset of 60 Card IDs. Card order is ignored and
multiplicity is preserved.

The canonical representation is the comma-separated sorted Card ID list.
SHA-256 over that representation produces:

- the complete hash for collision checks;
- `deck_id = "D-" + first 12 uppercase hexadecimal characters`.

`deck_id` is stable across dates, reruns, and usage-rank changes.

## Deck Census Table

Every row in `deck_summary.csv` represents one exact deck. Rows are ranked by
usage descending, then stable deck ID ascending.

Columns include:

- `rank`;
- `deck_archetype`;
- `classification_method`;
- `classification_evidence`;
- `representative_card_id`;
- `representative_card_name`;
- `deck_id`;
- complete SHA-256 hash;
- `uses`;
- `usage_percent`;
- `wins`;
- `losses`;
- `draws`;
- `win_rate`;
- `first_date`;
- `last_date`;
- `unique_opponent_decks`;
- sorted Card IDs as compact JSON;
- Card ID counts as compact JSON.

Usage counts player-deck rows, so every successfully extracted replay
contributes two uses. `usage_percent = uses / all valid deck rows`.
`win_rate = wins / uses`; draws count as uses but not wins.

`unique_opponent_decks` is computed by pairing the two player rows within each
episode and counting distinct opponent exact-deck IDs.

## Complete Pairwise Similarity

For `K` exact decks, `deck_similarity_pairs.csv` contains exactly
`K * (K - 1) / 2` unordered pairs. Self-pairs and reversed duplicates are
omitted.

For card multiplicities `n1[i]` and `n2[i]`:

```text
changed_slots = 0.5 * sum_i(abs(n1[i] - n2[i]))
weighted_jaccard = sum_i(min(n1[i], n2[i]))
                    / sum_i(max(n1[i], n2[i]))
```

Because every valid deck contains 60 cards, `changed_slots` is an integer in
`[0, 60]`. Weighted Jaccard is in `[0, 1]`, with larger values meaning greater
similarity.

Every pair row contains:

- `deck_id_1`, `deck_id_2`;
- `archetype_1`, `archetype_2`;
- `same_archetype`;
- `uses_1`, `uses_2`;
- `changed_slots`;
- `weighted_jaccard`.

Rows sort by changed slots ascending, weighted Jaccard descending, then both
deck IDs ascending.

## Notebook

The existing `deck/deck_eda.ipynb` is replaced rather than extended. It is an
incremental exploration notebook with:

1. `Goal`;
2. `Setup`;
3. `Load & Validate`;
4. `Deck Census`;
5. `Pairwise Similarity`;
6. `Checks`;
7. `Next Steps`.

The setup cell exposes the extracted-deck directory, English card table, and
analysis output directory. The notebook uses English headings and labels to
avoid font-rendering problems.

It saves:

- `data/deck_analysis/deck_summary.csv`;
- `data/deck_analysis/deck_similarity_pairs.csv`.

The notebook does not yet add charts or implement deck/card-isolated model
validation sets.

## Validation and Errors

Extraction errors remain summarized per archive without stopping unrelated
archives.

Analysis rejects or reports:

- malformed deck JSON;
- decks not containing exactly 60 integer Card IDs;
- duplicate `(date, episode_id, player)` rows;
- episodes without exactly two players;
- unknown Card IDs;
- invalid result labels;
- deck-hash collisions with different canonical decks.

Checks verify:

- exactly two rows per valid episode;
- usage percentages sum to 100% within floating tolerance;
- wins, losses, and draws sum to uses for every deck;
- the pair count equals `K * (K - 1) / 2`;
- changed slots and Weighted Jaccard remain in their valid ranges.

Unit tests cover configuration validation, stable exact-deck IDs, ordered
archetype rules, fallback behavior, census aggregation, opponent pairing, and
both distance formulas.
