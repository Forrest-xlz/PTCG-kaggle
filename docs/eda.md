# EDA and Data Coverage

EDA entrypoints live in `imitation_learning/eda/`; reusable logic lives in `analysis/`. Validation selection is training preparation and uses `training/select_validation_decks.py`; see [training and validation](training-validation.md).

## Deck EDA

Open `eda/deck_eda.ipynb` and run cells in order. TRAIN_CONFIG defaults to `cfg/train_policy.yaml`. Named top decks and training-selection settings are read from this file rather than duplicated in the notebook.

Inputs include deck CSVs, the training cache, replay manifests, the card table, and enabled holdout selection files. This analysis scans metadata; it does not train a model.

Outputs under `outputs/deck_eda/`:

| File | Contents |
|---|---|
| deck_coverage.csv | Coverage of all exact decks, including configured targets with no data |
| top_deck_coverage.csv | Named top-deck coverage summary |
| top_deck_archetype_variants.csv | Same-archetype variants, card_ids, similarity, and coverage |
| top_deck_coverage.png | Raw, training, and expert-training replay counts for exact top decks |
| top_deck_archetype_coverage.png | Similarity versus replay count for same-archetype variants |
| coverage_context.json | Paths, date ranges, selection settings, and total action count |

Definitions:

- raw_replays: unique (date, episode) count per exact deck; mirrors count once.
- train_replays / train_action_samples: replay and action counts in the selected training pool.
- expert_*: expert episodes and their actions under the expert-validation rule; the acting player is not necessarily above the expert cutoff.
- changed_slots: card slots changed relative to the top deck, accounting for duplicates; zero means an exact match.
- weighted_jaccard: count-aware deck similarity.

Only same-archetype comparisons against configured top decks are computed, not all-pairs similarity. Raw CSV dates may exceed cache dates; inspect context. Summing per-deck replay counts does not yield unique archetype replay counts. max_samples and repeated epochs affect exposure, not these training-pool counts.

Fixed chart and CSV filenames are overwritten. Low counts indicate potential coverage risks, not proof of a model bottleneck. Combine them with independent validation or data-scaling experiments using a fixed validation set.

## Deck Trends

```bash
python -m extraction.deck_trend_data
python eda/deck_trends_eda.py
```

Extraction configuration: `cfg/extract_deck_trend_data.yaml`. Analysis configuration: `cfg/deck_trends_eda.yaml`. Output: `outputs/deck_trends/`.

The line chart, Sankey, and matchup matrix independently configure dates, interval_days, min_share_percent, score_mode, and score_threshold. Line and matrix settings also include exclude_mirror_matches.

- The line chart shows usage share and win rate by date group.
- The Sankey shows changes in teams' predominant decks between selected snapshots.
- The matrix shows matchup win rates between archetypes.
- all disables score filtering; min/max/avg filter by the minimum, maximum, or average participant score.
- Excluding mirrors isolates results against other decks. Including them better reflects the observed opponent distribution including mirrors, but remains subject to sampling bias.

Outputs are three fixed PNGs—archetype_share_and_win_rate, archetype_sankey, and archetype_matchup_matrix—plus line_chart_daily_metrics.csv and matchup_matrix.csv. Existing outputs are replaced after the complete set renders successfully.

Rerun extraction after adding or replacing replay ZIPs. With force disabled, unchanged complete date shards are reused.

## Replay Timing

```bash
python -m extraction.replay_timing
python eda/replay_timing_eda.py
```

Extraction configuration: `cfg/extract_replay_timing.yaml`. Analysis configuration: `cfg/replay_timing_eda.yaml`. Both support date: latest or an explicit date such as 8.15.

Extraction writes dated caches under `data/replay_timing/` and an error CSV only when replay errors occur. Analysis writes to `outputs/replay_timing/`:

- Six charts: startup distributions, action-time distributions, and startup-versus-action scatterplots, for all data and score-filtered data.
- all_team_timings.csv: statistics for all teams.
- score_filtered_team_timings.csv: statistics after score filtering.
- cluster_summary.csv: cluster summaries.

Filenames are fixed. Successful runs replace corresponding outputs; rendering failures preserve the previous complete results.
