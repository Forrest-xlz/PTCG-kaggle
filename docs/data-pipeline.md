# Data Pipeline and Feature Cache

## Input resources

The default replay directory is `replay_episodes/` at the repository root. Extractors process ZIP archives, with dates parsed as `month.day`. This date representation has no year component; do not mix archives from different years with the same month and day.

Expert filtering reads `episode_id`, `min_score`, `sum_score`, and `agent_count` from the single `manifest.csv` in each ZIP. It requires two participants and reconstructs the other score as `sum_score - min_score`. Missing dates, duplicate dated archives, and invalid manifests are errors, not zero-sample observations.

Feature construction requires the competition `cg` package. Set `cg_path` in `cfg/build_feature_cache.yaml` and `cfg/train_policy.yaml` to its parent directory. Deck EDA and validation-deck selection also require `pokemon_tcg_ai_battle/EN_Card_Data.csv`.

## Main pipeline

| Command | Configuration | Default output |
|---|---|---|
| `python -m extraction.deck_lists` | `cfg/extract_deck_lists.yaml` | `data/deck/` |
| `python -m extraction.training_samples` | `cfg/extract_training_samples.yaml` | `data/training/` |
| `python -m training.build_feature_cache` | `cfg/build_feature_cache.yaml` | `data/training_cache/` |

Each extractor writes a data shard and metadata per ZIP and skips completed, valid shards. `workers` controls parallelism, `force` controls rebuilding, and `limit_members` supports small checks. ZIP decompression and JSON parsing consume both CPU and memory; increasing workers is not always beneficial.

Use separate output directories for smoke tests and update the cache input accordingly. For full extraction, restore `limit_members: null` and use a fresh full-data output directory or explicitly rebuild. Do not mix test and production artifacts.

## Output records

### Deck CSV

Each replay contributes two rows containing the players' complete sorted 60-card decks, date, episode ID, player, reward, and win/loss/draw result. These records do not contain action sequences or full observations.

A mirror match produces two usage records for the same deck. Deck usage count and unique replay count are therefore not always equal.

### Training JSONL

Each line of `data/training/<date>.jsonl.gz` is a decision sample containing replay identity, player, complete deck, active observation, and action. Replay actions are recorded in the next step, so extraction pairs the observation at `steps[t]` with the action at `steps[t+1]`, retaining only players with ACTIVE status.

Current records contain both players and their results. Training selects winner actions and optionally qualified loser actions later; changing loser augmentation does not require replay extraction again. The current extraction schema is 3.

### Feature cache

The current cache schema is 16. Model-ready mmap shards store stable replay keys (32-bit), exact-deck keys (64-bit), player results, the acting player's previous three actions, and encoded features. `samples_per_shard` bounds each physical shard; compatible, completed source caches can be reused.

Compact arrays reduce storage and memory requirements, and training tensors are materialized per batch. Candidate actions enumerate legal selection sizes from maxCount down to minCount, retaining at most the first 64 combinations. Selection order within an action is not treated as a distinction.

## When to rebuild

- Network width, depth, dropout, and other model-only settings: generally no extraction or cache rebuild.
- Training splits, expert ratios, isolation CSVs, or loser augmentation: reselect from the existing compatible two-player cache.
- Feature layout or cache schema changes: rebuild the cache.
- Old winner-only JSONL or an incompatible extraction schema: extract again, then rebuild.
- New or replaced replay ZIPs: rerun the relevant extractor, check compatibility, and use force when needed.

Training, standalone validation, and EDA must use compatible data ranges. A directory's existence does not prove completeness. Generated datasets and model artifacts are excluded from Git by default.
