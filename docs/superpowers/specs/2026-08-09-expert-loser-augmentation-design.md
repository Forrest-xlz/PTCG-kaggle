# Expert Loser Replay Augmentation Design

## Goal

Augment behavior-cloning training with losing-player actions from recent,
high-skill replays. A loss does not imply that each observed action was poor in
an imperfect-information stochastic game. The augmentation must leave every
existing validation split and validation metric unchanged so the experiment
measures whether these additional actions improve winning-policy imitation.

## Configuration

Add the following nested training configuration:

```yaml
train:
  loser_augmentation:
    enabled: true
    recent_dates: 3
    expert_ratio: 0.05
```

- `enabled` enables or disables loser augmentation without rebuilding extracted
  data or feature caches.
- `recent_dates` is the number of most recent training-eligible archive dates.
  The latest archive date is reserved for latest-date validation and is not
  counted. It must be at least one when augmentation is enabled.
- `expert_ratio` is the daily top-player fraction used to calculate the score
  cutoff. It must be in `(0, 1]`.

The existing `expert_validation_ratio` remains independent. It continues to
define expert validation subsets and does not control loser augmentation.

## Extraction and Cache Representation

Training extraction will extract action records for both players instead of
discarding the losing player. Every record will carry a three-state
`player_result` value (`win`, `loss`, or `draw`). A boolean winner flag would
conflate losses with draws and could accidentally admit draw samples as loser
augmentation.

The packed feature cache will store the player-result code as an unsigned byte
for each sample. The cache schema and extraction schema will be incremented so
old winner-only artifacts fail with an explicit rebuild message instead of
being silently reused.

This representation requires one initial extraction and cache rebuild. After
that rebuild, changing `enabled`, `recent_dates`, or `expert_ratio` only changes
training-time indices and requires neither extraction nor caching.

## Daily Expert Cutoffs

Replay ZIP archives contain `manifest.csv`, with each two-player replay's
`min_score` and `sum_score`. For each selected date:

1. Recover the two participant scores as `min_score` and
   `sum_score - min_score`.
2. Sort all participant scores for that date in descending order.
3. Set `top_count = max(1, ceil(participant_count * expert_ratio))`.
4. Set the daily cutoff to the last score in that top slice.
5. Mark a replay score-eligible when `min_score >= cutoff`.

Using `min_score` ensures both players, including the loser, meet the daily
skill threshold. The cutoff is calculated independently for every date because
the score distribution changes over the competition. Ties at the cutoff are
included, so the realized fraction can be slightly greater than the configured
ratio.

Only dates present in the cache are considered. After reserving the latest date
for validation, the chronologically latest `recent_dates` dates are selected.

## Split Ordering and Leakage Prevention

Split construction has the following fixed order:

1. Resolve replay-level isolation validation sets.
2. Reserve all winner samples from the latest date for latest-date validation.
3. Select the replay-grouped in-distribution validation set from earlier dates.
4. Build all expert and top-deck validation masks from those winner-only
   validation samples.
5. Remove every validation replay from the training candidate pool.
6. Apply `train_replay_ratio` at replay level to the remaining training replay
   pool.
7. Include winner samples from selected training replays.
8. Additionally include non-winner samples only when all of the following hold:
   - augmentation is enabled;
   - the replay date is one of the selected recent training dates;
   - the replay satisfies `min_score >= daily cutoff`;
   - the replay is not a draw;
   - the replay does not belong to any validation split;
   - the replay passes the same `train_replay_ratio` replay selection.

Thus a selected qualifying replay contributes both its winner and loser action
records. A replay excluded by `train_replay_ratio` contributes neither side.
Validation datasets contain winner samples only, preserving their current
meaning and sample composition.

## Training Logs and WandB Metrics

At startup, print one line per augmentation date with:

```text
loser_aug_date=7.21 cutoff=1184.0 participant_scores=18420 episodes=9210 score_eligible_episodes=436 after_validation_episodes=401 selected_train_episodes=381 loser_samples=58291
```

- `score_eligible_episodes`: all non-draw replays satisfying the daily
  `min_score` threshold.
- `after_validation_episodes`: eligible replays remaining after every
  validation replay is excluded.
- `selected_train_episodes`: eligible replays remaining after replay-level
  `train_replay_ratio` sampling; these actually contribute loser samples.
- `loser_samples`: non-winner action samples actually included in training.

Print an overall summary containing the number of selected augmentation dates,
selected replay count, loser sample count, and loser sample fraction in the
final training split. Log the same counts and fraction under WandB's `data/`
namespace. No model artifact, checkpoint, extracted data, or cache is uploaded
to WandB.

When augmentation is disabled, print a concise disabled status and report zero
loser-augmentation counts.

## Error Handling

Training must fail clearly when augmentation is enabled and any of these
conditions occurs:

- fewer than `recent_dates` training dates are available after excluding the
  latest date;
- a required replay ZIP or `manifest.csv` is missing;
- manifest score fields are invalid or inconsistent;
- extracted data or cache uses the old winner-only schema;
- player-result metadata is missing or inconsistent.

Draws are stored for representation completeness but never selected as expert
loser augmentation.

## Verification

Tests will cover:

- daily cutoff computation and cutoff ties;
- selection of the most recent non-latest dates;
- both-player extraction and win/loss/draw result codes;
- cache persistence of the player-result code;
- winner-only validation subsets;
- exclusion of every validation replay from loser augmentation;
- replay-level `train_replay_ratio` consistency across both players;
- enabled and disabled augmentation modes;
- per-date and aggregate count accounting;
- rejection of old extraction and cache schemas.

Model architecture, encoded model features, optimizer behavior, validation
metrics, and checkpoint formats are outside this change.
