# PTCG imitation learning

This subproject turns Kaggle replay archives into cached deck statistics and
supervised imitation-learning samples. The sparse feature encoder and
Transformer policy decoder are adapted from the MCTS sample notebook; the
value head is intentionally removed for pure behavioral cloning.

## Layout

```text
deck/       parallel deck extraction and deck EDA
model/      sparse features and the notebook transformer
training/   replay extraction and model training CLIs
data/       generated caches (gitignored)
```

## Setup

Run commands from this directory.  Python 3.10+ is required.  To construct
features, make the competition `cg` package importable (for example, add the
sample-submission directory to `PYTHONPATH`).

```bash
pip install -r requirements.txt
python -m deck.extract
python -m training.extract
python -m training.cache_features
python -m training.train
```

Both extractors create one output shard and one metadata file per ZIP. Existing
valid shards are skipped, so interrupted runs are resumable. Deck extraction
has no command-line parameters and reads `cfg/deck_extract.yaml`; its input,
output, worker count, smoke-test limit, and rebuild behavior are configured
there. Training extraction reads `cfg/extract.yaml`; its
`workers`, `limit_members`, and `force` fields control parallelism, smoke tests,
and rebuilding. Both players are written with an explicit `win`, `loss`, or
`draw` result so training can select losing-player actions without extracting
again. Replay actions are stored one step after the observation that produced them, so the
extractor pairs `steps[t]` observations with `steps[t + 1]` actions and keeps
only states whose player status is `ACTIVE`. Keep the worker count modest because ZIP decompression and JSON
parsing are both CPU- and memory-intensive.

`training.cache_features` reads `cfg/cache.yaml` and converts the two-player
JSONL records into model-ready mmap shards. `samples_per_shard` bounds the
number of samples in each physical shard; completed compatible source caches
are skipped, so cache construction is resumable. Encoder indices/offsets and
values use compact 16-bit storage where safe, decoder values are omitted
because they are always one, and only the current batch is materialized in
ordinary CPU memory. Candidate selections cover every legal size from
`maxCount` down to `minCount`, retain at most the first 64 combinations, and
treat replay selection order as irrelevant. Each sample also stores a stable
32-bit replay key used for validation splitting, a stable 64-bit key for
its complete deck, the player's result, and the acting player's previous three
actions. Cache schema 16 is required. Winner-only JSONL and older caches must
both be rebuilt once with `training.extract` followed by
`training.cache_features`.

Training has no command-line parameters. It reads `cfg/train.yaml`, whose
`train`, `model`, and `wandb` sections control cache paths, batching, network
depth/width, precision, validation, and experiment tracking. Training always
uses a compact global permutation (about 120 MB for 30 million samples), and
every eligible training sample is consumed once per epoch. Use
`train.max_samples` for bounded trials before setting it to `null`.

Training first reads the three reviewed exact-deck selections configured under
`train.isolation_validation`. It scans `data/deck/*.decks.csv`, so a selected
deck used by either player moves the entire replay into its isolation
validation set. The three isolation sets may overlap with one another, but
their replay union is removed before every later split. This lookup is
performed at training startup and does not require rebuilding feature caches.

After isolation, the numerically latest `month.day` source becomes the
latest-date validation set. Validation arrays retain winner samples only.
Older remaining replays are assigned as a group to
training or in-distribution validation using `validation_ratio` and
`validation_seed`; different states from the same replay can never cross
these splits. Every `eval_every_steps` successful optimizer updates, the
isolation union is forwarded once and accumulated into three independent
Wandb groups: `val_deck_isolation/*`, `val_archetype_isolation/*`, and
`val_top_deck_archetype_isolation/*`. Latest-date and in-distribution
validation retain their existing CE loss and Top-1/3/5 metrics. Training logs
use cross-epoch exponential moving averages controlled by `ema_alpha`.

Each replay ZIP under `train.replay_episodes` must contain one `manifest.csv`.
For every date independently, training reconstructs both player scores from
`min_score` and `sum_score`, then uses `expert_validation_ratio` to find the
top-score cutoff across all players. Ties at the cutoff are retained, and an
episode is marked expert when either player reaches it. The existing
winner-only samples from those episodes form expert subsets inside both
validation sets. Base and expert metrics share one model forward pass and are
logged separately as `val_in_distribution_expert/*` and
`val_latest_expert/*`.

`train.top_decks` accepts one or more complete 60-card lists. Card order is
ignored but multiplicity is preserved. Configuration order defines `deck1`,
`deck2`, and so on. Every deck receives separate in-distribution, latest-date,
in-distribution expert, and latest-date expert validation groups, such as
`val_in_distribution_deck1/*` and `val_in_distribution_expert_deck1/*`.
Each group contains loss and top-1/3/5 accuracy. All subgroup metrics reuse
their base validation batch's logits, so they do not add model forward passes.

After isolation, latest-date, and in-distribution validation are fixed,
`train.train_replay_ratio` selects a
deterministic fraction of the remaining replays using `train_replay_seed`.
Every selected replay keeps all of its samples, and the fixed subset is reused
for every epoch. The realized sample ratio can differ from the replay ratio
because games contain different numbers of decisions.

`train.loser_augmentation` optionally adds high-skill losing-player actions
after all validation replays are fixed. `recent_dates` selects the newest
training dates after excluding the latest-date validation date. For each date
independently, `expert_ratio` defines a participant-score cutoff; a replay's
loser is eligible only when `min_score` reaches that cutoff, which ensures both
players are above it. The same replay-level `train_replay_ratio` applies to
both sides. Draws and every replay assigned to any validation split remain
excluded. Startup logs show each date's cutoff and the replay/sample counts
after every filter stage. This ratio is independent of
`expert_validation_ratio`. After the one-time two-player extraction and
schema-16 cache rebuild, changing loser-augmentation settings requires only a
new training run.

`train.precision` accepts `fp32`, `fp16`, or `bf16`. FP16 uses autocast and
gradient scaling; BF16 uses autocast without a scaler and requires a supported
CUDA GPU. Model parameters and saved checkpoints remain FP32.

`train.cg_path` must point to the parent directory containing the competition
`cg` package. The default repository layout uses
`../pokemon_tcg_ai_battle/sample_submission`.

The root `version_name` can be reused as `${version_name}` in values such as
`train.output` and `wandb.name`. The AdamW optimizer supports configurable
`beta1`/`beta2`; its learning rate warms up linearly for `warmup_steps`
successful optimizer updates and then follows cosine decay to zero.
`log_every_steps`, `eval_every_steps`, and `save_every_steps` all use successful
optimizer steps. Epoch checkpointing remains controlled by
`save_every_epoch`.

`model.norm_mode` accepts `postnorm` or `prenorm`. PreNorm applies
normalization before every encoder/decoder sublayer and adds a final encoder
LayerNorm; PostNorm preserves the original notebook residual ordering.
`model.transformer_activation` selects `relu`, tanh-approximate `gelu`, or
`geglu` for Transformer FFNs only; all ordinary model MLPs retain ReLU.
`model.transformer_dropout` supplies one probability to four independent
switches. `dropout_embedding` applies LayerNorm and dropout to completed
encoder tokens and decoder action queries. `dropout_attention_probs` applies
dropout after attention softmax, `dropout_attention_output` applies it after
the attention output projection and before the residual, and
`dropout_ffn_output` applies it after the second FFN linear and before the
residual. The custom encoder preserves the former TransformerEncoderLayer
PreNorm/PostNorm ordering. ReLU with probability zero and all switches false
is checkpoint-compatible with the old architecture; GEGLU changes the first
FFN weight shape. These settings do not require feature-cache rebuilding.
`model.summary_mlp_layers` and `model.card_mlp_layers` control the projection
depths for numeric-summary tokens and static-card features. The first layer
maps the input width to `d_model`; additional layers are
`ReLU -> Linear(d_model, d_model)`. `model.card_mlp_scope: shared` keeps one
static-card MLP for the whole model. `model.card_mlp_scope: region` gives each
semantic card region its own MLP while sharing it among Pokemon, Tools, and
Energy cards inside that region; decoder cards reuse the corresponding encoder
region MLP. Setting `card_mlp_layers` to zero disables static-card embeddings
while preserving all learned Card ID embeddings.

The encoder has a 26-token base layout: eight bench slots per player, two
active Pokémon, three dense summary tokens, separate discard tokens for both
players, the own hand, remaining-deck estimate, and stadium. The own-player
(69), opponent-player (71), and global/select (73) numeric summaries replace
the old sparse summaries through independent `Linear(n, d_model)` projections.
Missing bench slots remain in the fixed layout but are excluded from encoder
self-attention and decoder cross-attention by a boolean key-padding mask.
Prize counts, selection type, and selection context are one-hot encoded.
`pokemon_appear_embedding` adds one shared three-state embedding (absent,
present from an earlier turn, present this turn) to the 18 Bench/Active Pokemon
tokens. Five `*_token_mlp_layers` settings control eight independent post-token
MLPs: own/opponent Bench, Active, and discard plus own hand and own deck. The
two sides share configured depths but not weights. `region_token_mlp_residual`
selects `token + MLP(token)` or `MLP(token)` globally for these modules.
Changing these features requires rebuilding the feature cache (schema 15), but
does not require replay extraction again.

The decoder stores each raw engine option once using eleven categorical fields:
option type, selection context, candidate/target Card IDs, Attack ID, number,
Energy count, player relation, area, in-play area, and special condition. Two
routed Pokemon dynamic blocks (46 values total) and six attack-matchup values
are projected separately and masked to exact zero when absent. Five remaining
numeric values (index, Tool index, Energy index, in-play index, and relative
option position) are projected by `model.option_numeric_mlp_layers`. Learned
ID and categorical embeddings, static Card/Attack projections, and these
numeric/dynamic projections are summed in `d_model` space.
`model.option_token_mlp_layers: 0`
uses that sum directly; positive values apply the standard projection MLP to
each completed option token. Exact candidate action combinations are still
enumerated up to 64, and their option tokens are summed before the
cross-attention-only decoder. Rebuild the feature cache after feature-layout
changes.

`model.history_encoding` optionally appends one action-history token to the
encoder. `basic` uses the previous three decisions' select type, select
context, and selected option types; `structural` additionally uses normalized
source/target areas, player relations, number/count, and special condition;
`full` uses independent decoder-like Card, Attack, static, and dynamic
features, while deliberately excluding all five option-position numerics.
Historical options in a combination action are summed, one shared
`history_action_mlp_layers` projection is applied at each of `[t-3,t-2,t-1]`,
and their concatenation is mapped by `history_sequence_mlp_layers` to one
token. All trainable history parameters are independent from the current-action
decoder. Switching among `basic`, `structural`, and `full` reuses the same
schema-16 cache.

When WandB is enabled, checkpoints and history are written to
`local-output/` beside that run's `files/` directory, keeping them inside the
local run-ID folder without uploading model artifacts. When WandB is disabled,
they are written under `train.output`.

To continue an interrupted run from a completed epoch checkpoint, configure:

```yaml
train:
  resume: true
  resume_checkpoint: outputs/ver_1.6.0/checkpoints/epoch-002.pt
  epochs: 5
```

Only `epoch-*.pt` training checkpoints are accepted. The checkpoint epoch is
already complete, so the example continues with epochs 3 through 5; `epochs`
is the final total rather than a number of additional epochs. Model, optimizer,
learning-rate scheduler, FP16 scaler, EMA metrics, history, and optimizer step
are restored. New checkpoints also preserve RNG state. Older epoch checkpoints
without RNG state remain usable, but their dropout sequence is not bit-for-bit
identical to an uninterrupted run. Inference-only and `step-*.pt` checkpoints
cannot be used for resume.

Open `deck/deck_eda.ipynb` after deck extraction. Set the extracted-deck and
`EN_Card_Data.csv` paths in its setup cell, then run top-to-bottom. The
notebook classifies rule-based archetypes, assigns stable SHA-256 exact-deck
IDs, and saves:

- `data/deck_analysis/deck_summary.csv`
- `data/deck_analysis/deck_similarity_pairs.csv`

The similarity table contains every unordered exact-deck pair. It reports the
minimum changed card slots and count-aware Weighted Jaccard similarity.

Open `eda/replay_timing.ipynb` to analyze agent startup time and mean
subsequent-action time from the numerically latest dated replay ZIP. Configure
`SCORE_MODE` (`avg`, `min`, or `max`) and `SCORE_THRESHOLD` in the parameter
cell. The notebook caches replay-player timings and exports the team-level
analysis under `data/replay_timing/`, then plots timing distributions and four
global K-Means timing clusters.

To analyze Deck usage, matchup, and team-switching trends, first build the
incremental per-date cache and then open the trend notebook:

```bash
python imitation_learning/deck/trend_extract.py
python -m jupyter notebook imitation_learning/eda/deck_trend.ipynb
```

Configure extraction and analysis in `cfg/deck_trend.yaml`. Changing the date
interval, chart threshold, mirror handling, or a Deck-Card-ID-only archetype
classifier requires only rerunning the notebook. Rerun the extractor after
adding or replacing replay ZIP archives; with `force: false`, unchanged complete
date shards are reused.

Before Kaggle submission, edit `CHECKPOINT_PATH`, `OUTPUT_PATH`, and
`PRECISION` at the top of `training/export_inference.py`, then strip the
optimizer and other training-only state:

```bash
python training/export_inference.py
```

The command writes `epoch-003.inference-fp16.pt` beside the source checkpoint
when `OUTPUT_PATH` is `None`, and prints both sizes. `PRECISION` accepts
`fp16`, `bf16`, or `fp32`. Upload this inference checkpoint as a Kaggle
Dataset, then attach it to
`kaggle_submission_imitation_agent.ipynb` together with a Dataset containing
the `cg` directory. In the first code cell, set the exact `MODEL_PATH`,
`CG_PATH`, and the agent's 60-card `DECK`, then run all cells.
The notebook reads width, FFN size, attention heads, encoder/decoder depth,
normalization mode, static-card projections, and action-history mode from the checkpoint;
these architecture fields are not configured twice. It embeds the inference
code, including the 26-token base encoder layout and optional history token, and creates
`/kaggle/working/submission.tar.gz`.

## Training records

Each deck cache contains only two rows per replay: the two complete sorted
60-card lists plus the final reward/result needed by deck win-rate EDA. It
does not save observations, steps, actions, or per-card rows.

Each line in `data/training/<date>.jsonl.gz` contains the episode id, player,
full 60-card deck, active observation, and the selected option indices recorded
in the following replay step. Extraction schema 3 is required by the cache
builder; older JSONL files and all feature caches built from them must be
rebuilt.
