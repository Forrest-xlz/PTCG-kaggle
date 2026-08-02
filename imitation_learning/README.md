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
and rebuilding. With `winner_only: true`, only decisions made by players whose
final replay reward is positive are written; draws produce no samples. Replay
actions are stored one step after the observation that produced them, so the
extractor pairs `steps[t]` observations with `steps[t + 1]` actions and keeps
only states whose player status is `ACTIVE`. Keep the worker count modest because ZIP decompression and JSON
parsing are both CPU- and memory-intensive.

`training.cache_features` reads `cfg/cache.yaml` and converts the winner-only
JSONL records into model-ready mmap shards. `samples_per_shard` bounds the
number of samples in each physical shard; completed compatible source caches
are skipped, so cache construction is resumable. Encoder indices/offsets and
values use compact 16-bit storage where safe, decoder values are omitted
because they are always one, and only the current batch is materialized in
ordinary CPU memory. Candidate selections cover every legal size from
`maxCount` down to `minCount`, retain at most the first 64 combinations, and
treat replay selection order as irrelevant. Each sample also stores a stable
32-bit replay key used for validation splitting and a stable 64-bit key for
its complete deck. Cache schema 4 is required, so older feature caches must be
rebuilt; replay extraction does not need to be repeated.

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
latest-date validation set. Older remaining replays are assigned as a group to
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
ignored but multiplicity is preserved. These decks define
`val_in_distribution_top_deck`, `val_latest_top_deck`, and the intersection
`val_latest_expert_top_deck`. All subgroup metrics reuse their base validation
batch's logits, so they do not add model forward passes.

After isolation, latest-date, and in-distribution validation are fixed,
`train.train_replay_ratio` selects a
deterministic fraction of the remaining replays using `train_replay_seed`.
Every selected replay keeps all of its samples, and the fixed subset is reused
for every epoch. The realized sample ratio can differ from the replay ratio
because games contain different numbers of decisions.

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
`model.card_feature_ratio` sets the active width of the globally shared
`Linear(54, k)` static-card projection, where
`k = int(d_model * card_feature_ratio)`. The remaining dimensions are zero
padded before the projection is added to every Card ID embedding.

The encoder has a fixed 20-token layout: five bench slots per player, two
active Pokémon, three dense summary tokens, separate discard tokens for both
players, the own hand, remaining-deck estimate, and stadium. The own-player
(60), opponent-player (62), and global/select (73) numeric summaries replace
the old sparse summaries through independent `Linear(n, d_model)` projections.
Prize counts, selection type, and selection context are one-hot encoded.
Changing this layout requires rebuilding the feature cache (schema 7), but
does not require replay extraction again.

The decoder stores each raw engine option as 11 compact categorical IDs and
three scalars. Low-cardinality states expand to a 28-dimensional scalar/one-hot
vector during forward. Encoder tokens and options share learned location
embeddings, so spatial options can identify their candidate and target tokens.
Learned ID embeddings, the shared static-card projection, auxiliary projection,
and static-attack projection are added in `d_model` space. Exact candidate
action combinations are still enumerated up to 64, and their selected option
embeddings are summed before the cross-attention-only decoder. The empty
combination uses a learned no-action embedding. This decoder change requires
rebuilding only the feature cache; existing winner-only extracted JSONL files
remain valid.

When WandB is enabled, checkpoints and history are written to
`local-output/` beside that run's `files/` directory, keeping them inside the
local run-ID folder without uploading model artifacts. When WandB is disabled,
they are written under `train.output`.

Open `eda/deck.ipynb` after deck extraction. Set the extracted-deck and
`EN_Card_Data.csv` paths in its setup cell, then run top-to-bottom. The
notebook classifies rule-based archetypes, assigns stable SHA-256 exact-deck
IDs, and saves:

- `data/deck_analysis/deck_summary.csv`
- `data/deck_analysis/deck_similarity_pairs.csv`

The similarity table contains every unordered exact-deck pair. It reports the
minimum changed card slots and count-aware Weighted Jaccard similarity.

Open `eda/replay_timing.ipynb` to analyze the newest dated replay archive.
Configure `SCORE_MODE` (`avg`, `min`, or `max`) and `SCORE_THRESHOLD` in the
parameter cell. The notebook caches replay-player timings and exports the
team-level analysis under `data/replay_timing/`, then plots timing
distributions and four global K-Means timing clusters.

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
normalization mode, and the static-card projection ratio from the checkpoint;
these architecture fields are not configured twice. It embeds the inference
code, including the fixed 20-token numeric-summary layout, and creates
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
