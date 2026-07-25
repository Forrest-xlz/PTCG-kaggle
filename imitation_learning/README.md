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
python -m deck.extract --input "../replay episodes" --output data/decks
python -m training.extract
python -m training.cache_features
python -m training.train
```

Both extractors create one output shard and one metadata file per ZIP. Existing
valid shards are skipped, so interrupted runs are resumable. Training extraction
has no command-line parameters and always reads `cfg/extract.yaml`; its
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
treat replay selection order as irrelevant.

Training has no command-line parameters. It reads `cfg/train.yaml`, whose
`train`, `model`, and `wandb` sections control cache paths, batching, network
depth/width, precision, shuffle, and experiment tracking. `shuffle_mode:
global` builds a compact global uint32 permutation (about 120 MB for 30 million
samples); `shuffle_mode: shard` shuffles physical shard order and samples
within each shard for better storage locality. In either mode, every cache
sample is consumed once per global epoch. Use `train.max_samples` for bounded
trials before setting it to `null`.

`train.precision` accepts `fp32`, `fp16`, or `bf16`. FP16 uses autocast and
gradient scaling; BF16 uses autocast without a scaler and requires a supported
CUDA GPU. Model parameters and saved checkpoints remain FP32.

`train.cg_path` must point to the parent directory containing the competition
`cg` package. The default repository layout uses
`../pokemon_tcg_ai_battle/sample_submission`.

The root `version_name` can be reused as `${version_name}` in values such as
`train.output` and `wandb.name`. The AdamW optimizer supports configurable
`beta1`/`beta2`; its learning rate warms up linearly for `warmup_ratio` of all
optimizer steps and then follows cosine decay to zero. `log_every_steps` controls
the WandB/console logging interval in optimizer steps rather than samples.

Open `deck/deck_eda.ipynb` after deck extraction. It ranks complete deck types
by usage, analyzes win rates, and plots usage and win-rate trends by date.

For Kaggle submission, upload a trained `epoch-*.pt` file as a Kaggle Dataset,
attach it to `kaggle_submission_imitation_agent.ipynb` together with a Dataset
containing the `cg` directory, and run all cells. The notebook embeds the inference code and deck list and creates
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
