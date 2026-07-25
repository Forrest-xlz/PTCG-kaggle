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
python -m training.train
```

Both extractors create one output shard and one metadata file per ZIP. Existing
valid shards are skipped, so interrupted runs are resumable. Training extraction
has no command-line parameters and always reads `cfg/extract.yaml`; its
`workers`, `limit_members`, and `force` fields control parallelism, smoke tests,
and rebuilding. Keep the worker count modest because ZIP decompression and JSON
parsing are both CPU- and memory-intensive.

Training has no command-line parameters. It always reads `cfg/train.yaml`, whose
`train`, `model`, and `wandb` sections control data paths, batching/preloading,
network depth and width, and experiment tracking. Decoder candidates are padded
to 64 and invalid candidates are masked out of the policy loss. Because replay
datasets can contain millions of samples, use `train.max_samples` for bounded
trials before setting it to `null` for all samples.

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
full 60-card deck, observation, and selected option indices. Invalid/empty states are counted in the sidecar
metadata rather than aborting a whole archive.
