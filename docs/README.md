# Project Documentation

Start with the [quick-start guide](../imitation_learning/README.md) to prepare data and run training.

The [solution write-up](../kaggle_writeup.md) contains the competition reasoning, figures, and experiment results. The [experiment branch map](experiment-branches.md) links the retained code snapshots to all 22 W&B runs.

| Guide | Contents |
|---|---|
| [Data pipeline](data-pipeline.md) | Replays, extraction, caches, formats, and rebuilding |
| [Training and validation](training-validation.md) | Modes, expert filtering, isolation, sampling, checkpoints, and troubleshooting |
| [Model architecture](model.md) | Encoder, action decoder, history, and configuration boundaries |
| [EDA](eda.md) | Entrypoints, configurations, metric definitions, and outputs |
| [Inference submission](submission.md) | Checkpoint export, Kaggle notebook, and evaluation caveats |

Run the documented commands from `imitation_learning/`. Configuration files are in [cfg](../imitation_learning/cfg/). For an experiment, save its YAML, replay range, and checkpoint together; a branch name alone does not identify a trained model.
