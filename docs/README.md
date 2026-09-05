# Project Documentation

Start with the [quick-start guide](../imitation_learning/README.md). These documents describe the current implementation, not historical development plans, migration steps, or unverified experimental conclusions.

| Guide | Contents |
|---|---|
| [Data pipeline](data-pipeline.md) | Replays, extraction, caches, formats, and rebuilding |
| [Training and validation](training-validation.md) | Modes, expert filtering, isolation, sampling, checkpoints, and troubleshooting |
| [Model architecture](model.md) | Encoder, action decoder, history, and configuration boundaries |
| [EDA](eda.md) | Entrypoints, configurations, metric definitions, and outputs |
| [Inference submission](submission.md) | Checkpoint export, Kaggle notebook, and evaluation caveats |

Run all documented commands from `imitation_learning/`. See [cfg](../imitation_learning/cfg/) for the complete parameter lists and the corresponding modules for implementation details. Preserve the YAML configuration, data range, and checkpoint for each experiment; a branch name alone is not sufficient to identify a result.
