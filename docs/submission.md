# Inference Export and Kaggle Submission

## Export an inference checkpoint

Edit the constants at the top of `imitation_learning/training/export_inference.py`:

- CHECKPOINT_PATH: an existing, trusted training checkpoint.
- OUTPUT_PATH: destination; None generates a default inference filename beside the source.
- PRECISION: fp16, bf16, or fp32.

Run from imitation_learning:

```bash
python training/export_inference.py
```

Export retains model and config, removes training-only state such as the optimizer, and converts weight precision. The script reports source and destination sizes. Example paths do not imply a checkpoint exists locally; update them before running. Inference files cannot be used for training resume.

## Kaggle notebook

Use [kaggle_submission_imitation_agent.ipynb](../imitation_learning/kaggle_submission_imitation_agent.ipynb):

1. Upload the exported model as a Kaggle Dataset.
2. Prepare a Dataset containing the competition cg directory.
3. Attach both Datasets to the notebook.
4. Set MODEL_PATH, CG_PATH, and the complete 60-card DECK in its initial configuration.
5. Run cells in order to generate `/kaggle/working/submission.tar.gz`.

Width, depth, normalization, static-card projections, and action-history settings are reconstructed from checkpoint config rather than configured twice. The submission notebook embeds inference code, so verify compatibility after changing training-side features or architecture.

## Checks and interpretation

- Load only trusted checkpoints; PyTorch files may contain executable pickle payloads.
- Verify card IDs, engine resources, model paths, and inference precision against the target environment.
- Offline CE/Top-k accuracy is not game win rate; inference and game-level checks are still needed.
- A full_data model may have seen samples reconstructed by standalone validation; those results are not independent test performance.
- This document does not claim a final competition ranking for any notebook or checkpoint.
