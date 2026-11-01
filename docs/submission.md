# Kaggle submission

## Final notebook

[submission.ipynb](../submission.ipynb) is the competition submission. It loads the 3+3-layer `final_d2_33.pt` checkpoint and the 4+4-layer `final_d2_44.pt` checkpoint. The files are stored locally in `archive (3).zip` and `archive (2).zip`, respectively. Upload the checkpoint files and competition `cg` package as Kaggle Datasets, then set their paths in the notebook's parameter cell.

The notebook uses the fixed 60-card Festival Lead deck. At each decision, both models score the same legal candidates. Their FP32 softmax probabilities are averaged equally; the damage-counter KO eligibility mask is applied before choosing the highest-scoring remaining action. Running the notebook creates `/kaggle/working/submission.tar.gz`.

## Exporting another checkpoint

For a new model, edit `CHECKPOINT_PATH`, `OUTPUT_PATH`, and `PRECISION` in `imitation_learning/training/export_inference.py`. From `imitation_learning/`, run:

```bash
python training/export_inference.py
```

The exporter keeps model weights and config while removing optimizer and other training state. The resulting file is for inference, not training resume. The configurable [submission template](../imitation_learning/kaggle_submission_imitation_agent.ipynb) supports a different checkpoint or deck; it is separate from the final notebook above.

Check that checkpoint features, card IDs, engine resources, and inference paths match the Kaggle environment. Offline action CE and Top-k accuracy do not measure game win rate. A full-data model may have seen samples reconstructed by standalone validation, so those samples are not an independent holdout.
