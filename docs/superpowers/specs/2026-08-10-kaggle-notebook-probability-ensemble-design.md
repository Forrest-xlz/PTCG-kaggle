# Kaggle Submission Notebook Probability Ensemble Design

## Goal

Extend `imitation_learning/kaggle_submission_imitation_agent.ipynb` with an
optional equal-weight probability ensemble while preserving its generated
`main.py`, deck behavior, and Kaggle submission format.

## Notebook Parameters and Packaging

The parameter cell exposes:

```python
ENSEMBLE_ENABLED = True
MODEL_PATHS = [Path("/kaggle/input/model-a/model.pt"),
               Path("/kaggle/input/model-b/model.pt")]
```

Disabled mode requires exactly one path. Enabled mode requires at least two
distinct paths. The parameter cell previews and validates every checkpoint.
The packaging cell copies models into the archive as `model-001.pt`,
`model-002.pt`, and so on, and writes `model_manifest.json` containing the
enabled flag and ordered archive names. It reports the final compressed size
and visibly warns when it exceeds 197.7 MiB.

## Runtime Loading and Compatibility

Generated `main.py` reads the manifest with the existing asset lookup. Every
checkpoint independently restores its `ModelConfig` and weights. Models may
differ in learned architecture settings such as width, depth, normalization,
dropout, and projection depth, but must share the feature/action contract used
to construct one observation: `card_count`, `attack_count`, and
`encoder_size`. Incompatible checkpoints fail during agent initialization.

Static Card/Attack tables and the numeric catalog are built once using the
shared vocabulary sizes, then reused to construct every model. All models stay
resident on CPU because the competition engine executes the packaged agent on
CPU.

## Probability Averaging

For each decision, build encoder, decoder, and history tensors once. Run each
model on those same tensors under inference mode, cast logits to FP32, apply
softmax across the enumerated legal actions, and average probabilities with
equal weight. Select the action with the highest averaged probability. There
is no padded invalid-action region in this submission path because logits are
created only for the enumerated legal actions.

Disabled mode retains the current single-model `argmax(logits)` decision,
which is equivalent to `argmax(softmax(logits))` and avoids unnecessary work.

## Validation

Preserve notebook cell order and unrelated implementation text. Validate the
notebook with `nbformat`, compile the generated `main.py` source after removing
the `%%writefile` directive, verify manifest/archive references, and run
targeted static assertions for strict cardinality and probability averaging.
Full execution requires Kaggle-mounted checkpoint and `cg` paths and therefore
is not expected locally unless those assets exist.
