# Validation Probability Ensemble Design

## Goal

Extend the standalone validation evaluator with an optional equal-weight
probability ensemble while preserving single-checkpoint behavior and all
existing validation splits and metrics.

## Configuration

Replace the singular checkpoint value with an ensemble mapping:

```yaml
validation:
  train_config: cfg/train.yaml
  ensemble:
    enabled: true
    checkpoints:
      - outputs/model_a/checkpoints/epoch-005.pt
      - outputs/model_b/checkpoints/epoch-005.pt
  batch_size: 4096
  device: cuda
  precision: bf16
```

When `enabled` is false, `checkpoints` must contain exactly one path. When it
is true, the list must contain at least two distinct paths. Relative paths and
`${version_name}` interpolation retain their current behavior.

## Model Loading and Compatibility

Each checkpoint independently reconstructs `ModelConfig` and
`PTCGTransformer`, loads weights strictly, and moves to the configured device.
Architectural widths, depths, normalization modes, and projection settings may
differ. All checkpoints must nevertheless produce the same action space and
must have identical cache feature signatures, including Card/Attack vocabulary
sizes, encoder vocabulary size, cache schema, and action enumeration. Fail
before opening evaluation loops when compatibility checks do not pass.

All models remain resident on the selected device during evaluation. This
maximizes throughput and makes GPU memory approximately the sum of model
parameter and inference-activation requirements.

## Probability Averaging

For each validation batch, run every model on the same cached features. For
each model independently, mask positions at or above that sample's legal action
count and apply softmax in FP32. Average the resulting legal-action
probabilities with equal weight. Compute CE as negative log likelihood of the
averaged probability assigned to the replay action, and compute Top-1/3/5 from
the averaged probabilities.

Single-model mode retains the existing logits-based metric path exactly. The
isolation, latest-date, and in-distribution subgroup masks continue to reuse
their parent batch predictions without additional forwards.

## Output and Scope

Startup output identifies whether ensemble mode is active, lists the resolved
checkpoint paths, and prints each automatically recovered architecture. Metric
names and validation-set construction remain unchanged. The evaluator remains
read-only and does not modify training, model, feature, cache, or checkpoint
files.

## Tests

Add focused tests for configuration cardinality and path ordering, compatible
and incompatible checkpoint signatures, legal-action probability averaging,
CE and Top-K values, one forward per model per batch, and preservation of the
single-model metric path.
