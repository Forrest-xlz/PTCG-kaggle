# Logarithmic Date Sampling Design

Extend date-weighted training sampling with a `logarithmic` mode while preserving the existing split, sample expansion, and reporting behavior.

For normalized elapsed-calendar coordinate `x` in `[0, 1]`, use:

```text
g(x) = log(1 + curvature * x) / log(1 + curvature)
w(x) = start + (end - start) * g(x)
```

The curve passes exactly through `start` at the earliest training date and `end` at the latest training date. `curvature` must be finite and strictly positive; larger values increase weights more rapidly near the beginning and flatten the curve near the end.

Configuration:

```yaml
date_sampling:
  mode: logarithmic  # linear, power, or logarithmic
  logarithmic:
    start: 0.5
    end: 1.2
    curvature: 9.0
```

Both existing curve configurations remain required and validated. Add and validate the logarithmic mapping regardless of the selected mode, matching the existing configuration policy. A single training date continues to use `end`. No extraction, cache, split, model, loss, sample-expansion, resume, or W&B behavior changes.

Tests cover exact endpoints, a known midpoint, curvature shape, invalid curvature, YAML parsing, and regression of linear and power modes.
