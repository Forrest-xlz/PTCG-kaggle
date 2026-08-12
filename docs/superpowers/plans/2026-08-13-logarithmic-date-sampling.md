# Logarithmic Date Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configurable normalized logarithmic curve to existing date-weighted training sampling.

**Architecture:** Extend `DateSamplingCurve` with `curvature`, calculate normalized `log1p(curvature*x) / log1p(curvature)`, and expose a third nested YAML curve. Existing sample expansion and training integration remain unchanged.

**Tech Stack:** Python, NumPy, PyYAML, pytest.

## Global Constraints

- Preserve exact existing linear and power behavior.
- Require finite `curvature > 0`.
- Preserve all validation, expert, cache, model, resume, and logging semantics.
- No extraction or cache rebuild.

---

### Task 1: Logarithmic curve and configuration

**Files:**
- Modify: `imitation_learning/training/date_sampling.py`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/cfg/train.yaml`
- Modify: `imitation_learning/tests/test_date_sampling.py`
- Modify: `imitation_learning/tests/test_date_sampling_integration.py`

**Interfaces:**
- Extend `DateSamplingCurve` and `DateCurveSettings` with `curvature: float | None`.
- Accept `mode="logarithmic"` in `date_weights` and training settings.

- [ ] Write failing tests asserting endpoints, `curvature=9` midpoint, single-date `end`, YAML parsing, and rejection of zero/non-finite curvature.
- [ ] Run focused tests and confirm failures refer to unsupported logarithmic mode/config.
- [ ] Implement the normalized `math.log1p` curve and nested YAML/config validation.
- [ ] Run date sampling, integration, expert signature, scheduler, and feature-cache regressions.
- [ ] Compile modified production sources and run `git diff --check`.
- [ ] Commit only the logarithmic feature and tests.
