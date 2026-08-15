# Encoder Value Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a tanh scalar value from the post-Transformer global token while initializing and fine-tuning the complete encoder from an existing policy checkpoint.

**Architecture:** Extract the current encoder forward path into a reusable method and build an encoder-only value model whose state keys map strictly to the policy checkpoint. Add encoder-only cache collation, replay-level all-outcome value splits, exact streaming regression metrics, and an isolated YAML-driven trainer with WandB and step evaluation.

**Tech Stack:** Python 3.10+, PyTorch, NumPy mmap caches, PyYAML, WandB, pytest.

## Global Constraints

- Existing policy logits and policy checkpoint loading must remain compatible.
- Value forward must not construct, move, or execute policy decoder modules.
- Targets are acting-player results: win `1.0`, draw `0.0`, loss `-1.0`.
- Validation partitions use the policy replay-allocation rules but retain all outcomes.
- The policy checkpoint is the only source of encoder architecture.
- Existing user changes in `cfg/beam_search.yaml` and `cfg/deck_strength.yaml` must remain untouched.

---

### Task 1: Reusable Encoder Forward and Encoder-Only Value Model

**Files:**
- Modify: `imitation_learning/model/network.py`
- Create: `imitation_learning/value/__init__.py`
- Create: `imitation_learning/value/model.py`
- Create: `imitation_learning/tests/test_value_model.py`

**Interfaces:**
- Add `PTCGTransformer.encode_state(...) -> torch.Tensor`, returning sequence-first post-Transformer encoder output.
- Add `PTCGEncoderBackbone.from_policy_state(config, card_features, attack_features, state_dict)`.
- Add `PTCGValueModel(backbone, head_layers, dropout)` with scalar tanh forward.

- [ ] Write a failing policy-equivalence test that compares logits before and after routing `forward()` through `encode_state`, plus value tests for global-token position 25, zero initial output, tanh bounds, absence of decoder parameter names, and strict rejection of a missing encoder tensor.
- [ ] Run `python -m pytest imitation_learning/tests/test_value_model.py -q` and confirm failures are caused by missing value interfaces.
- [ ] Move the existing encoder construction and Transformer call into `encode_state` without changing policy option/decoder code. Implement an encoder-only subclass that deletes policy-only modules before strict filtered loading, and implement the configurable value head with a zero-initialized final linear layer.
- [ ] Run the value-model test and existing `test_card_static_embedding.py`, `test_action_history_network.py`, `test_option_network.py`, `test_transformer_regularization.py`, and `test_beam_search_model_agent.py` tests.
- [ ] Commit `model/network.py`, `value/__init__.py`, `value/model.py`, and the tests as `feat: add pretrained encoder value model`.

### Task 2: Encoder-Only Cache Batches and All-Outcome Value Splits

**Files:**
- Modify: `imitation_learning/training/feature_cache.py`
- Create: `imitation_learning/value/data.py`
- Create: `imitation_learning/tests/test_value_data.py`

**Interfaces:**
- Add `EncoderBatch` containing encoder/card-summary/history tensors plus `player_result`.
- Add `MmapFeatureDataset.collate_encoder(IndexBatch) -> EncoderBatch`.
- Add `ValueSplits` and `build_value_splits(dataset, validation_ratio, validation_seed, expert_episode_keys, top_deck_keys, isolation_episode_keys)`.

- [ ] Write failing tests showing encoder collation omits every current-option/action array, maps stored player-result codes to float targets, keeps both players of a replay together, excludes latest/isolation/in-distribution validation replays from training, and retains win/draw/loss samples in validation.
- [ ] Run `python -m pytest imitation_learning/tests/test_value_data.py -q` and confirm the interfaces are missing.
- [ ] Implement encoder-only collation using the same offset arithmetic as policy collation, but copy only encoder summaries and history arrays. Implement deterministic replay-hash splits with isolation-first/latest-second precedence and aligned expert/per-Deck subgroup masks.
- [ ] Run the value-data tests and existing `test_feature_cache.py` and `test_history_feature_cache.py` tests.
- [ ] Commit the cache, value data module, and tests as `feat: add all-outcome value datasets`.

### Task 3: Exact Regression Metrics and Evaluation

**Files:**
- Create: `imitation_learning/value/metrics.py`
- Create: `imitation_learning/tests/test_value_metrics.py`

**Interfaces:**
- Add `RegressionAccumulator.update(predictions, targets)` and `.finalize() -> RegressionMetrics`.
- Add `evaluate_value_dataset(model, dataset, indices, batch_size, device, precision, subgroup_masks)`.

- [ ] Write failing tests for exact RMSE, explained variance, target/prediction means, zero-target-variance NaN, mask alignment, empty subgroup rejection, and one-forward-pass subgroup accumulation.
- [ ] Run `python -m pytest imitation_learning/tests/test_value_metrics.py -q` and confirm failures.
- [ ] Implement float64 sum/count/squared-sum accumulation so final metrics are dataset-exact rather than averages of batch metrics. Add evaluation over `collate_encoder` with simultaneous subgroup updates.
- [ ] Run the metrics tests.
- [ ] Commit as `feat: add value regression evaluation`.

### Task 4: YAML-Driven Value Trainer, WandB, and Checkpoints

**Files:**
- Create: `imitation_learning/cfg/value_train.yaml`
- Create: `imitation_learning/value/train.py`
- Create: `imitation_learning/tests/test_value_training.py`
- Modify: `imitation_learning/README.md`

**Interfaces:**
- Add strict layered settings for `value_train`, `value_model`, and `wandb`.
- Add `load_value_settings()`, `train_value_batch()`, `value_checkpoint_payload()`, and `should_trigger()`.
- Entrypoint: `python value/train.py` from `imitation_learning`.

- [ ] Write failing tests for variable expansion, required pretrained checkpoint, head/dropout validation, result targets, warmup/evaluation scheduling, one joint MSE backward pass, checkpoint architecture metadata, resume fields, and separate WandB validation namespaces.
- [ ] Run `python -m pytest imitation_learning/tests/test_value_training.py -q` and confirm failures.
- [ ] Implement configuration parsing, CG/card-table bootstrap, cache signature validation, policy-backbone loading, AdamW, mixed precision, warmup-cosine schedule, persistent EMA, replay split reporting, step/epoch saving, resume, validation, and WandB metric-only logging.
- [ ] Populate `value_train.yaml` with the current policy top Decks and isolation selections while keeping policy model architecture absent. Document execution and checkpoint behavior in README.
- [ ] Run value tests and `python -m compileall -q imitation_learning/value imitation_learning/model imitation_learning/training`.
- [ ] Commit as `feat: add value training workflow`.

### Task 5: Real Checkpoint and Cache Smoke Verification

**Files:**
- No persistent files beyond fixes required by observed failures.

**Interfaces:**
- Verify the configured policy checkpoint can initialize every encoder tensor and a real cache batch can complete value forward/backward/save/reload.

- [ ] Run all value and affected policy tests in one command.
- [ ] Run a temporary CPU or CUDA smoke using one real cache batch: load the configured policy checkpoint, assert initial values are zero, perform one optimizer update, compute latest and in-distribution RMSE/explained variance, save and reload a value checkpoint, and assert decoder parameters are absent.
- [ ] Remove temporary smoke artifacts.
- [ ] Run focused tests and compile checks again, inspect `git diff --check`, and verify only the user's pre-existing YAML edits remain unstaged.
- [ ] Commit any smoke-discovered fixes with a focused message.
