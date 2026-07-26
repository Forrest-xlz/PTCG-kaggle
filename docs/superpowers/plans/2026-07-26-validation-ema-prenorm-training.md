# Validation, EMA, PreNorm, and Training Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add replay-grouped and latest-date validation, global training EMA and Top-k metrics, step-based evaluation/checkpointing, step-count warmup, and configurable PreNorm/PostNorm while removing shard shuffle.

**Architecture:** Extend each packed sample with a stable episode key and derive three mmap-backed index collections at train startup. Keep metric/schedule/checkpoint helpers as small pure functions in `training/train.py`, while `feature_cache.py` owns packed storage, source-date parsing, split construction, and global-only batching. The model configuration drives identical encoder/decoder normalization in local training and the Kaggle notebook.

**Tech Stack:** Python 3.10+, NumPy mmap arrays, PyTorch, PyYAML, WandB, pytest.

## Global Constraints

- Split only at replay level; no episode may cross train and validation.
- The numerically latest `month.day` source is fully held out.
- In-distribution validation is a deterministic approximate ratio configured in `train.yaml`.
- Validation always evaluates both complete validation sets and logs separate WandB namespaces.
- WandB uploads metrics only; checkpoints remain local.
- Training uses only global shuffle.
- Decoder candidates never self-attend.
- Existing replay JSONL files remain valid; packed feature caches must be rebuilt.

---

### Task 1: Persist Stable Episode Keys in Packed Cache

**Files:**
- Modify: `imitation_learning/training/feature_cache.py`
- Modify: `imitation_learning/training/cache_features.py`
- Modify: `imitation_learning/tests/test_feature_cache.py`

**Interfaces:**
- Produces: `stable_episode_key(episode_id: object) -> int`
- Produces: `FeatureRecord.episode_key: int`
- Produces: `FeatureView.episode_key: int`
- Produces: packed section `episode_key` with dtype `<u4`

- [ ] **Step 1: Write the failing cache round-trip and stable-hash tests**

```python
from training.feature_cache import stable_episode_key

def test_episode_key_is_stable_for_equivalent_ids() -> None:
    assert stable_episode_key(12345) == stable_episode_key("12345")
    assert 0 <= stable_episode_key("episode-a") <= np.iinfo(np.uint32).max

def test_packed_shard_round_trip(tmp_path: Path) -> None:
    path = build_shard(tmp_path / "part.cache", [7])
    shard = PackedShard(path, expected_signature=SIGNATURE)
    try:
        assert shard.sample(0).episode_key == stable_episode_key("episode-7")
        assert shard.arrays["episode_key"].dtype == np.dtype("<u4")
    finally:
        shard.close()
```

Update the test `record()` helper to set
`episode_key=stable_episode_key(f"episode-{marker}")`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
cd imitation_learning
pytest tests/test_feature_cache.py -q
```

Expected: failure because `stable_episode_key`, the record field, and packed
section do not exist.

- [ ] **Step 3: Implement stable hashing and the packed field**

Use a process-independent four-byte BLAKE2s digest:

```python
def stable_episode_key(episode_id: object) -> int:
    payload = str(episode_id).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2s(payload, digest_size=4).digest(), "little"
    )
```

Add `episode_key` to `SECTION_DTYPES`, `FeatureRecord`, `FeatureView`, writer
validation/appending, reader length validation, and `sample()`. Bump
`CACHE_SCHEMA_VERSION` from 2 to 3. In `cache_features._prepare_record`, set
the field from `record["episode_id"]`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
pytest tests/test_feature_cache.py -q
```

Expected: all feature-cache tests pass.

- [ ] **Step 5: Commit**

```bash
git add imitation_learning/training/feature_cache.py imitation_learning/training/cache_features.py imitation_learning/tests/test_feature_cache.py
git commit -m "feat: persist replay keys in feature cache"
```

### Task 2: Build Replay-Grouped Temporal Splits and Remove Shard Shuffle

**Files:**
- Modify: `imitation_learning/training/feature_cache.py`
- Modify: `imitation_learning/tests/test_feature_cache.py`

**Interfaces:**
- Produces: `parse_source_date(name: str) -> tuple[int, int]`
- Produces: `DatasetSplits(train, in_distribution, latest)` containing NumPy global-index arrays
- Produces: `MmapFeatureDataset.build_splits(validation_ratio: float, validation_seed: int) -> DatasetSplits`
- Changes: `iter_index_batches(indices, batch_size, seed, shuffle, max_samples=None)`
- Changes: `iter_batches(indices, batch_size, seed, shuffle, max_samples=None)`

- [ ] **Step 1: Write failing date, grouping, and global iteration tests**

Build three tiny shards with source names `7.5.jsonl.gz`,
`7.19.jsonl.gz`, and `7.24.jsonl.gz`; repeat episode keys within each shard.

```python
def test_source_dates_sort_numerically() -> None:
    assert parse_source_date("7.19.jsonl.gz") > parse_source_date("7.5.jsonl.gz")
    assert parse_source_date("7.1.part-00000.cache") == (7, 1)

def test_splits_hold_out_latest_and_keep_episodes_together(tmp_path: Path) -> None:
    dataset = build_dated_dataset(tmp_path)
    splits = dataset.build_splits(validation_ratio=0.5, validation_seed=9)
    latest_ids = set(splits.latest.tolist())
    assert latest_ids == set(dataset.global_ids_for_date((7, 24)).tolist())
    assert not latest_ids & set(splits.train.tolist())
    assert not latest_ids & set(splits.in_distribution.tolist())
    for episode_key in dataset.unique_episode_keys():
        memberships = dataset.memberships_for_key(episode_key, splits)
        assert sum(bool(group) for group in memberships) == 1

def test_global_indices_visit_each_selected_sample_once(tmp_path: Path) -> None:
    indices = np.asarray([0, 2, 4, 5], dtype=np.uint32)
    batches = dataset.iter_index_batches(
        indices, batch_size=2, seed=123, shuffle=True
    )
    visited = np.concatenate([batch.global_ids for batch in batches])
    np.testing.assert_array_equal(np.sort(visited), np.sort(indices))
```

Delete the shard-shuffle test and remove `shuffle_mode` from existing calls.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest tests/test_feature_cache.py -q
```

Expected: missing split/date APIs and old iterator signature failures.

- [ ] **Step 3: Implement date parsing, split hashing, and global-only iteration**

Extract the first `month.day` pair from source names with a strict regular
expression. Reject month outside 1–12 and day outside 1–31.

Mix the cached key and validation seed with a vectorized `uint32` avalanche
function, then compare against:

```python
threshold = int(validation_ratio * (1 << 32))
```

Construct compact train/in-distribution/latest global IDs shard by shard.
Latest-date shards go only to `latest`. Older shards are partitioned by the
mixed key. Raise `ValueError` if any collection is empty.

Remove the shard branch and `shuffle_mode` parameter. Copy the selected
indices only when shuffling so validation traversal remains stable; shuffle
training indices with `np.random.default_rng(seed).shuffle(order)`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
pytest tests/test_feature_cache.py -q
```

Expected: all feature-cache tests pass and no shard-mode test remains.

- [ ] **Step 5: Commit**

```bash
git add imitation_learning/training/feature_cache.py imitation_learning/tests/test_feature_cache.py
git commit -m "feat: add replay grouped validation splits"
```

### Task 3: Add EMA and Masked CE/Top-k Metric Helpers

**Files:**
- Create: `imitation_learning/tests/test_training_metrics.py`
- Modify: `imitation_learning/training/train.py`

**Interfaces:**
- Produces: `ExponentialMovingAverage(alpha: float)`
- Produces: `PolicyMetrics(loss: float, top1_correct: int, top3_correct: int, top5_correct: int, samples: int)`
- Produces: `policy_metrics(logits, targets, action_counts) -> tuple[Tensor, PolicyMetrics]`

- [ ] **Step 1: Write failing EMA and Top-k tests**

```python
def test_ema_initializes_from_first_value_and_never_resets() -> None:
    ema = ExponentialMovingAverage(alpha=0.99)
    assert ema.update(2.0) == pytest.approx(2.0)
    assert ema.update(1.0) == pytest.approx(1.99)
    assert ema.update(0.0) == pytest.approx(1.9701)

def test_policy_metrics_mask_invalid_actions_and_compute_topk() -> None:
    logits = torch.tensor([
        [5.0, 4.0, 3.0, 100.0, 100.0],
        [5.0, 4.0, 3.0, 2.0, 1.0],
    ])
    targets = torch.tensor([2, 3])
    action_counts = torch.tensor([3, 5])
    loss, metrics = policy_metrics(logits, targets, action_counts)
    assert torch.isfinite(loss)
    assert metrics.samples == 2
    assert metrics.top1_correct == 0
    assert metrics.top3_correct == 1
    assert metrics.top5_correct == 2
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest tests/test_training_metrics.py -q
```

Expected: import failures for the new helpers.

- [ ] **Step 3: Implement the minimal metric helpers**

Mask positions at or beyond each `action_count` with the dtype minimum before
cross-entropy. Compute each Top-k with `k=min(k, logits.shape[1])`; masked
invalid positions cannot outrank legal logits. Store correct counts rather
than pre-averaged ratios so callers can aggregate by sample.

EMA validates `0 <= alpha < 1`, initializes directly from the first value, and
retains state until training ends.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
pytest tests/test_training_metrics.py -q
```

Expected: all metric tests pass.

- [ ] **Step 5: Commit**

```bash
git add imitation_learning/training/train.py imitation_learning/tests/test_training_metrics.py
git commit -m "feat: add EMA and policy top-k metrics"
```

### Task 4: Replace Warmup Ratio and Add Training Trigger Helpers

**Files:**
- Create: `imitation_learning/tests/test_training_schedule.py`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/training/precision.py`

**Interfaces:**
- Changes: `build_lr_scheduler(optimizer, total_steps: int, warmup_steps: int)`
- Produces: `should_trigger(interval: int, global_step: int) -> bool`
- Produces: `checkpoint_payload(...) -> dict`
- Produces: `PrecisionContext.state_dict() -> dict`

- [ ] **Step 1: Write failing schedule and trigger tests**

```python
def test_warmup_then_cosine_reaches_boundaries() -> None:
    optimizer = AdamW([torch.nn.Parameter(torch.zeros(()))], lr=1.0)
    scheduler = build_lr_scheduler(optimizer, total_steps=6, warmup_steps=2)
    assert learning_rates_for_six_successful_steps(optimizer, scheduler) == pytest.approx(
        [0.5, 1.0, 0.75, 0.25, 0.0, 0.0], abs=1e-7
    )

def test_step_trigger_uses_successful_optimizer_steps() -> None:
    assert should_trigger(100, 100)
    assert not should_trigger(100, 99)
```

The exact cosine expectations may be expressed directly from the documented
formula as long as the second warmup update equals 1.0 and the final planned
update equals 0.0.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest tests/test_training_schedule.py -q
```

Expected: old ratio signature and missing trigger helper.

- [ ] **Step 3: Implement the step-count schedule and serializable precision state**

Validate `0 <= warmup_steps < total_steps`. Use a LambdaLR multiplier indexed
by completed successful updates, with explicit boundary tests preventing
off-by-one behavior. Return only the scheduler; the configured warmup count is
already known.

Expose the GradScaler state through:

```python
def state_dict(self) -> dict:
    return self.scaler.state_dict() if self.scaler.is_enabled() else {}
```

Checkpoint payload includes model, optimizer, scheduler, scaler, global step,
epoch, EMA values, configurations, and history.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
pytest tests/test_training_schedule.py tests/test_mixed_precision.py -q
```

Expected: all schedule and precision tests pass.

- [ ] **Step 5: Commit**

```bash
git add imitation_learning/training/train.py imitation_learning/training/precision.py imitation_learning/tests/test_training_schedule.py
git commit -m "feat: use step based training schedule"
```

### Task 5: Implement Configurable PreNorm and PostNorm

**Files:**
- Modify: `imitation_learning/model/network.py`
- Modify: `imitation_learning/tests/test_mixed_precision.py`

**Interfaces:**
- Adds: `ModelConfig.norm_mode: str = "postnorm"`
- Changes: `DecoderLayer(..., norm_mode: str)`
- Simplifies: `PTCGTransformer.forward(index_encoder, value_encoder, offset_encoder, index_decoder, offset_decoder)`

- [ ] **Step 1: Write failing normalization construction and forward tests**

```python
@pytest.mark.parametrize("mode", ["prenorm", "postnorm"])
def test_normalization_modes_preserve_policy_shape(mode: str) -> None:
    model = tiny_model(norm_mode=mode).eval()
    logits = model(
        encoder_index, encoder_value, encoder_offset,
        decoder_index, decoder_offset,
    )
    assert logits.shape == (1, 2)
    assert model.encoder.layers[0].norm_first is (mode == "prenorm")
    assert (model.encoder.norm is not None) is (mode == "prenorm")

def test_invalid_normalization_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="norm_mode"):
        PTCGTransformer(tiny_config(norm_mode="sandwich"))
```

Add a decoder hook test or direct sublayer test showing PreNorm feeds
`norm1(x)` into cross-attention while PostNorm feeds `x`.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest tests/test_mixed_precision.py -q
```

Expected: missing `norm_mode`, unchanged forward signature, and no final
PreNorm encoder norm.

- [ ] **Step 3: Implement both normalization paths**

Pass `norm_first=norm_mode == "prenorm"` to
`TransformerEncoderLayer`. Give `TransformerEncoder` a final LayerNorm only
for PreNorm.

Implement decoder branches exactly as:

```python
if self.prenorm:
    query = self.norm1(x)
    attended, _ = self.attention(query, encoder_out, encoder_out, need_weights=False)
    x = x + attended
    return x + self.fc2(F.relu(self.fc1(self.norm2(x))))
attended, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
x = self.norm1(x + attended)
return self.norm2(x + self.fc2(F.relu(self.fc1(x))))
```

Remove the explicit decoder-value compatibility branch from the local model.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
pytest tests/test_mixed_precision.py -q
```

Expected: all normalization and precision tests pass.

- [ ] **Step 5: Commit**

```bash
git add imitation_learning/model/network.py imitation_learning/tests/test_mixed_precision.py
git commit -m "feat: configure transformer normalization"
```

### Task 6: Integrate Validation, EMA, Evaluation, and Checkpoint Routing

**Files:**
- Create: `imitation_learning/tests/test_training_integration.py`
- Modify: `imitation_learning/training/train.py`

**Interfaces:**
- Produces: `evaluate_dataset(...) -> PolicyMetrics`
- Produces: `resolve_output_root(train_cfg, wandb_run) -> Path`
- Uses: `DatasetSplits`, EMA helpers, Top-k helpers, and successful `global_step`

- [ ] **Step 1: Write failing validation and output routing tests**

```python
def test_output_root_prefers_wandb_run_directory(tmp_path: Path) -> None:
    run = SimpleNamespace(dir=str(tmp_path / "wandb-run-files"))
    assert resolve_output_root(train_cfg(output=tmp_path / "fallback"), run) == Path(run.dir)

def test_output_root_uses_config_when_wandb_disabled(tmp_path: Path) -> None:
    assert resolve_output_root(train_cfg(output=tmp_path / "fallback"), None) == tmp_path / "fallback"

def test_evaluation_aggregates_topk_over_complete_split() -> None:
    metrics = evaluate_dataset(
        deterministic_model(), tiny_dataset(), validation_indices(),
        batch_size=2, device=torch.device("cpu"), precision=fp32_context(),
    )
    assert metrics.samples == len(validation_indices())
    assert metrics.top1_correct <= metrics.top3_correct <= metrics.top5_correct
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest tests/test_training_integration.py -q
```

Expected: missing evaluation and routing helpers.

- [ ] **Step 3: Implement evaluation and restructure `main()`**

Load settings, create the dataset, build splits, and calculate
`steps_per_epoch` from the training collection only. Initialize WandB before
selecting the output root. Define WandB metrics so all train, epoch, and
validation series use `optimizer_step` as their step metric.

Use one persistent EMA object for loss and each Top-k ratio. Keep exact
sample-weighted epoch counters. At each successful `global_step`:

1. log EMA metrics when `log_every_steps` divides the step;
2. evaluate and separately log both validation groups when
   `eval_every_steps` divides the step;
3. save a step checkpoint when `save_every_steps` divides the step.

Always evaluate at completion unless the final step was already evaluated.
Restore training mode after evaluation. Save epoch checkpoints when enabled.
Write resolved YAML/JSON configuration and `history.json` under the chosen
output root. Never call `wandb.Artifact`, `wandb.save`, or model logging.

- [ ] **Step 4: Run focused and full tests**

Run:

```bash
pytest tests/test_training_integration.py -q
pytest -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add imitation_learning/training/train.py imitation_learning/tests/test_training_integration.py
git commit -m "feat: evaluate replay validation sets during training"
```

### Task 7: Update Configuration, Submission Notebook, and Documentation

**Files:**
- Modify: `imitation_learning/cfg/train.yaml`
- Modify: `imitation_learning/kaggle_submission_imitation_agent.ipynb`
- Modify: `imitation_learning/README.md`

**Interfaces:**
- Documents the final YAML fields and mandatory cache rebuild.
- Keeps Kaggle inference architecture compatible with `norm_mode`.

- [ ] **Step 1: Write a failing notebook/config validation test**

Create `imitation_learning/tests/test_artifacts.py`:

```python
def test_train_yaml_uses_step_and_validation_configuration() -> None:
    cfg = yaml.safe_load(TRAIN_YAML.read_text(encoding="utf-8"))
    train = cfg["train"]
    assert "shuffle_mode" not in train
    assert "warmup_ratio" not in train
    assert train["warmup_steps"] >= 0
    assert train["validation_ratio"] == pytest.approx(0.05)
    assert cfg["model"]["norm_mode"] in {"prenorm", "postnorm"}

def test_submission_notebook_contains_both_normalization_modes() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )
    assert "norm_mode" in source
    assert "norm_first" in source
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest tests/test_artifacts.py -q
```

Expected: old YAML fields and missing notebook normalization support.

- [ ] **Step 3: Update YAML, notebook, and README**

Apply the exact fields from the approved design. In the notebook:

- add `norm_mode="postnorm"` to its `ModelConfig`;
- construct encoder `norm_first` and final norm consistently;
- implement both decoder residual paths;
- load old checkpoint config with `norm_mode` defaulting to `postnorm`;
- keep its explicit sparse decoder value tensor because notebook inference
  builds one directly.

README must state:

- packed caches need rebuilding for `episode_key`;
- latest-date and replay-grouped validation semantics;
- global-only shuffle;
- EMA and validation metric namespaces;
- step warmup/evaluation/save settings;
- WandB/local output routing;
- PreNorm/PostNorm behavior.

- [ ] **Step 4: Run all verification**

Run:

```bash
pytest -q
python -m json.tool kaggle_submission_imitation_agent.ipynb > /dev/null
python -m training.cache_features
```

For the cache smoke validation, temporarily point `cfg/cache.yaml` at the
existing smoke extracted data if the full input is too large, then restore the
checked-in configuration before committing. Inspect one new shard and confirm
`episode_key` exists with dtype `uint32`.

- [ ] **Step 5: Commit**

```bash
git add imitation_learning/cfg/train.yaml imitation_learning/kaggle_submission_imitation_agent.ipynb imitation_learning/README.md imitation_learning/tests/test_artifacts.py
git commit -m "docs: configure validation baseline training"
```

### Task 8: Final Requirements and Regression Verification

**Files:**
- Inspect all modified files

**Interfaces:**
- Confirms the complete approved design without adding behavior.

- [ ] **Step 1: Run the complete test suite**

```bash
cd imitation_learning
pytest -q
```

Expected: zero failures.

- [ ] **Step 2: Run syntax and artifact checks**

```bash
python -m compileall model training tests
python -m json.tool kaggle_submission_imitation_agent.ipynb > /dev/null
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 3: Audit removed logic and required fields**

```bash
rg -n "shuffle_mode|warmup_ratio|running_loss|running_accuracy" imitation_learning
rg -n "episode_key|validation_ratio|eval_every_steps|save_every_steps|ema_alpha|norm_mode" imitation_learning
```

Expected: the first search has no production/config matches; historical design
documents may still contain old terms. The second search covers cache, loader,
training, model, YAML, tests, README, and notebook.

- [ ] **Step 4: Inspect repository scope**

```bash
git status --short
git diff --stat
git diff
```

Expected: only intended implementation files differ; pre-existing unrelated
user changes remain untouched.
