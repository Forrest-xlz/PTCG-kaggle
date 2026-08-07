# Encoder Pokemon Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional 38-dimensional runtime Pokemon embeddings, independent immediate-pre-evolution embeddings, and expanded player summaries to the encoder and Kaggle agent.

**Architecture:** Feature extraction always produces a cache superset for all 18 field Pokemon. Two checkpoint-owned booleans independently enable one shared runtime projection and one independent pre-evolution Card ID table; player summaries always use the expanded 79-dimensional common layout. Existing JSONL remains sufficient, while the packed cache schema changes.

**Tech Stack:** Python 3.10, NumPy, packed mmap cache, PyTorch, PyYAML, Jupyter notebook JSON.

## Global Constraints

- Encoder runtime Pokemon features contain exactly 38 values in the approved order.
- Decoder Pokemon dynamics remain exactly 23-dimensional.
- Energy readiness supports exact types, Rainbow, Team Rocket for Psychic/Darkness, and Colorless fallback.
- Only the first two card attacks are represented; missing attacks are zero padded.
- The two switches default to false and are checkpoint self-describing.
- Pre-evolution uses only `preEvolution[-1]` and a trainable table independent of every existing Card embedding.
- Old extracted JSONL is reused; old packed caches are rebuilt.
- No unrelated decoder, history, validation, or loss behavior changes.

---

### Task 1: Runtime Pokemon and player-summary features

**Files:**
- Modify: `imitation_learning/model/features.py`
- Create: `imitation_learning/tests/test_encoder_pokemon_dynamics.py`

**Interfaces:**
- Produces `ENCODER_POKEMON_DYNAMIC_DIM = 38`.
- Produces `encoder_pokemon_dynamic_features(pokemon, player, is_active, is_own, catalog) -> np.ndarray`.
- Extends `EncoderFeatures` with `pokemon_dynamic` and `pre_evolution_ids` for 18 slots.
- Changes summary constants to common 79, own 94, opponent 96.

- [ ] Write tests asserting the 38 columns, absent zeros, special conditions only on Active, two attack slots, and the immediate pre-evolution ID.
- [ ] Write tests for exact Energy, Rainbow, Team Rocket, Colorless, and insufficient-Energy matching.
- [ ] Write tests asserting summary dimensions 79/94/96, bench capacity ratio, both Energy counts, and empty-Bench minimum HP.
- [ ] Confirm the focused tests fail because the new interfaces/constants do not exist. If Python is unavailable, record that RED execution is blocked before production edits.
- [ ] Implement a shared 23-value base helper, a deterministic attack-cost matcher, the 15 encoder-only values, and the expanded append-only summary fields.
- [ ] Populate runtime arrays and pre-evolution sentinel IDs in the existing 18-slot encoder order.
- [ ] Run the focused tests when a Python interpreter is available.

### Task 2: Packed cache schema and training plumbing

**Files:**
- Modify: `imitation_learning/training/feature_cache.py`
- Modify: `imitation_learning/training/cache_features.py`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/tests/test_feature_cache.py`

**Interfaces:**
- Adds cache arrays `encoder_pokemon_dynamic [samples,18,38]` and `encoder_pre_evolution [samples,18]`.
- Extends `FeatureRecord`, `FeatureView`, and `CachedBatch` with both arrays.

- [ ] Extend cache tests first to require exact round-trip and collated shapes/dtypes.
- [ ] Confirm the cache test fails from missing dataclass fields when Python is available.
- [ ] Increment the schema, add compact float16/uint16 sections, validate widths and Card ID bounds, and implement sample/collate paths.
- [ ] Store the new `EncoderFeatures` fields during cache conversion and update the cache signature identically in cache and train modules.
- [ ] Transfer runtime arrays as float32 and pre-evolution IDs as long in `_forward_batch`.
- [ ] Run focused cache tests when Python is available.

### Task 3: Optional encoder modules and YAML configuration

**Files:**
- Modify: `imitation_learning/model/network.py`
- Modify: `imitation_learning/training/train.py`
- Modify: `imitation_learning/cfg/train.yaml`
- Create: `imitation_learning/tests/test_encoder_pokemon_network.py`

**Interfaces:**
- Adds `ModelConfig.pokemon_dynamic_embedding: bool = False`.
- Adds `ModelConfig.pre_evolution_embedding: bool = False`.
- Extends `PTCGTransformer.forward` with the two encoder tensors.

- [ ] Write model tests first for all four switch combinations, exact-zero absent contribution, independent pre-evolution parameters, and old defaults.
- [ ] Confirm tests fail because the config fields/modules are absent when Python is available.
- [ ] Construct `Linear(38,d_model)` only when runtime dynamics are enabled.
- [ ] Construct `Embedding(card_count+1,d_model,padding_idx=card_count)` only when pre-evolution is enabled.
- [ ] Add enabled contributions to the first 18 encoder tokens before regional token MLPs; validate input shapes.
- [ ] Add YAML/dataclass/checkpoint plumbing and run focused model tests when Python is available.

### Task 4: Kaggle parity, documentation, and verification

**Files:**
- Modify: `imitation_learning/kaggle_submission_imitation_agent.ipynb`
- Modify: `imitation_learning/README.md`

**Interfaces:**
- Generated `main.py` restores both switches from checkpoint configuration and constructs identical online features.

- [ ] Mirror constants, Energy matching, summaries, runtime feature construction, ModelConfig, modules, and forward arguments in the notebook.
- [ ] Preserve old-checkpoint defaults in the parameter preview and model loader.
- [ ] Document both switches, summary dimensions, schema rebuild, and the fact that extraction is unnecessary.
- [ ] Validate notebook JSON, compare project/notebook module names and constants, run `git diff --check`, and run Python compile/tests if an interpreter is available.
- [ ] Commit the implementation on the current feature branch without pushing or merging.
