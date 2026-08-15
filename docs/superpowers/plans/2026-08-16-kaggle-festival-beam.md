# Kaggle Festival Beam Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional first-N-decision turn-level Beam Search to the standalone Kaggle Festival submission notebook.

**Architecture:** Embed the CG search-state traversal directly in the notebook-generated `main.py`, reuse its existing feature/model/ensemble inference, reconstruct only Festival's hidden zones exactly, and fill opponent hidden zones with harmless legal placeholders. A per-game eligible-decision counter gates Beam and every search failure returns the precomputed Greedy action.

**Tech Stack:** Python 3.10+, PyTorch, CG `search_*` API, Jupyter notebook JSON, pytest.

## Global Constraints

- The generated submission remains standalone and adds no packaged project modules.
- Existing Greedy, ensemble probability averaging, model loading, damage mask, deck, and archive behavior remain unchanged when Beam is disabled.
- Search scores only Festival actions and ends when its turn changes.
- Opponent immediate responses are deterministic legal actions and never invoke the policy model.
- Beam setup decisions are excluded from the first-N budget.

---

### Task 1: Notebook Contract and Pure Beam Helpers

**Files:**
- Create: `imitation_learning/tests/test_kaggle_submission_beam_notebook.py`
- Modify: `imitation_learning/kaggle_submission_imitation_agent.ipynb`

**Interfaces:**
- Produces notebook parameters `BEAM_SEARCH_ENABLED`, `BEAM_SEARCH_FIRST_N_DECISIONS`, `BEAM_WIDTH`, `BEAM_EXPANSION_TOP_K`, `BEAM_ALPHA`, `BEAM_MAX_DEPTH`, `BEAM_SEED`, and `BEAM_LOG_ENABLED`.
- Produces generated-main helpers `trajectory_score(log_sum, steps, alpha)` and `beam_is_eligible(enabled, decision_index, first_n, turn)`.

- [ ] Write a failing notebook-contract test that loads notebook JSON, extracts the `%%writefile main.py` cell, checks all parameters are transferred through `beam_config.json`, compiles the generated source, and AST-executes the two pure helpers to assert score normalization and first-N/setup gating.
- [ ] Run `pytest imitation_learning/tests/test_kaggle_submission_beam_notebook.py -q` and confirm failure because Beam parameters/helpers are absent.
- [ ] Add validated Beam parameters to the notebook parameter cell, write `beam_config.json`, include it in the archive, load it in generated `main.py`, and add the two pure helpers.
- [ ] Run the notebook-contract test and confirm it passes.
- [ ] Commit the notebook and test as `feat: configure Kaggle submission beam search`.

### Task 2: Embedded Search, Fallback, and Local Festival Smoke

**Files:**
- Modify: `imitation_learning/kaggle_submission_imitation_agent.ipynb`
- Modify: `imitation_learning/tests/test_kaggle_submission_beam_notebook.py`

**Interfaces:**
- Produces `evaluate_policy(obs, deck, history)` returning legal actions, ensemble-averaged probabilities, log probabilities, and encoded options.
- Produces `build_festival_search_inputs(obs, seed)` with exact own hidden zones and placeholder opponent hidden zones.
- Produces `beam_select(obs, history, fallback_action, decision_number)` returning the selected root action plus diagnostics.

- [ ] Extend the failing test to require CG search lifecycle calls, trajectory-local history cloning, top-K expansion, width pruning, turn/depth termination, deterministic opponent responses, cleanup, and Greedy fallback.
- [ ] Run the focused test and confirm it fails because embedded search is absent.
- [ ] Refactor current Greedy inference into `evaluate_policy`, embed the trajectory/search implementation, update `agent()` to reset and increment its eligible counter, append the real selected action once, and print diagnostics only when configured.
- [ ] Run the focused test, validate notebook JSON, and compile the exact generated `main.py` cell.
- [ ] Run a local real-checkpoint Festival smoke with placeholder opponent hidden zones. Require `expanded_nodes > 0`, `branch_errors == 0`, and `greedy_fallback == false`; separately verify disabled and exhausted-budget paths do not call `search_begin`.
- [ ] Run affected submission/Beam tests and `git diff --check` on the files owned by this change.
- [ ] Commit as `feat: add Beam search to Kaggle Festival agent`.

