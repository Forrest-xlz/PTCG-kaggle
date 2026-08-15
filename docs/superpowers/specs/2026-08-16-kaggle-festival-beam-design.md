# Kaggle Festival Beam Search Design

## Goal

Add optional turn-level policy Beam Search to
`imitation_learning/kaggle_submission_imitation_agent.ipynb`. The agent uses
Beam only for its first configured number of eligible decisions, then retains
the existing Greedy Top-1 behavior. The generated submission remains a single
`main.py` plus the existing model, deck, manifest, and `cg` assets.

## Configuration

The notebook parameter cell exposes:

```python
BEAM_SEARCH_ENABLED = True
BEAM_SEARCH_FIRST_N_DECISIONS = 10
BEAM_WIDTH = 6
BEAM_EXPANSION_TOP_K = 2
BEAM_ALPHA = 1.0
BEAM_MAX_DEPTH = 64
BEAM_SEED = 42
BEAM_LOG_ENABLED = True
```

An eligible decision is an agent call after setup with `current.turn > 0`.
Setup decisions do not consume the budget. The counter resets on the deck
request that starts a new game. A non-positive `FIRST_N` value is invalid when
Beam is enabled.

## Search Semantics

At every root-player search state, enumerate the same maximum 64 legal action
combinations used by policy inference. Expand the top
`BEAM_EXPANSION_TOP_K` actions according to the current ensemble-averaged
policy probabilities. Retain `BEAM_WIDTH` live trajectories. Rank a trajectory
with:

```text
sum(log(policy_probability)) / number_of_scored_actions ** BEAM_ALPHA
```

Stop a trajectory when the game terminates, the root player's turn changes,
or `BEAM_MAX_DEPTH` transitions have been simulated. Return the first action
of the best completed trajectory. Only root-player actions contribute to the
score.

If the CG engine asks the opponent for an immediate response before the turn
number changes, choose the first enumerated legal response deterministically.
The model is never evaluated as the opponent.

## Hidden-State Completion

The own hidden deck and prize identities are reconstructed from the configured
Festival deck after subtracting all visible own cards. The identities are
shuffled reproducibly from `BEAM_SEED` and the decision counter.

Festival Lead has only two cards that manipulate opponent hidden zones:
Unfair Stamp and Judge. Their opponent card identities do not affect later
root-player legal actions within the current turn. The opponent hidden deck,
hand, and prizes are therefore filled with Basic Grass Energy Card ID `1`,
preserving only the required zone sizes. A hidden opponent Active, if present,
uses Basic Pokemon Card ID `92`; setup itself is never searched.

This construction was locally verified against the CG engine with 47 opponent
deck cards, 6 hand cards, and 6 prize cards replaced by placeholders.
`search_begin`, `search_step`, and a real policy Beam decision all completed.

## State and History

Each Beam trajectory owns a clone of the existing three-action model history.
When a root-player action advances a branch, its encoded selected action is
appended to that branch only. The action chosen at the real root is appended
once to the global history after search completes, matching Greedy behavior.

## Failure and Diagnostics

Any determinization error, CG search error, empty completed Beam, or cleanup
error falls back to the already-computed Greedy action. Every CG search ID is
released and `search_end()` is attempted in `finally` cleanup.

When logging is enabled, each eligible decision prints the decision number,
expanded node count, maximum depth, score, branch error count, and whether it
fell back. When Beam is disabled or its decision budget is exhausted, no CG
search calls are made.

## Validation

- Validate notebook JSON and compile the generated `main.py` source.
- Unit-test the decision-budget gate and trajectory scoring.
- Run a local Festival decision with placeholder opponent hidden cards and a
  real policy checkpoint; require nonzero expanded nodes, zero branch errors,
  and no Greedy fallback.
- Verify disabled Beam and exhausted budget select the existing Greedy path.

