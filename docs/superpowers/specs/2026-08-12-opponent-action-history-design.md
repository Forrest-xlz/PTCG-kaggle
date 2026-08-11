# Opponent Public-Action History Design

## Goal

Add an opponent action-history token alongside the existing own-action history
token. The new token uses only information visible in `Observation.logs`, so
training, standalone validation, and Kaggle inference have the same information
boundary.

The opponent encoder mirrors the existing three-action temporal structure but
does not share embeddings, projections, or MLP parameters with the own-history
encoder.

## Scope

- Preserve the existing own-action history implementation.
- Model the latest three observable opponent macro actions.
- Support independent `off`, `basic`, `structural`, and `full` modes.
- Add one optional opponent-history token to the main encoder.
- Update cache generation, model configuration, standalone validation, and the
  Kaggle submission notebook.
- Do not use replay-only opponent option selections or hidden card identities.

## Information Boundary

Opponent history is derived exclusively from the current agent observation:

```text
Observation.logs
  -> logs whose playerIndex is the opponent
  -> observable macro actions
  -> latest-three opponent history
```

Replay data may contain the other player's exact selected option, but that
information must not enter the opponent features. It is unavailable to the
Kaggle agent and would create train/inference leakage.

An absent, hidden, or invalid Card ID maps to the card vocabulary's explicit
unknown entry. An absent or invalid Attack ID and Area use their corresponding
unknown entries. Missing dynamic state is represented by zero dynamic features,
while the learned unknown entity embedding remains present.

## Macro-Action Aggregation

Primary public events start a macro action:

- `PLAY`
- `ATTACH`
- `EVOLVE`
- `SWITCH`
- `ATTACK`
- `TURN_END`

Result events such as HP changes, card movement, and special-condition changes
are attached to the preceding primary action until the next primary event or
turn boundary. A sequence such as:

```text
ATTACK -> HP_CHANGE -> HP_CHANGE -> MOVE_CARD
```

occupies one of the three history positions. Its public log-event embeddings
are summed within that action, analogous to pooling selected options in an own
combination action. Observable effect logs without a primary event form an
`OTHER_PUBLIC_EFFECT` action.

Logs without `playerIndex` are not classified as opponent actions. Logs are not
deduplicated heuristically: the API contract makes them incremental, while two
identical adjacent events may be legitimate actions.

## Opponent Action Representation

Each public log event may contribute independent opponent-side embeddings for:

- log or macro-action type;
- source Card ID;
- target Card ID;
- Attack ID;
- source Area;
- target Area;
- player relation;
- observable numeric or state result.

In `full` mode, visible source and target cards also receive separate static
card-feature projections, attacks receive an attack-feature projection, and a
publicly locatable Pokemon may receive dynamic features from the current board.
Failure to locate a Pokemon by its public serial is not an error; its dynamic
features are zero.

The temporal pipeline is:

```text
event embeddings within one macro action: sum
  -> opponent-history action MLP
  -> concatenate [t-3, t-2, t-1]
  -> opponent-history sequence MLP
  -> one opponent-history encoder token
```

Every opponent-side embedding, static projection, dynamic projection, action
MLP, and sequence MLP is independent from the corresponding own-history
parameter.

## Configuration

Add independent model settings:

```yaml
model:
  opponent_history_encoding: full
  opponent_history_action_mlp_layers: 1
  opponent_history_sequence_mlp_layers: 2
```

Modes:

- `off`: no opponent-history token;
- `basic`: macro-action type only;
- `structural`: action type, public areas, relation, and public quantities;
- `full`: structural data plus public cards, attacks, static features, and
  publicly recoverable dynamic features.

The settings are saved in checkpoint architecture metadata and automatically
consumed by standalone validation and the Kaggle notebook. For an older
checkpoint with no opponent-history settings, loading defaults the mode to
`off`.

## Cache and Training Data Flow

No replay re-extraction is required because extracted observations already
retain `logs`. The packed feature cache must be rebuilt and its schema version
must increase.

For each episode and each player perspective, cache construction owns a separate
opponent-history deque. On every training record it:

1. resets histories when the episode changes;
2. parses only opponent logs visible in the record's observation;
3. appends completed macro actions to that perspective's deque;
4. packs the latest three opponent actions;
5. builds the current supervised sample.

The update occurs before packing the current sample because the logs are already
visible at that decision. Perspective-specific state prevents histories from
leaking across players or episodes.

## Kaggle Inference

The generated agent owns two independent deques:

```python
ACTION_HISTORY = deque(maxlen=3)
OPPONENT_ACTION_HISTORY = deque(maxlen=3)
```

At every agent call it parses `obs.logs` and updates the opponent deque before
encoding the current decision. Both deques are cleared at deck selection / new
episode initialization. Notebook parsing and feature construction must remain
behaviorally identical to cache construction.

## Model Integration

When enabled, the opponent history becomes one additional main-encoder token,
placed alongside the existing own-history token. Its padding mask is true only
when all three opponent history slots are empty. The own-history token and all
existing encoder and decoder behavior remain unchanged.

## Error Handling

- Empty logs leave opponent history unchanged.
- Missing `playerIndex` logs are ignored for opponent history.
- Invalid entity identifiers map to unknown entries rather than raising.
- Missing dynamic lookup results produce zero dynamic vectors.
- Unsupported but observable effects map to `OTHER_PUBLIC_EFFECT`.
- Malformed container shapes still raise contextual cache errors with source and
  line information.

## Compatibility and Rebuild Requirements

- Replay extraction: unchanged; no rebuild required.
- Packed feature cache: schema bump and full rebuild required.
- New checkpoints: include opponent-history architecture settings.
- Old checkpoints: opponent history defaults to `off` and remains loadable.
- New enabled models: must be trained after rebuilding the cache.

## Verification

Tests must cover:

1. aggregation boundaries for `PLAY`, `ATTACH`, `EVOLVE`, `SWITCH`, `ATTACK`,
   and `TURN_END`;
2. multiple result logs occupying one macro-action slot;
3. unknown Card, Attack, and Area mapping without hidden-information recovery;
4. filtering of the current player's logs;
5. cache update-before-pack ordering;
6. isolation across episode and player perspective;
7. complete parameter independence between own and opponent encoders;
8. `off`, `basic`, `structural`, and `full` behavior and padding masks;
9. cache schema/signature enforcement;
10. Kaggle notebook parity, JSON validity, and generated-agent compilation.
