# Revealed-Hand and Learnable CLS Tokens Design

## Goal

Add two encoder tokens that summarize publicly revealed cards still held in
each player's hand, plus an optional learnable CLS token. Preserve the current
policy decoder, training split logic, and all unrelated model behavior.

## Scope

This change covers feature caching, model configuration, encoder assembly,
checkpoint architecture metadata, inference export, and the Kaggle submission
notebook. It does not change replay extraction or the policy target.

## Public Revealed-Hand State

The cache builder processes each replay episode sequentially and maintains one
publicly known hand map per player:

```text
known_hand[player_index][card_serial] = card_id
```

Card serial is used only for exact state maintenance. The model input aggregates
the surviving entries by Card ID, so duplicate copies of the same card contribute
multiple times to the pooled token.

The tracker applies every observation's logs before encoding that observation:

- A public `MOVE_CARD` whose destination is `HAND` adds its `serial` and
  `cardId` to that player's known hand.
- `PLAY`, `ATTACH`, and `EVOLVE` remove their card serial from that player's
  known hand.
- A public `MOVE_CARD` whose source is `HAND` removes its serial, regardless of
  destination.
- A face-down `MOVE_CARD_REVERSE` entering a hand does not add knowledge.
- A face-down or otherwise ambiguous movement out of a hand clears the known
  entries for that player conservatively, because their identity can no longer
  be aligned safely.
- Ordinary private draws do not become revealed-hand cards. In particular,
  seeing one's own private hand does not imply that those cards are publicly
  known to the opponent.
- The tracker resets at each episode boundary and at inference match reset.

For a sample, `obs.current.yourIndex` determines which maintained map becomes
`own_revealed_hand` and which becomes `opponent_revealed_hand`.

## Encoder Representation

Two fixed pooled encoder positions are appended after the existing 26 base
tokens:

1. `own_revealed_hand`
2. `opponent_revealed_hand`

Each side has an independent learned Card ID embedding and an independent token
MLP. Both sides reuse the current static 54-dimensional card projection policy:
the globally shared projection remains shared when `card_mlp_scope: shared`, and
the appropriate independent region projection is used when
`card_mlp_scope: region`.

The token is the weighted sum of learned Card ID embeddings and projected static
card features. Duplicate cards are represented through count weights. An empty
revealed-hand token is padding-masked before the main encoder.

`revealed_hand_token_mlp_layers` controls both token MLP depths:

- `0`: use the pooled embedding sum directly.
- `N > 0`: apply N independent `d_model -> d_model` layers to each side, using
  the project's existing region-token MLP convention and residual setting.

The two MLPs have the same configured depth but do not share parameters.

## Optional Learnable CLS Token

`learnable_cls_token` is a Boolean model parameter:

- `false`: no CLS parameter or encoder position is created.
- `true`: create one learnable vector of shape `[d_model]`, expand it across the
  batch, and append it to the encoder sequence.

The CLS token is never padding-masked. It participates in the same input
LayerNorm/dropout and main transformer encoder as every other token. No auxiliary
head consumes it in this change; it serves as an additional learned global
workspace visible to decoder cross-attention.

With all optional inputs enabled, the encoder sequence length is:

```text
26 base + 2 revealed-hand + 1 action-history + 1 CLS = 30 tokens
```

## Configuration and Checkpoints

The model configuration adds:

```yaml
revealed_hand_token_mlp_layers: 1
learnable_cls_token: true
```

Both values are stored in checkpoint architecture metadata and restored by all
automatic model constructors. The two additional independent Card ID ranges
raise `encoder_size` from 22,000 to 25,000. Consequently, checkpoints trained
before this feature remain usable with their original branch/notebook but are
not shape-compatible with the revised architecture.

The submission notebook exposes the same fields and maintains the revealed-hand
tracker between calls to `agent`. Ensemble checkpoints must still agree on all
architecture compatibility fields.

## Cache Compatibility

The packed cache gains two sparse encoder words and their presence information.
The cache schema version and encoder-layout signature are bumped. Existing
extracted JSONL files already contain complete observations and logs, so replay
extraction is unchanged. Users must rerun only `training/cache_features.py`.

## Error Handling

- Malformed public log entries without a usable player, serial, or Card ID are
  ignored when adding knowledge rather than fabricating a card.
- Removal of an unknown serial is a no-op.
- Unknown log types do not affect tracker state.
- Configuration rejects negative MLP depths and non-Boolean CLS values.
- Model forward validates the revised encoder token count and mask width.

## Verification

Tests will cover:

- add, remove, duplicate-card, private-draw, and conservative-clear tracker
  behavior;
- perspective-relative own/opponent token construction;
- empty revealed-hand padding masks;
- sequence lengths with history and CLS independently enabled or disabled;
- configuration validation and explicit rejection of incompatible old caches;
- training and Kaggle inference feature parity on a synthetic log sequence;
- cache schema rejection of old feature caches.
