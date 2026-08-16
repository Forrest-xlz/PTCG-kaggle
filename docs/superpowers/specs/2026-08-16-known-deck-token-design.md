# Known-Deck Token Design

## Goal

Add one encoder token representing the exact physical cards that the acting
player currently knows are inside their own deck. This captures cards that
were revealed or otherwise identified before being returned to the deck.

The token represents current knowledge, not a count of return events. A
physical card is keyed by its match-unique `serial`, so moving the same card
out and back repeatedly never creates duplicate copies. Two physical copies
with the same Card ID contribute twice when both are currently known in deck.

## State Tracking

Maintain a per-player mapping:

```python
known_deck_cards: dict[int, int]  # serial -> card_id
```

Process observation logs in chronological order before encoding the current
decision:

- An own `MOVE_CARD` whose `toArea` is `DECK` inserts or replaces
  `known_deck_cards[serial] = cardId`.
- An own `MOVE_CARD` whose `fromArea` is `DECK` removes that serial.
- An own `DRAW` removes that serial.
- `SHUFFLE` does not change the mapping: the card's position becomes unknown,
  but its membership in the deck remains known.
- Opponent events never affect the acting player's mapping.
- Missing card IDs or serials are ignored rather than fabricating identity.
- Repeated logs are idempotent because insertion and removal use serial keys.
- Start each replay/match with an empty mapping.

The cache builder derives this state from the observations already present in
the extracted JSONL. It updates state even when a record is subsequently
skipped, because later decisions still depend on the event. Extraction output
does not change and does not need rebuilding.

The Kaggle agent uses the identical transition function. It clears state when
called for initial deck selection (`obs.select is None`) and applies logs before
building every policy input.

## Encoder Feature

Increase fixed encoder tokens from 26 to 27. Place the new own-known-deck token
next to the existing full-deck token.

For every `(serial, card_id)` currently in the mapping, contribute once:

```text
known_deck_card_embedding(card_id)
+ projected_54_dim_static_card_features(card_id)
```

Sum all contributions with weight 1.0. The learned Card ID embedding belongs
to a new `own_known_deck` region and is independent from the existing full-deck,
hand, discard, and field embeddings. Static-card projection follows the
existing `card_mlp_scope`: `shared` reuses the global projection, while
`region` gives `own_known_deck` its own projection. An empty mapping produces
the same zero aggregate used by other empty fixed-area tokens.

## Token MLP

Add this model configuration field:

```yaml
known_deck_token_mlp_layers: 1
```

- `0`: use the summed token directly.
- `1`: apply one `d_model -> d_model` linear layer.
- `N > 1`: append `ReLU` and `d_model -> d_model` for each later layer.

The existing `region_token_mlp_residual` controls composition consistently
with other region tokens: `false` replaces the aggregate with the MLP result;
`true` adds the MLP result to the aggregate.

## Cache and Checkpoint Compatibility

The packed cache adds the 27th sparse encoder offset, so increment the cache
schema and require rebuilding cache. Do not change the extraction schema.

The model config stored in checkpoints includes
`known_deck_token_mlp_layers`. Old checkpoints do not contain the new token or
its parameters and are intentionally incompatible with this architecture.
The Kaggle notebook constructs the architecture from checkpoint config and
must support the new field.

## Files and Boundaries

- A focused tracker module owns log transitions and can be reused by cache
  construction and tests.
- `model/features.py` only converts the current mapping values into the new
  sparse token.
- `model/network.py` owns the new embedding region, token count, and token MLP.
- `training/cache_features.py` owns per-replay/per-player tracker lifecycle.
- `training/feature_cache.py` stores the new 27-token sparse offsets.
- `kaggle_submission_imitation_agent.ipynb` embeds the same tracker and model
  behavior for online inference.

## Validation

- State tests cover add, idempotent repeated add, remove by draw, remove by
  move from deck, shuffle retention, opponent-event isolation, two physical
  copies of one Card ID, and replay reset.
- Feature tests prove one contribution per current serial and a zero aggregate
  when empty.
- Cache tests prove chronology is preserved across samples, including skipped
  samples, and the schema rejects old cache.
- Network tests cover 27-token shape, independent known-deck embedding region,
  static projection, MLP layer counts, and residual behavior.
- Notebook tests verify the generated agent resets and updates the tracker and
  loads checkpoint-driven architecture.
