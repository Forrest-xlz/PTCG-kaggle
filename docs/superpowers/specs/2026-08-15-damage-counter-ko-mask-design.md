# Damage-Counter KO Action Mask Design

## Goal

Prevent the behavior-cloning policy from learning or choosing wasteful
Dragapult-style damage-counter placements on Bench Pokemon whose current HP is
already zero or lower, while preserving a legal fallback when every available
target is already knocked out.

## Scope

The rule applies only when `select.context == DAMAGE_COUNTER_ANY` (numeric API
value 14). It does not change ordinary damage-counter effects, attack targets,
healing, damage-counter removal, action enumeration order, decoder embeddings,
or the model architecture.

## Shared Rule

For every enumerated action:

1. Resolve each selected option to its target Pokemon.
2. Mark the action undesirable if any selected target has `hp <= 0`.
3. If at least one enumerated action is not undesirable, retain only those
   actions.
4. If every enumerated action is undesirable, retain the complete original
   action list so the required game effect can finish.

Although current `DAMAGE_COUNTER_ANY` observations select one option per step,
the rule is defined at action level so it remains correct if the engine later
allows multi-option selections.

## Training Path

Cache construction applies the shared rule before producing action membership
and the policy target. If the recorded replay choice is removed by the mask,
the sample is excluded from the policy dataset with a dedicated skip reason.
The recorded choice is still encoded into action history and appended by the
caller, so later samples see the true preceding action sequence.

Retained actions preserve their raw option indices. Option features continue to
encode all raw engine options; only action combinations presented to the policy
head are filtered. This avoids changing option positions or decoder features.

The feature signature's action-enumeration identifier changes, invalidating old
packed caches. Extracted JSONL remains compatible, so only `cache_features.py`
must be rerun.

## Inference Path

The Kaggle agent enumerates raw actions, applies the same shared rule, builds
decoder action membership for the retained list, and returns the selected raw
option indices. No additional checkpoint parameter or tensor is required.

## Testing

Tests cover:

- non-damage-counter contexts remain unchanged;
- `HP <= 0` targets are removed when an `HP > 0` alternative exists;
- all targets are restored when every available target has `HP <= 0`;
- multi-option actions are rejected when any selected target is knocked out;
- a masked replay label skips policy training but still produces history;
- notebook inference uses the same action list as project code;
- the feature signature invalidates caches built before this rule.
