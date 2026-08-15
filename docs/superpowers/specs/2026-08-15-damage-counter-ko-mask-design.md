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

Cache construction stores one boolean eligibility value for every enumerated
action without deleting actions or samples. The packed cache therefore supports
both masked and original BC training without another cache rebuild.

`train.damage_counter_ko_mask` controls the behavior at training and validation
runtime. When enabled, ineligible action logits are masked. If the replay target
itself is ineligible, that row is excluded from policy loss and accuracy while
the rest of the batch remains active. The recorded choice remains in action
history, so later samples see the true preceding action sequence. When disabled,
the stored eligibility values are ignored and behavior exactly matches the
original BC pipeline.

All actions preserve their raw option indices. Option features continue to
encode all raw engine options. This avoids changing option positions, policy
targets, or decoder features.

The cache schema and feature signature change to persist action eligibility,
invalidating old packed caches. Extracted JSONL remains compatible, so only
`cache_features.py` must be rerun. Once rebuilt, changing the training switch
does not require another cache rebuild.

## Inference Path

The Kaggle notebook exposes `DAMAGE_COUNTER_KO_MASK`. When enabled, the agent
enumerates raw actions, computes the same eligibility values, masks policy
scores, and returns the selected raw option indices. When disabled, no policy
scores are masked. No additional checkpoint architecture parameter is required.

## Testing

Tests cover:

- non-damage-counter contexts remain unchanged;
- `HP <= 0` targets are removed when an `HP > 0` alternative exists;
- all targets are restored when every available target has `HP <= 0`;
- multi-option actions are rejected when any selected target is knocked out;
- a masked replay label is excluded from policy metrics without changing its
  cached history;
- disabling the training switch reproduces original policy metrics;
- notebook inference uses the same action list as project code;
- the feature signature invalidates caches built before this rule.
