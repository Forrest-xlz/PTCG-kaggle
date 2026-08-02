# Remove Effect Option Feature

## Goal

Remove `select.effect` from the behavior-cloning model input while retaining
`select.contextCard` as selection-level card context.

## Feature layout

Option categorical rows contain six fields in this exact order:

1. option type
2. select context
3. candidate card ID
4. target card ID
5. attack ID
6. context-card ID

The model keeps an independent learned context-card embedding and adds the
globally shared 54-dimensional static-card projection for that card. It does
not extract, cache, embed, or project the effect card.

## Compatibility

The packed feature-cache schema advances from 11 to 12 and the decoder layout
signature advances to `option-components-context-card-v7`. Existing extracted
winner replay JSONL files remain valid, but feature caches and checkpoints must
be rebuilt.

## Synchronized artifacts

The training feature builder, packed cache, policy network, submission
notebook, focused tests, and README must all use the same six-field layout.

