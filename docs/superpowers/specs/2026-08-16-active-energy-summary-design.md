# Active Energy Summary Design

## Goal

Expose the physical and effective Energy totals of each player's Active
Pokemon in that player's summary token.

## Feature definition

Each player summary appends two public scalar features:

1. `active_energy_cards`: the sum of `len(pokemon.energyCards)` over the
   player's Active Pokemon, normalized by `32`.
2. `active_effective_energy`: the sum of `len(pokemon.energies)` over the
   player's Active Pokemon, normalized by `64`.

Both values are zero when the player has no Active Pokemon. The own summary
uses the own Active Pokemon and the opponent summary uses the opponent Active
Pokemon. No hidden information is introduced.

## Integration

- Append the features to the existing public setup summary so training,
  validation, and inference share one implementation.
- Increase summary dimensions from `84/82` to `86/84`.
- Increase the packed-cache schema from `18` to `19` and update the feature
  signature from setup v4 to setup v5.
- Mirror the feature extraction and dimensions in the Kaggle submission
  notebook.

## Verification

- Unit-test physical and effective Active Energy independently.
- Verify empty Active areas produce zeros.
- Verify cache/network/notebook dimensions and signatures agree.

