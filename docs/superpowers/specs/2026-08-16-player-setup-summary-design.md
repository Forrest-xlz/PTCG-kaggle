# Player Setup Summary Feature Design

## Goal

Add compact player-summary features that expose whole-board energy state and
early-game formation progress. The features must help the policy recognize
bad or incomplete openings without leaking the opponent's hidden hand.

## Scope

- Extend the existing own and opponent dense player summaries.
- Keep the encoder, decoder, policy loss, history encoding, and sampling logic
  otherwise unchanged.
- Use only information observable by the acting player at inference time.
- Keep training, validation, and Kaggle inference feature construction exactly
  aligned.
- Do not change the extracted replay JSONL schema.

## In-Play Population

Every board aggregate uses the player's Active Pokemon, when present, plus all
Pokemon on that player's Bench. Attached pre-evolutions are not counted as
separate in-play Pokemon.

## Shared Public Features

Both player summaries receive the following eleven scalar features:

1. `total_energy_cards`: sum of `len(pokemon.energyCards)` over all in-play
   Pokemon, divided by 32.
2. `total_effective_energies`: sum of `len(pokemon.energies)` over all in-play
   Pokemon, divided by 64. This deliberately differs from physical Energy
   cards and reflects effects such as Meganium's Wild Growth.
3. `basic_in_play`: Basic Pokemon count divided by 9.
4. `stage1_in_play`: Stage 1 Pokemon count divided by 9.
5. `stage2_in_play`: Stage 2 Pokemon count divided by 9.
6. `incomplete_evolution_lines`: in-play Pokemon that have at least one direct
   evolution present in the configured starting deck, divided by 9.
7. `completed_evolution_lines`: terminal members of an evolution line present
   in the configured starting deck, divided by 9. A standalone Basic Pokemon
   with neither a parent nor a child in that deck is excluded.
8. `attack_ready_pokemon`: in-play Pokemon whose effective Energy satisfies at
   least one printed attack cost, divided by 9. Pokemon without attacks are
   excluded.
9. `total_min_attack_deficit`: for every in-play Pokemon with attacks, compute
   its smallest Energy deficit over its attacks, sum those minima, and divide
   by 45. Pokemon without attacks contribute zero.
10. `nearest_attack_deficit`: minimum Energy deficit among all attacks of all
    in-play Pokemon, divided by 5 and clipped to 1. If no in-play Pokemon has
    an attack, use 1.
11. `empty_bench_slots`: `max(player.benchMax - len(player.bench), 0)` divided
    by 8.

These features use only public board state, so the same definitions are safe
for the own and opponent summaries.

## Own-Hand Features

Only the acting player's summary receives these four additional features:

1. `immediately_evolvable`: distinct in-play Pokemon that have a matching
   direct evolution in hand and satisfy normal timing rules, divided by 9.
2. `matching_evolution_cards_in_hand`: number of physical cards in hand whose
   `CardData.evolvesFrom` matches an in-play Pokemon name, divided by 20.
3. `basic_pokemon_in_hand`: Basic Pokemon cards in hand, divided by 20.
4. `basic_energy_in_hand`: Basic Energy cards in hand, divided by 20.

Opponent hand contents must never be read to construct corresponding values.
The opponent projection remains a separate MLP and therefore does not require
the same input width as the own projection.

## Evolution Graph and Timing

Build a direct evolution graph from `CardData.name` and
`CardData.evolvesFrom`, restricted to Card IDs in the configured starting
deck. Duplicate physical copies do not create duplicate graph edges.

An in-play Pokemon is structurally matched by a hand card when the hand card's
`evolvesFrom` equals the current Pokemon's card name. `immediately_evolvable`
also requires:

- `state.turn >= 2` (turn indices 0 and 1 are the two players' first turns);
  and
- the Pokemon was not played during the current turn, unless the active
  Stadium's static rules permit that Pokemon's Energy type to evolve during
  the turn it was played.

Same-turn evolution Stadiums are discovered once by scanning static Stadium
skill text for the engine's canonical "can evolve ... during the turn they
play" rule and recording its affected Energy type. This avoids a hard-coded
Festival deck ID. If no such rule is found, `appearThisTurn=True` remains
ineligible. This is an eligibility estimate based on public game rules; the
model still receives the engine's legal options as the authoritative action
set.

## Attack-Energy Matching

Extend the numeric catalog with each card's attack IDs and each attack's
printed Energy requirements. Compare requirements against
`Pokemon.energies`, not `energyCards`.

For each attack:

1. Match specifically typed requirements first.
2. Rainbow Energy may satisfy any specifically typed requirement.
3. Team Rocket Energy may satisfy Psychic or Darkness requirements.
4. Match Colorless requirements from any remaining effective Energy units.
5. The deficit is the number of unmatched requirements.

The minimum deficit for a Pokemon is the smallest deficit among its attacks.
This calculation measures formation readiness; it does not claim a Benched
Pokemon can attack before becoming Active and does not override special
conditions or engine legality.

## Dimensions and Data Flow

- Existing own summary: 69 dimensions.
- Add eleven public features and four own-hand features.
- New own summary: 84 dimensions.
- Existing opponent summary: 71 dimensions.
- Add eleven public features.
- New opponent summary: 82 dimensions.

The configured starting deck and an extended immutable numeric catalog are
passed into summary construction. Cache construction stores the resulting
dense arrays, so training performs no new per-batch CPU feature work.

## Compatibility

- Increment the packed feature-cache schema and feature signature.
- Re-run feature caching; replay extraction is not required.
- Existing checkpoints are incompatible because both summary projection input
  widths change.
- Update the Kaggle submission notebook with the identical catalog and feature
  calculations.
- Ensemble checkpoints must share the new summary dimensions and feature
  semantics.

## Validation

Tests cover:

- physical versus effective whole-board Energy totals;
- own/opponent visibility boundaries;
- stage counts and deck-restricted evolution graph behavior;
- matching evolution cards and timing eligibility;
- terminal versus incomplete evolution lines;
- exact, Colorless, Rainbow, and Team Rocket Energy matching;
- attack-ready counts and both deficit aggregates;
- empty Bench slots using `benchMax`;
- expected own/opponent dimensions;
- cache signature rejection for old caches; and
- Kaggle generated-agent source parity.
