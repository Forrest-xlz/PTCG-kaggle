# The Pokémon Company – PTCG AI Battle Challenge Simulation: Transformer Behavior Cloning

Final rank: 59 / 6,807 teams. This is my solution to the [Kaggle simulation competition](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/). The final submission is a two-model Transformer policy ensemble trained on game replays. I used supervised next-action prediction rather than an online reinforcement-learning or search pipeline.

The experiments below show how I chose the data, representation, and training setup. The plots track action-prediction cross-entropy (CE) on training and validation replays.

---

## Overview

In this simulation competition, an agent builds a 60-card deck and repeatedly plays Pokémon TCG games against agents with similar skill ratings. At each decision the simulator provides the visible board, game log, and legal options; the agent returns the option indices it chooses. A turn can involve playing cards, choosing targets, managing the Active Pokémon and Bench, attaching Energy, retreating, and attacking. A locally sensible move can be poor because it spends a card needed for a later combination, gives up a favorable matchup, or leaves the next turn without a way to attack. [Kaggle's competition description](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/evaluation) describes the simulator interface and rating-based opponent pool; the [Pokémon TCG rulebook](https://www.pokemon.com/static-assets/content-assets/cms2/pdf/trading-card-game/rulebook/par_rulebook_en.pdf) explains the board zones and turn actions.

The difficulty has three connected parts. Complexity: cards have different effects, a deck combines many of them, and the same card can be useful or wasteful depending on the current board, available Energy, and what has already been played. The agent may be asked to choose one option or a combination of options, so the decision space changes with the state. Uncertainty: the opponent's hand and future draws are hidden, and draws or coin tosses can change which plan is possible. A lost game therefore does not mean that every choice made by its loser was bad. Adversarial play: the best action depends on the opponent's deck and likely response. A policy that handles one archetype well may struggle against another; as competing agents switch decks and improve, the opponent distribution also changes. The deck and matchup EDA below shows that this was visible in the replay data.

My solution was to learn from a large, varied set of games. Instead of writing a separate rule for every card interaction, I trained a Transformer to choose among the legal actions by predicting what strong replay players did in similar states. State tokens represent the visible game; action tokens represent the choices the simulator currently offers. This made action prediction the central training task and let the same model learn across decks and decision types.

The pipeline is:

1. Extract each decision's observation, legal options, selected action, player, result, deck, and replay identity from dated replay archives.
2. Convert state regions and candidate actions into model-ready features. Keep complete replays together when making holdout sets.
3. Train a Transformer encoder over state tokens and an action decoder over the legal candidates, using behavior cloning.
4. Compare configurations on in-distribution, later-date, expert-subset, and deck/archetype-isolation holdouts.
5. Train the chosen 3+3- and 4+4-layer configurations on the full selected data. At inference, average their action probabilities and select a legal action.

---

## Why behavior cloning

### Inspiration from GPT: learn the game by predicting actions

The idea came from the training logic of GPT. [*Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) uses next-token prediction on broad text to learn representations that transfer to other tasks. [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) explores how one language-modeling objective can support many different tasks. What interested me was the training signal: predicting the next item in a rich sequence forces the model to use the context preceding it.

I translated that idea to the game: given the visible state and legal choices, predict the next action selected in a replay. Choosing a Pokémon, attack, target, or card effect requires the model to connect the current choice with the board and previous decisions. Across many decks and opponents, this gives it a chance to learn reusable card roles and strategies. I selected stronger replays and checked performance on new games and later dates.

The same idea suggested learning broadly before specializing. I explored sampling, auxiliary tasks, and deck fine-tuning, then kept the broad replay-trained policy for the final agent.

### Why I chose offline imitation first

I considered reinforcement learning at the start. Self-play could eventually improve beyond the actions in the archive, but generating new games and learning from their delayed results would consume time I could instead spend training directly on existing decisions. The available replay archive already covered many cards, decks, and opponents. I therefore started with behavior cloning: it gave the policy a useful learning signal from every recorded decision and let me iterate quickly on data and model design.

[*Kickstarting Deep Reinforcement Learning*](https://arxiv.org/pdf/1803.03835) reinforced this choice. It combines guidance from previously trained expert policies with an RL objective, reducing the teachers' influence as the student improves. In its multi-teacher DMLab-30 experiment, the student reached a specified level of performance in 9.58× fewer environment frames than training from scratch. The lesson I took was to establish useful behavior from strong examples first, while leaving room for later improvement through self-play. For this competition I completed the imitation stage; the final agent does not include RL training.

The [original AlphaGo](https://research.google/pubs/mastering-the-game-of-go-with-deep-neural-networks-and-tree-search/) and [AlphaStar](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii) systems also began with recorded play before improving through RL. I focused on the demonstration-learning stage because the competition provided a large replay archive.

The model learns to predict the demonstrated legal action. I used action CE to compare training choices while keeping in mind that several actions may be reasonable in the same position and a few critical turns can decide a game.

### Replay selection and stronger players' losses

The decision I ultimately care about is the action that gives the best chance of winning from the information currently available:

$$a^*(o)=\underset{a\in\mathcal A(o)}{\arg\max}\;P(\mathrm{win}\mid o,a).$$

Here $o$ is the visible state and $\mathcal A(o)$ is the set of legal actions. The game result is a noisy signal about an earlier choice: hidden cards, future draws, coin tosses, and the opponent's responses can cause a player to lose even after choosing a strong action. That makes some actions in lost games relevant to the same goal as actions in won games. I expected these useful losing-game decisions to be more common among higher-scoring players, so I used score as a condition for including them rather than treating every loss alike.

The replay extractor keeps each player's actions and the game result. Winner actions form the core behavior-cloning data. I also included losing-player actions when both players passed the daily score cutoff. This adds decisions from difficult games between strong players, including positions a winner-only dataset may miss.

I limited this augmentation to the newest 25 eligible dates and used a daily top-player fraction of 6% to define the cutoff. Recent high-level games were especially relevant while the competing agents and deck choices were changing. The expert validation subset uses its own cutoff, so it answers a different question from the rule that admits losing-player actions.

I also explored whether the training mixture should favor the target deck or stronger players. A model may need broad games first to learn card mechanics and use rare expert examples better later. That led to oversampling and deck fine-tuning trials. Neither produced a sufficiently convincing improvement for the final training recipe, so I kept the broad data mixture.

---

## EDA: what the replay archive said before modeling

I used the replay archive to check deck coverage, changes in the opponent pool, matchup patterns, and inference time. These findings shaped the training data, validation splits, and final agent.

### 1. How many distinct games support each deck?

![Exact top-deck replay coverage: raw, training and expert-training replays](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/eda/top_deck_coverage.png)

The exact-deck chart counts raw, selected training, and expert-training replays. The largest Grimmsnarl deck has 63,433 / 34,571 / 3,640 games in those groups, while the exact Festival Lead deck has only 1,570 / 671 / 226. Common decks can dominate training, and the target deck has relatively few examples. This led me to track deck-specific validation and favor broad training over narrow fine-tuning.

![Same-archetype variants by changed card slots and replay coverage](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/eda/top_deck_archetype_coverage.png)

The companion chart compares each deck variant with a reference top deck by changed card slots; the vertical axis shows replay counts on a log scale. Alakazam has many observed variants, while Mega Froslass and Festival Lead show groups of both close and substantially changed builds. Expert games are sparse for many variants. A random replay split can therefore contain almost the same deck on both sides, so I added exact-deck and archetype holdouts to test transfer beyond familiar lists.

### 2. The opponent meta moved over time

![Archetype usage share and observed win rate across four replay dates](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/eda/archetype_share_and_win_rate.png)

From 8.12 to 8.15, Dragapult's replay share rose from 17.2% to 27.5%, Marnie Grimmsnarl's fell from 23.0% to 9.5%, and Festival Lead's grew from 10.6% to 16.3%. Popularity and results also differed: on 8.12, Marnie Grimmsnarl won about 41.3% of observed games, while Festival Lead won about 56.2%. The changing deck mix made latest-date validation especially useful.

![High-score-band archetype shares and team deck switches](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/eda/archetype_sankey.png)

The Sankey focuses on games where both players scored at least 1,060. Blocks show archetype use by date, and ribbons connect each team's most-used archetype on successive dates. Dragapult's share in this high-score group rose from 29.0% on 8.12 to 33.9% on 8.15. Stronger players faced a changing deck mix, which reinforced my decision to examine expert and latest-date validation separately.

![Archetype matchup win-rate matrix for score-filtered non-mirror games](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/eda/archetype_matchup_matrix.png)

The matchup matrix uses games where both players scored at least 1,000 and excludes mirror matches. Festival Lead won 71% of 451 observed games against Marnie Grimmsnarl, but 40% of 502 against Alakazam and 46% of 980 against Dragapult. These matchup differences made deck choice and the changing opponent pool important parts of the submission, alongside the policy itself.

### 3. Did replay timing make search unattractive?

![Startup time versus subsequent-action time for high-scoring teams](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/eda/score_filtered_startup_vs_action.png)

The timing chart compares startup and average action time for teams with `min_score >= 1100`. Only 13 teams met that threshold. Their action times suggested that competitive agents could work with relatively light inference, so I prioritized a fast policy ensemble over building a search system.

Together, the EDA pushed me toward broad replay coverage, whole-game and latest-date validation, separate deck-transfer checks, and fast inference.

---

## Game representation and model architecture

I represent the observation as game-state tokens and the legal choices as action tokens. This keeps the board objects separate while allowing each candidate action to inspect the entire visible state. The tables show the final feature layout; the diagram below shows how those features enter the Transformer.

### State tokens

The submitted models encode the visible game as 26 state tokens plus one token for the previous three actions. Separate tokens for Bench slots, Active Pokémon, player summaries, discard piles, hand, deck, Stadium, and global information let the encoder attend across game entities while preserving where each feature came from. For example, an Active Pokémon can attend to its attached energy and damage context alongside the opponent's board.

| Final state-token indices | Region | Information represented |
|---|---|---|
| 0–7, 8–15 | Own and opponent Bench | Individual Pokémon slots, attached cards, status, and appearance state |
| 16–17 | Own and opponent Active | Current Pokémon and its attached cards and state |
| 18–19 | Player summaries | Hand, deck, prize, and other player-level quantities; own and opponent inputs use separate projections |
| 20–21 | Own and opponent discard | Known discarded-card identities |
| 22–24 | Own hand, own remaining deck, Stadium | Known hand and public Stadium cards; deck information is an estimate from the submitted list and observed cards |
| 25 | Global and selection summary | Turn and current selection information, projected from numeric and one-hot fields |
| 26 | Recent-action history | One token summarizing the previous three choices; appended to the 26 base tokens |

The card-bearing regions use weighted embedding bags plus static card features; dense player/global summaries use separate MLPs. Region-specific token MLPs then bring the resulting representations into the shared 512-dimensional space. Empty Bench slots and an absent history token are masked in attention. Hidden opponent cards are never supplied as known card identities.

### Legal-action option tokens

I treat each decision as a question with candidate answers. The engine's `select` context identifies what kind of choice is being requested; an option carries its type, candidate/target card IDs, attack ID, area and relation, numeric positions, and available static and dynamic card/attack features. This matters in a card game because the same card identity can mean different things when it is a source, target, effect, or context card, and because the legal set changes at every decision. The implementation sums learned and projected feature channels into a 512-dimensional option vector, then applies configured region/option MLPs. Candidate combinations are explicitly enumerated and formed by summing their constituent option vectors. When no option is selected, a learned no-action vector is used. The policy scores at most 64 enumerated legal combinations.

| Option feature group | What it tells the model | Encoding |
|---|---|---|
| 11 categorical fields | Option type, selection context, candidate/target card, attack, number/count, player relation, areas, special condition | Learned embeddings |
| 5 numeric fields | Card/Tool/Energy/in-play indices and relative position in the offered option list | Normalized values through an MLP |
| Static card and attack features | Properties of the referenced cards and attack | Feature projections added to the option token |
| Dynamic Pokémon and attack features | The relevant Pokémon's current state and attack matchup | Separate projections, zero when unavailable |
| Multi-option action | The set of options selected together | Sum their option tokens; use a learned vector for an empty choice |

I also encoded an option's position in the offered list. This can help imitate replay players who favor the first of several similar choices. The effect/context-card experiment later in the write-up tested whether adding more card context helped the policy.

### Model architecture: encoder and candidate-query decoder

![Detailed architecture of the submitted two-model behavior-cloning policy](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/model_architecture.png)

The state path produces 27 contextual tokens: 26 current-state tokens and one history token. The option path produces one query for each enumerated legal action. Both paths use the same representation in the 3+3-layer and 4+4-layer models.

The encoder self-attends over the visible game entities. The decoder starts from one query per legal candidate action and cross-attends to the same encoded state at each decoder layer. There is deliberately no self-attention among candidate actions. A candidate's score can depend on the entire visible board, while the score for one candidate does not depend on which other candidates happen to be placed beside it in the decoder tensor. The last linear layer returns one logit per action. During BC training, cross-entropy compares these logits with the demonstrated action; during inference, invalid enumerations are masked and the surviving probabilities are compared.

Scoring the current legal options avoids a fixed output class for every possible card combination. The same state–action interaction can be used across different decision menus.

### Using the previous three actions

Short action sequences often form a combination: the next choice can depend on the card or target selected immediately before it. The model encodes the previous three actions, sums the options within each action, and combines the three action vectors into one history token. The final model uses the richest `full` history setting, including card, attack, static, and dynamic features. The history experiment below showed lower CE on all four tracked measures.

### Final model configuration

Both final models use the feature construction shown above. Their capacity differs only in the number of encoder and decoder layers.

Both submitted models use an encoder-decoder Transformer with 8 attention heads, FFN width 2,048, PostNorm, GEGLU FFNs, and dropout probability 0.1 with the configured dropout switches enabled. Their common input representation and policy head are the same; the layer count differs:

| Model | Encoder + decoder layers | Width | Trainable parameters |
|---|---:|---:|---:|
| A | 3 + 3 | 512 | 49,329,665 |
| B | 4 + 4 | 512 | 57,735,681 |

The saved weights are stored in FP16; I load them into FP32 models for inference. The policy has no value head.

---

## Validation design

Randomly splitting individual decisions would place highly related states from the same game on both sides of the split. The training pipeline therefore assigns whole replays to a split. It first removes selected deck/archetype isolation replays, then reserves the latest date, then deterministically partitions older remaining replays into training and in-distribution validation. The selector also supports exact-deck and top-deck-archetype isolation groups. Validation uses winner actions; the training pool can include qualified loser actions.

Each split answered a practical question. Ordinary held-out games checked familiar decks, the latest date checked a changing meta, and the expert subset checked stronger players. Exact-deck and archetype holdouts asked whether the model could handle unfamiliar card combinations and deck families.

The saved `expert_validation_ratio` is 0.05: a replay enters the expert validation group if either player clears the daily score cutoff. This group is separate from the training rule that adds loser actions only when both players clear its 6% cutoff.

The four CE series shown in this write-up are:

| Series | Meaning |
|---|---|
| Training CE | EMA of the main action-prediction loss, as logged during training |
| In-distribution CE | New replays held out from the older, non-isolated dates |
| Latest-date CE | Replays from the latest date excluded from holdout training |
| Latest-date expert CE | Expert subset of that latest-date validation group |

Lower CE means better action prediction. These plots show training and validation behavior; the competition evaluates games won.

---

## Experiments that shaped the solution

The table summarizes matched W&B comparisons at the last step shared by each pair. Negative CE differences favor the variant. Curves, values, and run links are in the [experiment report](https://github.com/Forrest-xlz/PTCG-kaggle/blob/main/docs/solution-experiment-results.md).

| Variant − control | Shared step | Train CE Δ | In-distribution CE Δ | Latest-date CE Δ | Latest expert CE Δ |
|---|---:|---:|---:|---:|---:|
| Small − Tiny, early full-replay split | 11,340 | −0.3064 | −0.0933 | −0.0060 | −0.0335 |
| Full history − no history | 9,720 | −0.0088 | −0.0216 | −0.0328 | −0.0204 |
| GEGLU − GELU | 9,720 | −0.0379 | +0.0021 | +0.0116 | +0.0024 |
| Full dropout − dropout off | 9,990 | +0.0880 | −0.0115 | −0.0253 | −0.0082 |
| Multitask − pure BC | 19,980 | −0.0011 | −0.0021 | −0.0037 | −0.0141 |
| effect/contextCard roles − no roles | 5,400 | −0.0015 | −0.0107 | +0.0026 | −0.0040 |

Each row compares one experiment with its own control.

### 1. Data coverage mattered more than fitting a small replay subset

GPT-style predictive pretraining suggested a simple question: would a policy improve as I gave it more replay data and more model capacity? I tested both axes, inspired by [GPT's pretraining paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf?file=language_understanding_paper.pdf) and [language-model scaling work](https://arxiv.org/abs/2001.08361).

I trained early Tiny (128 width, 1+1 layers) and Small (512 width, 3+3 layers) models on 1%, 10%, 30%, and 100% of selected replays.

![Data scaling: four CE metrics for Tiny and Small](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/01_data_scaling.png)

With 1% of replays, Small reached training CE 0.3985 but latest-date CE 2.1403. With the full replay pool, training CE was 0.6312 and latest-date CE 0.9156. The smaller dataset was easier to memorize but transferred much worse to recent games. This convinced me to keep broad replay coverage even when it made training CE higher. ([1% run](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/f1ecq0x8); [full-replay run](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/izjmqifz))

### 2. A larger model improved action prediction

On the full replay pool, Small improved over Tiny at their shared step 11,340: training CE −0.3064, in-distribution CE −0.0933, latest-date CE −0.0060, and latest-date expert CE −0.0335. The largest gains were in training and familiar validation games, but the result encouraged me to explore larger models.

![Model scaling: Tiny, Small, and Base](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/02_model_scaling.png)

I also tried a Base model. The early scale runs helped choose a workable model size; later experiments added the richer features and action history used by the final models.

### 3. Recent action history carried useful decision context

I compared no history, basic history, and a fuller encoding of the previous three actions. At step 9,720, full history improved training, in-distribution, latest-date, and latest-date expert CE by 0.0088, 0.0216, 0.0328, and 0.0204 over no history. It also beat basic history on all four measures, so I kept the full history token.

![Three-action history encodings](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/03_action_history.png)

### 4. Dropout helped validation more than a new activation

I tested GEGLU after reading [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202), then compared it with GELU on the replay splits.

GEGLU lowered training CE by 0.0379 relative to GELU at step 9,720, while its three validation CE values were slightly higher (+0.0021, +0.0116, +0.0024). The activation alone did not give a clear validation gain.

Adding dropout to GEGLU raised training CE by 0.0880 but lowered the three validation CE values by 0.0115, 0.0253, and 0.0082. That trade-off favored the GEGLU-plus-dropout setup used in the final models.

![Activation and dropout CE comparisons](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/04_activation_dropout.png)

### 5. Multitask learning helped early, with mixed late results

I wondered whether extra training targets could help the encoder understand game state beyond the immediate action. Inspired partly by the multitask idea in [ExT5](https://arxiv.org/abs/2111.10952), I added predictions for the next decision, the opponent's deck, and the final own prize count.

These three auxiliary losses used weights 0.08, 0.08, and 0.14. I had also considered predicting opponent strength, but did not include it in this run.

Early and mid-training, the multitask model had lower main-task training CE than pure behavior cloning. That lead shrank from about 0.0065 at step 1,080 to 0.0011 at step 19,980. At the shared final step, the four CE measures in the figure were still slightly better for multitask (by 0.0011, 0.0021, 0.0037, and 0.0141), but its top-deck-archetype isolation CE was worse: 0.578 versus 0.563. The late result was mixed rather than a clear win, so I kept the simpler action-only objective for the submission.

![Multitask versus pure behavior cloning](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/05_multitask.png)

### 6. Extra card features had mixed results

I tested extra IDs and static features for the card causing an effect and its context card. At step 5,400, training, in-distribution, and latest-date expert CE improved by 0.0015, 0.0107, and 0.0040, but latest-date CE worsened by 0.0026. The mixed result did not make these extra fields a priority for the final model.

![Effect and context-card feature comparison](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/07_effect_context.png)

One possible explanation is that more card IDs help the model remember familiar cards without improving how it handles new combinations.

---

## Decisions and ideas that did not become the final method

### Deck-specific fine-tuning did not help

I tried a language-model-style path: learn broadly from many games, then specialize on the strongest deck or players. Deck-specific fine-tuning worsened the validation groups I checked, including the target deck. I therefore kept broad training and left deck-specific fine-tuning, heavy expert oversampling, and expert-identity conditioning out of the submission.

The target deck had far fewer selected games than the most common decks in the EDA. Keeping the broad replay pool gave the final models more card interactions and opponent situations to learn from.

### Deck choice and opponents also mattered

The matchup chart showed why a strong policy still needs a sensible deck: Festival Lead performed differently against Grimmsnarl, Alakazam, and Dragapult. As opponents changed decks, the challenge changed too. I considered following high-score teams' deck choices to anticipate the meta, but the submitted agent uses one Festival Lead deck throughout.

I also considered that early results might change an agent's rating band and thus the opponents it faces later. This remained an idea rather than a strategy in the final submission.

### What I would test next: actual games

My next step would be to play candidate models against a fixed mix of decks and opponents, then review the turns that changed the result. Action CE helped me iterate, but an agent ultimately has to win games. This would show whether better history features, deck specialization, or a larger model actually improve play.

I would then revisit the original BC-to-RL idea: start self-play from the replay-trained policy and compare it with the same fixed opponents. That would test whether reinforcement learning can improve on the imitation starting point.

---

## Final training and inference

After choosing the setup, I trained the final 3+3- and 4+4-layer policies on the full selected replay pool. They share the same features and training configuration apart from Transformer depth.

At inference, both models score the same enumerated legal actions. I apply softmax to each model's FP32 logits, average their probability vectors equally, apply the damage-counter KO eligibility mask, and take the highest-scoring remaining action. The agent retains the previous three actions between decisions and uses a fixed 60-card Festival Lead deck. It does not run a search loop.

Earlier validation runs favored ensembling on several subsets, so I combined the two final models.

---

## What I learned

Broad replay coverage mattered most: a model trained on 1% of selected replays fit its training actions well but struggled on recent games. The previous-three-action token helped across all four CE measures, and dropout improved validation despite a higher training loss. I used the larger Transformer setup and combined two model depths for the final submission.

The main lesson was to learn game decisions from diverse replays first, then use recent and expert games to check whether the policy transfers. The next improvement should be measured in actual battles, with self-play RL as a possible continuation of the replay-trained policy.

---

## Sources and experiment details

- [Competition description and evaluation](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/evaluation); [Pokémon TCG rulebook](https://www.pokemon.com/static-assets/content-assets/cms2/pdf/trading-card-game/rulebook/par_rulebook_en.pdf).
- [Exact experiment values, curves, and W&B run links](https://github.com/Forrest-xlz/PTCG-kaggle/blob/main/docs/solution-experiment-results.md).
- [GPT generative pretraining](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), [GPT-2 multitask behavior](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), and [Kickstarting Deep Reinforcement Learning](https://arxiv.org/pdf/1803.03835).
