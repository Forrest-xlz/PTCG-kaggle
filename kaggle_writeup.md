# The Pokémon Company – PTCG AI Battle Challenge Simulation: Transformer Behavior Cloning

Final rank: 59 / 6,807 teams. This is my solution to the [Kaggle simulation competition](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/). The final submission is a two-model Transformer policy ensemble trained on game replays. I used supervised next-action prediction rather than an online reinforcement-learning or search pipeline.

The experiments below follow how I chose the data, representation, and training setup. Their four curves measure action-prediction cross-entropy (CE) on training and held-out replays; battle strength is discussed separately.

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

I translated that idea to the game: given the visible state and legal choices, predict the next action selected in a replay. To choose a Pokémon, attack, target, or card effect well, the model has to connect the choice to the board, the current selection question, and previous decisions. Across many decks and opponents, that task may encourage reusable knowledge about card roles and strategy rather than one rule per card. Unlike text, however, replay actions come from players of varying strength and several actions may be equally good. I therefore used strong-player filtering and multiple holdouts to judge what the model learned.

The same GPT papers suggested a broad-to-specific path: learn general card and decision relationships from diverse games, then see whether a narrower deck or expert distribution needs extra attention. I explored sampling, auxiliary tasks, and fine-tuning with that goal in mind. The final agent retained the broad replay-trained policy because the specialized trials did not produce a clear enough improvement to use.

### Why I chose offline imitation first

I considered reinforcement learning at the start. Self-play could eventually improve beyond the actions in the archive, but generating new games and learning from their delayed results would consume time I could instead spend training directly on existing decisions. The available replay archive already covered many cards, decks, and opponents. I therefore started with behavior cloning: it gave the policy a useful learning signal from every recorded decision and let me iterate quickly on data and model design.

[*Kickstarting Deep Reinforcement Learning*](https://arxiv.org/pdf/1803.03835) reinforced this choice. It combines guidance from previously trained expert policies with an RL objective, reducing the teachers' influence as the student improves. In its multi-teacher DMLab-30 experiment, the student reached a specified level of performance in 9.58× fewer environment frames than training from scratch. The lesson I took was to establish useful behavior from strong examples first, while leaving room for later improvement through self-play. For this competition I completed the imitation stage; the final agent does not include RL training.

The [original AlphaGo](https://research.google/pubs/mastering-the-game-of-go-with-deep-neural-networks-and-tree-search/) and [AlphaStar](https://deepmind.google/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii) systems also began by learning from recorded play before improving through RL. They showed a possible longer-term path for this project. I prioritized the part that fit the available replays and development time: learning a strong policy from demonstrations. I did not use search or RL in the submitted solution.

The training objective is the CE of the demonstrated legal action. It is a useful proxy for learning decisions from strong play, but it has an important limitation: actions that match a replay are not necessarily the only winning actions, and a small number of errors at critical turns can matter more than many easy decisions. I therefore treated offline CE as a way to select experiments, not as a direct measure of battle strength.

### Replay selection and stronger players' losses

The decision I ultimately care about is the action that gives the best chance of winning from the information currently available:

$$a^*(o)=\underset{a\in\mathcal A(o)}{\arg\max}\;P(\mathrm{win}\mid o,a).$$

Here $o$ is the visible state and $\mathcal A(o)$ is the set of legal actions. The game result is a noisy signal about an earlier choice: hidden cards, future draws, coin tosses, and the opponent's responses can cause a player to lose even after choosing a strong action. That makes some actions in lost games relevant to the same goal as actions in won games. I expected these useful losing-game decisions to be more common among higher-scoring players, so I used score as a condition for including them rather than treating every loss alike.

The replay extractor keeps each player's actions and the game result. Winner actions form the core behavior-cloning data. I additionally included losing-player actions from games in which both players passed the daily score cutoff. Requiring two strong participants focuses this augmentation on competitive games, while retaining difficult, losing positions that a winner-only dataset may miss. Behavior cloning learns to predict these demonstrated choices; the win-probability expression above describes the strategic goal behind the data selection, not a win-probability label available for every action.

I limited this augmentation to the newest 25 eligible dates and used a daily top-player fraction of 6% to define the cutoff. Recent high-level games were especially relevant while the competing agents and deck choices were changing. The expert validation subset uses its own cutoff, so it answers a different question from the rule that admits losing-player actions.

I also explored whether the training mixture should favor the target deck or stronger players. A model may need broad games first to learn card mechanics and use rare expert examples better later. That led to oversampling and deck fine-tuning trials. Neither produced a sufficiently convincing improvement for the final training recipe, so I kept the broad data mixture.

---

## EDA: what the replay archive said before modeling

The EDA changed which data I trusted, which holdouts I built, and how I interpreted the leaderboard. It covers replay coverage by deck, how the meta moved over time, matchup patterns, and inference-time clues. The charts describe the observed replay population; the model experiments below test separate questions about action prediction.

### 1. How many distinct games support each deck?

![Exact top-deck replay coverage: raw, training and expert-training replays](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/eda/top_deck_coverage.png)

The exact-deck coverage chart counts unique replays, in three increasingly selective pools: all raw replays, selected training replays, and expert-qualified training replays. This is a necessary check before trusting a narrow deck-specific CE score. The largest exact Grimmsnarl deck has 63,433 raw / 34,571 training / 3,640 expert-training replays, whereas the exact Festival Lead deck has 1,570 / 671 / 226. A highly frequent deck can dominate ordinary training and aggregate validation, while a target deck with only hundreds of selected games yields noisier specialized conclusions. This motivated named top-deck subsets, minimum coverage requirements for isolation splits, and caution about fine-tuning a broad model on a small exact deck.

![Same-archetype variants by changed card slots and replay coverage](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/eda/top_deck_archetype_coverage.png)

The companion chart places each same-archetype deck variant at its number of changed card slots relative to a reference top deck, with a log-scaled vertical axis for replay counts. It distinguishes exact-deck familiarity (distance zero) from exposure to related builds. The variants form very different patterns: Alakazam has many observed variants across moderate card changes; Mega Froslass and Festival Lead have separated groups of related and substantially altered builds; expert-qualified variants are sparse in many cells. Thus a random replay split can test a new game while still seeing nearly the same deck. I added exact-deck and archetype isolation to ask harder transfer questions, with minimum sample checks to avoid treating a tiny cell as a stable result. A “changed slot” is a multiset card-count difference, not proof that two decks play alike.

### 2. The opponent meta moved over time

![Archetype usage share and observed win rate across four replay dates](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/eda/archetype_share_and_win_rate.png)

The daily trend figure uses the 8.12–8.15 snapshots with no score filter and plots each archetype's share of player appearances and observed win rate. Dragapult's share rose from 17.2% to 27.5%; Marnie Grimmsnarl's fell from 23.0% to 9.5%; Festival Lead's grew from 10.6% to 16.3%. Popularity and observed win rate are visibly different quantities: on 8.12 the frequently used Marnie Grimmsnarl group won about 41.3%, while the less common Festival Lead group won about 56.2% in this replay sample. These are conditional replay-sample rates, affected by opponent mix and player strength. The shifting usage made the latest-date holdout substantively useful: it checks a policy against a later mix of states rather than only new replays from the same period.

![High-score-band archetype shares and team deck switches](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/eda/archetype_sankey.png)

The Sankey narrows to replays whose minimum player score is at least 1,060. Its colored blocks show each day's archetype use in that filtered pool; ribbons connect the modal archetype of the same team on successive dates, rather than tracking individual matches or card substitutions. In the plotted high-score sample, Dragapult rises from 29.0% on 8.12 to 33.9% on 8.15. The figure supports the idea that the high-score opponent mix can differ from the full replay population and that teams sometimes switch decks. It does not demonstrate that high-score behavior predicts the broader population or that copying one switch would improve our ranking. It reinforced my decision to keep expert and time-based validation slices separate.

![Archetype matchup win-rate matrix for score-filtered non-mirror games](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/eda/archetype_matchup_matrix.png)

The matchup matrix uses non-mirror replays with minimum player score at least 1,000 over the same dates. Each cell is the row archetype's observed win rate against the column archetype, with the matchup count shown. For example, Festival Lead won 71% of 451 observed games against Marnie Grimmsnarl, but 40% of 502 against Alakazam and 46% of 980 against Dragapult. Those contrasts explain why I separated policy strength, submitted deck choice, opponent distribution, and meta adaptation in this solution. A single aggregate win rate can change as the matchup mix changes even if the agent itself does not. The observational matrix is not a counterfactual deck-selection experiment; player strength, dates, and strategy versions differ between cells.

### 3. Did replay timing make search unattractive?

![Startup time versus subsequent-action time for high-scoring teams](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/eda/score_filtered_startup_vs_action.png)

The replay-timing script aggregates startup time and mean subsequent action time per team, uses log-scaled axes, and fits four K-means clusters on log-transformed times for all teams before assigning the score-filtered subset. In the saved analysis, the global cluster centers range from roughly 0.54 or 10.4 seconds startup and 0.10–0.14 or 2.25 seconds per subsequent action; only 13 teams pass `min_score >= 1100`, and none belongs to the slow-action/fast-start global cluster. My competition-time interpretation was that some strong entrants might be using relatively light inference, so I lowered the priority of a heavy search implementation and tested an ensemble instead. That is only an indirect resource clue. Replay timing includes runtime, implementation, hardware, loading, and game-state effects; it cannot reveal whether another team used search or how many nodes it explored.

The EDA therefore led to four concrete modeling choices: use broad replay coverage; validate by whole replay and by later date; examine deck/archetype transfer separately; and keep the inference path light enough for the competition setting. The timing analysis informed the priority I gave to search; it did not settle whether search would have helped.

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

The numeric position fields are also a possible shortcut: replay-generating agents may systematically choose the first of equivalent options. I suspected that relative position could raise imitation accuracy without changing game strength. I did not establish a controlled game-level test of that hypothesis, so the feature remains an implementation detail rather than a claimed performance mechanism. Likewise, the effect/context-card experiment later in this write-up gave mixed validation signals; having access to more observable information did not guarantee better transfer to a new card family.

### Model architecture: encoder and candidate-query decoder

![Detailed architecture of the submitted two-model behavior-cloning policy](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/model_architecture.png)

The state path produces 27 contextual tokens: 26 current-state tokens and one history token. The option path produces one query for each enumerated legal action. Both paths use the same representation in the 3+3-layer and 4+4-layer models.

The encoder self-attends over the visible game entities. The decoder starts from one query per legal candidate action and cross-attends to the same encoded state at each decoder layer. There is deliberately no self-attention among candidate actions. A candidate's score can depend on the entire visible board, while the score for one candidate does not depend on which other candidates happen to be placed beside it in the decoder tensor. The last linear layer returns one logit per action. During BC training, cross-entropy compares these logits with the demonstrated action; during inference, invalid enumerations are masked and the surviving probabilities are compared.

This design answers a specific representation problem. A fixed action-class classifier would need to encode every possible card combination and context in one global output space. Scoring the currently legal options lets the model transfer a learned interaction such as “this target, in this state, makes this option appropriate” across many specific decision menus. The argument is architectural; there is no ablation here proving that the candidate-query decoder is superior to every simpler baseline.

### Using the previous three actions

The present observation does not always expose the intent behind a sequence of choices. I thought of short action chains as “combos”: a card selection may only make sense in relation to the immediately preceding selection. The model encodes up to three previous actions with independent history embeddings, sums option rows within each historical action, applies a per-action MLP, concatenates the three vectors in chronological order, and maps them through a sequence MLP to a single 27th encoder token. `basic`, `structural`, and `full` history modes include progressively richer fields. The final model uses `full`, including card, attack, static, and dynamic history features, while excluding the five option-position numerics from the history representation. I initially considered a 1D CNN, but the final implementation uses concatenation plus MLP. The history ablation below supports retaining this context; it does not identify which individual historical field caused the gain.

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

This evaluation design grew in layers rather than from one headline number. A replay-level random holdout asks whether the model can imitate actions in a new game from the familiar time and deck mix. A latest-date holdout asks whether it can handle a later population; the EDA shows that this population can shift quickly. A daily high-score subset asks whether the policy tracks stronger observed players, but it is still judged by their chosen actions, not by the value of counterfactual alternatives. Exact-deck isolation tests a new 60-card combination whose components may be familiar; archetype isolation makes the distribution shift harder by withholding a core deck family. Top-deck-archetype isolation withholds related variants so that matching a reference list is not mistaken for transfer to its family. Where possible, the selector audits replay count and card coverage after the isolation groups are combined.

These splits answer different questions, and none alone gives an unbiased estimate of submission win rate. The latest-date set may also change in player quality, deck composition, and option frequencies at the same time. Isolation groups can be small and can overlap each other, even though their union is excluded from training. The saved `expert_validation_ratio` is 0.05, defining a daily high-score cutoff across participants; a replay enters the expert group if either participant clears it. That is not identical to taking the top 5% of validation actions, and it is distinct from the 6% loser-augmentation threshold that requires both players to qualify. This distinction is important when reading `latest expert CE` in the figures.

The four CE series shown in this write-up are:

| Series | Meaning |
|---|---|
| Training CE | EMA of the main action-prediction loss, as logged during training |
| In-distribution CE | New replays held out from the older, non-isolated dates |
| Latest-date CE | Replays from the latest date excluded from holdout training |
| Latest-date expert CE | Expert subset of that latest-date validation group |

Lower CE is better. The logged training EMA is not the total main-plus-auxiliary loss in the multitask experiment. I also monitored action Top-k accuracy and isolation groups, but the figures here deliberately focus on these four CE measures. None is a measured game win rate or Elo. Full-data final training does not preserve the same independent holdouts; evaluating its checkpoint on reconstructed holdouts it may have seen would not be a valid generalization estimate.

---

## Experiments that shaped the solution

The comparisons use W&B history rather than run names alone. For each paired table, I selected the last optimizer step at which both runs recorded all four CE metrics. A positive delta means the experimental variant had higher, worse CE. Runs are single configurations without repeated seeds; several early scale runs ended before their planned step count. The raw curves, exact values, and run links are available in the [experiment report](https://github.com/Forrest-xlz/PTCG-kaggle/blob/main/docs/solution-experiment-results.md).

| Variant − control | Shared step | Train CE Δ | In-distribution CE Δ | Latest-date CE Δ | Latest expert CE Δ |
|---|---:|---:|---:|---:|---:|
| Small − Tiny, early full-replay split | 11,340 | −0.3064 | −0.0933 | −0.0060 | −0.0335 |
| Full history − no history | 9,720 | −0.0088 | −0.0216 | −0.0328 | −0.0204 |
| GEGLU − GELU | 9,720 | −0.0379 | +0.0021 | +0.0116 | +0.0024 |
| Full dropout − dropout off | 9,990 | +0.0880 | −0.0115 | −0.0253 | −0.0082 |
| Multitask − pure BC | 19,980 | −0.0011 | −0.0021 | −0.0037 | −0.0141 |
| effect/contextCard roles − no roles | 5,400 | −0.0015 | −0.0107 | +0.0026 | −0.0040 |

The rows are separate experiments at different development stages; their CE levels and deltas must not be ranked against each other as if they shared one training set and architecture. The table is a compact index to the matched run pairs, not a factorial ablation of the final submission.

### 1. Data coverage mattered more than fitting a small replay subset

The plan to vary both replay fraction and Transformer capacity came from my broader question about whether action prediction is a scalable way to learn the game. [GPT's generative pretraining paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf?file=language_understanding_paper.pdf) was an analogy for learning a useful representation through prediction; [neural-language-model scaling laws](https://arxiv.org/abs/2001.08361) suggested looking at data, model size, and compute together. Neither paper supplies a scaling law for this simulator. I used their experimental questions, then checked what the PTCG runs actually showed.

I began with Tiny (128 width, 1+1 layers) and Small (512 width, 3+3 layers) V1 policies, using 1%, 10%, 30%, or 100% replay-selection fractions. The percentages describe replay selection, not exactly the same percentage of individual actions. Across fractions, epoch counts and learning-rate schedules differed, so the terminal values are descriptive rather than a clean scaling-law experiment.

![Data scaling: four CE metrics for Tiny and Small](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/01_data_scaling.png)

The most revealing run is the Small model trained on 1% of replays. Its logged training CE reached 0.3985, while latest-date CE rose to 2.1403 by the observed endpoint. With the full replay split, the Small model's terminal training CE was 0.6312 and latest-date CE 0.9156. The smaller pool made the training objective easier to fit, but its holdout behavior became much worse. That directly corrected my early intuition that a lower training loss alone would always signal a better policy. The [1% run](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/f1ecq0x8) stopped at 810 of 2,800 planned steps; the [full-split run](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/izjmqifz) stopped at 11,610 of 13,300, so I do not compare them as equal-compute endpoints.

### 2. A larger model improved action prediction

In the early full-replay experiment, Small had lower CE than Tiny at the last common four-metric evaluation, step 11,340: training −0.3064, in-distribution −0.0933, latest-date −0.0060, and latest-date expert −0.0335. Tiny and Small both used batch size 4,096. This supported testing a larger policy, while the small latest-date CE difference cautioned against treating capacity as an unconditional gain.

![Model scaling: Tiny, Small, and Base](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/02_model_scaling.png)

The Base run used a different batch size (2,048) and ended after fewer processed actions. Its endpoint is informative about that run, but it cannot isolate the effect of scaling width and depth. The final models have the later V2 feature layout and should not be presented as direct outcomes of these early V1 curves.

### 3. Recent action history carried useful decision context

At step 9,720, the full three-action history variant had lower CE than history-off: training −0.0088, in-distribution −0.0216, latest-date −0.0328, and latest-date expert −0.0204. Relative to the Basic history variant, the same four differences were −0.0046, −0.0082, −0.0119, and −0.0157. The comparison supports including recent decisions when scoring the current choice; it does not prove which individual historical field is responsible.

![Three-action history encodings](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/03_action_history.png)

### 4. Activation and dropout involved a training–validation trade-off

I tested GEGLU because [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202) reports improvements from gated Transformer feed-forward blocks in its own setting. That paper was a reason to try the activation, not a reason to declare it superior here before looking at holdouts.

At the shared step 9,720, GEGLU lowered training CE by 0.0379 relative to GELU, but its in-distribution, latest-date, and latest-date expert CE were higher by 0.0021, 0.0116, and 0.0024. I therefore cannot claim that GEGLU won on every validation metric. The ReLU run lasted only 270 optimizer steps and is not an equally trained third comparison.

In the separate GEGLU dropout comparison at step 9,990, full dropout raised training CE by 0.0880 but lowered in-distribution, latest-date, and latest-date expert CE by 0.0115, 0.0253, and 0.0082. This was a clearer regularization signal. The final checkpoint configuration uses GEGLU with dropout; its selection reflects the broader development process and submitted setup, not a claim that the GEGLU-only CE ablation unanimously favored it.

![Activation and dropout CE comparisons](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/04_activation_dropout.png)

### 5. Auxiliary tasks did not establish a decisive advantage

The motivation was about how much information the training objective asks the network to use. A replay contains more than the chosen action: the next decision type, opponent deck, and later outcome are related to state understanding. If the action label can often be predicted from a shallow shortcut, ordinary BC may not force the encoder to represent those other relationships. [ExT5](https://arxiv.org/abs/2111.10952) was one of the multi-task references in my reading list, but its language-task results do not predict a benefit for these game labels. I explored multitask supervision to encourage a richer state representation without collecting new raw games.

I considered next Select Type/Context, opponent deck, opponent strength, and prize-related signals. This raised a useful optimization question: an auxiliary weight scales its gradient, while gradient norm and cosine with the main-task gradient determine whether the shared encoder receives reinforcing or conflicting updates. A fixed auxiliary weight might help early representation learning but distract the policy late; reducing the weight over time was an untested follow-up. The actual logged run used a CLS representation and auxiliary targets for the next decision, opponent deck, and final own prize count, with weights 0.08, 0.08, and 0.14. Opponent-strength prediction remained an idea; it was not a head in this experiment. “Final own prize count” is also not the same label as the originally proposed *prize-count difference*.

Compared with matched pure BC at step 19,980, the multitask variant's training, in-distribution, latest-date, and latest-date expert CE differences were −0.0011, −0.0021, −0.0037, and −0.0141. These are small single-run differences. The training curves suggest a modest early or middle-stage benefit whose margin narrows later; they do not support my original shorthand that the main-task training CE had clearly become worse at the end. Multitask learning was an informative experiment, not a component of the two submitted checkpoints.

![Multitask versus pure behavior cloning](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/05_multitask.png)

### 6. More card identity features were not a uniform improvement

Adding both effect and context-card role IDs plus static-card projections yielded a mixed result at step 5,400. Relative to the matched variant without them, training CE changed by −0.0015, in-distribution CE by −0.0107, latest-date CE by +0.0026, and latest-date expert CE by −0.0040. This is not a clean argument that the extra fields either help or hurt every distribution. Because several inputs changed together, the comparison also cannot assign the effect to either card role alone.

![Effect and context-card feature comparison](https://raw.githubusercontent.com/Forrest-xlz/PTCG-kaggle/main/docs/figures/07_effect_context.png)

One possible mechanism: extra card IDs could encourage memorization of card identity rather than transferable card relationships. That remains a hypothesis. The current four-CE comparison does not demonstrate it, and the archetype-isolation behavior would need a matched comparison for a quantitative conclusion.

---

## Decisions and ideas that did not become the final method

### Specialized adaptation was harder than the pretraining analogy suggested

My initial mental model was “learn general PTCG first, then specialize to the strongest deck or strongest demonstrators.” This sounds like a natural continuation of language-model pretraining, but the target here is an interactive policy. Fine-tuning changes the distribution of decisions the model makes, so an offline action metric on the narrow deck can improve while match outcomes or off-deck behavior worsen. Deck-specific fine-tuning degraded the observed validation groups, including the target group. I cannot give a clean numeric conclusion from that attempt because its controls and training budget do not meet the standard used for the six figures. I therefore did not carry specialized fine-tuning, high expert oversampling, or explicit expert-identity conditioning into the final submission.

That failed direction still informed the final choice: use the full selected replay pool for the two submitted models, preserve broad coverage, and avoid presenting the target-deck distribution as if it were the whole game. It also sharpened an open question: is the obstacle sample scarcity, an unstable fine-tuning schedule, conflicting deck strategies, a missing observation feature, or simply CE's poor alignment with win rate? The current repository cannot separate those explanations. A better follow-up would use matched total updates, a fixed target-deck holdout, at least one non-target holdout, repeated seeds, and paired game evaluations.

### A good agent is only one term in a changing competition environment

The deck-trend and matchup EDA led me to think of leaderboard performance as an interaction among four quantities: agent decisions, submitted deck, encountered opponent distribution, and adaptation to that distribution. This is a conceptual decomposition, not an additive formula. For an opponent mixture $p_t(j)$ at time $t$, even a fixed policy/deck pair $(\pi,d)$ has a mixture-dependent expected win rate,

$$W_t(\pi,d)=\sum_j p_t(j)\,P(\text{win}\mid\pi,d,j).$$

The matchup matrix makes the dependence on $j$ concrete. It also explains why latest-date CE and exact-deck CE answer different questions from actual battle performance. I considered whether high-score teams' deck choices might be an early signal of a later meta shift, and whether a counter-meta deck could help. The final agent uses one Festival Lead deck throughout; it does not switch decks online.

I also considered a rating-path hypothesis: early random outcomes might put an agent into a rating band with a different opponent/deck mix, which could affect later scores. That would require verification of the competition's matchmaking and rating rules plus repeated, controlled submissions or match logs. I did not use it to explain the final rank or to claim the same agent would score differently solely from initial luck.

### The missing experiment was a game-level evaluation loop

I built increasingly careful offline validation, but I did not close the loop from action CE to games won. A stable opponent pool should include different decks, strengths, and policy families; each comparison should use paired seeds or mirrored sides where the simulator permits, enough repeated games for uncertainty intervals, and replay review of pivotal mistakes. In particular, aggregate Top-1 accuracy can reward predictable forced or nearly equivalent choices while underweighting a small number of decisive turns. A future analysis would stratify errors by game phase, option type, deck, matchup, and estimated consequence. It would also test whether history and representation gains persist outside the demonstration distribution.

Only after establishing such a loop would I revisit the literature-inspired BC→RL transition: initialize from the broad replay policy, measure the cost of self-play and opponents, retain an expert anchor or distillation term if helpful, and evaluate the resulting policy against the same fixed opponent pool. The literature suggests a possible route; this competition's recovered experiments support the BC starting point, not the unrun RL stage.

---

## Final training and inference

After development on holdout runs, I trained the final 3+3- and 4+4-layer policies with the project's full-data training mode. In that mode all cached dates are eligible for training selection, and training-time validation is skipped; winner filtering, eligible loser augmentation, and replay-selection settings still apply. The submitted checkpoints use the same main model configuration apart from Transformer depth. I do not infer an unseen-set score from their full-data training logs.

At inference, both models score the same enumerated legal actions. I apply softmax to each model's FP32 logits, average their probability vectors equally, apply the damage-counter KO eligibility mask, and take the highest-scoring remaining action. The agent retains the previous three actions between decisions and uses a fixed 60-card Festival Lead deck. It does not run a search loop.

Earlier offline validation suggested that ensembling improved several subsets, which motivated the final two-model combination. I did not measure the exact final pair against each constituent model in enough games to attach a battle-strength gain to the ensemble.

---

## Closing assessment

The recovered evidence supports a narrower, more useful story than “a bigger Transformer always wins.” Broad replay coverage prevented an apparent training-loss success from becoming severe latest-date overfitting. A three-action history token improved all four matched CE measures. Dropout traded harder training fit for better holdout CE. Capacity, activation, extra card-role features, and auxiliary tasks had either uneven, small, or confounded comparisons. The final ensemble was the implemented solution, but its exact game-level gain over either constituent model remains unmeasured here.

The most important change in my thinking was to separate the objective I optimized, the holdouts I used to select models, and the match outcomes I actually cared about. The broad action-prediction objective was chosen because it might force useful transfer across decks and card interactions, much as a broad predictive pretraining objective can teach reusable representations. The results show meaningful offline learning, but they do not prove general strategic understanding. My next research iteration would begin with the missing game-level evaluation loop, then test whether the model's broad representation can be improved by better replay mixtures or a carefully measured RL continuation.

---

## Sources and experiment details

- [Competition description and evaluation](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview/evaluation); [Pokémon TCG rulebook](https://www.pokemon.com/static-assets/content-assets/cms2/pdf/trading-card-game/rulebook/par_rulebook_en.pdf).
- [Exact experiment values, curves, and W&B run links](https://github.com/Forrest-xlz/PTCG-kaggle/blob/main/docs/solution-experiment-results.md).
- [GPT generative pretraining](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf), [GPT-2 multitask behavior](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), and [Kickstarting Deep Reinforcement Learning](https://arxiv.org/pdf/1803.03835).
