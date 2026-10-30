# The Pokémon Company – PTCG AI Battle Challenge Simulation: Transformer Behavior Cloning

**Final rank: 59 / 6,807 teams.** This is my solution to the [Kaggle simulation competition](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/). The final submission is a two-model Transformer policy ensemble trained on game replays. I used supervised next-action prediction rather than an online reinforcement-learning or search pipeline.

The experiments below report **offline action-prediction cross-entropy (CE)**, not game win rate. They explain the route to the solution, but the plotted holdout runs are not measurements of the final full-data checkpoints. The six figures and their exact W&B run matches are also collected in the [experiment report](docs/solution-experiment-results.md).

---

## Overview

The competition agent receives an observation and a list of legal options, and must select an action. A Pokémon TCG game contains hidden information, many card interactions, varying decks, and decisions whose consequences span several turns. My central bet was that a model trained to predict actions from many replayed games could learn reusable relationships between the state, the current selection question, and candidate answers.

The pipeline is:

1. Extract each decision's observation, legal options, selected action, player, result, deck, and replay identity from dated replay archives.
2. Convert state regions and candidate actions into model-ready features. Keep complete replays together when making holdout sets.
3. Train a Transformer encoder over state tokens and an action decoder over the legal candidates, using behavior cloning.
4. Compare configurations on in-distribution, later-date, expert-subset, and deck/archetype-isolation holdouts.
5. Train the chosen 3+3- and 4+4-layer configurations on the full selected data. At inference, average their action probabilities and select a legal action.

The final notebook is [submission.ipynb](submission.ipynb). Its two local checkpoint archives are `archive (3).zip` (`final_d2_33.pt`) and `archive (2).zip` (`final_d2_44.pt`). See [README.md](README.md) for the repository workflow. The competition-process notes that motivated the experiments are in [Notion](https://app.notion.com/p/39f888c4bb6180d1a714f6a930465490).

---

## Why behavior cloning

I initially considered reinforcement learning, but environment interaction was expensive relative to reusing the available replay data. My early estimate of a roughly order-of-magnitude efficiency advantage for behavior cloning was a **planning judgment**, not a controlled benchmark of equally tuned RL and BC systems. It led me to prioritize a supervised policy and spend the available compute on data coverage, representation, and model capacity.

The training objective is the CE of the demonstrated legal action. It is a useful proxy for learning decisions from strong play, but it has an important limitation: actions that match a replay are not necessarily the only winning actions, and a small number of errors at critical turns can matter more than many easy decisions. I therefore treated offline CE as a way to select experiments, not as a direct measure of battle strength.

### Replay selection and stronger players' losses

The extraction pipeline retains actions from both players, with each player's result recorded. Training selects winner actions and can additionally include loser actions from replays where **both players pass a daily score threshold**. This matters because a strong player can make sound decisions in a game ultimately lost to hidden information or randomness. It also broadens the situations seen during training, including difficult positions that winner-only sampling may underrepresent. It does **not** imply that every move in a strong player's loss is optimal.

The code implements this as configurable loser augmentation over recent eligible dates; the daily cutoff and date count are training choices. A separate expert threshold defines expert validation subsets. These two thresholds have different meanings and should not be conflated. The pipeline details are documented in [data-pipeline.md](docs/data-pipeline.md) and [training-validation.md](docs/training-validation.md).

---

## Representation and policy model

The model represents the **observable** game state, not the opponent's hidden cards. The final feature layout has 26 base encoder tokens: both players' Bench and Active Pokémon, player and global summaries, discard, the acting player's hand and deck-related information, and Stadium. Empty Bench positions are masked in attention. Categorical identities, numeric summaries, static card and attack attributes, and local Pokémon/attack dynamics are projected into a shared 512-dimensional space.

The current selection question and each legal option are encoded separately. Candidate options include categorical fields such as option type, context, card and attack IDs, and locations, alongside numerical and dynamic features. For a multi-option choice, the selected option representations are combined into one candidate action representation. The implementation enumerates up to 64 legal combinations. The action decoder scores these candidates against the encoded state; invalid choices are masked before the policy decision.

The final model also receives the **previous three actions** through an additional history token. Each historical action is encoded, the three representations are concatenated in chronological order, and MLP layers map them to one token. My original notes described this as a 1D CNN idea, but the saved training and submission implementations use **concatenation plus MLP**, so that is the architecture described here. The current model details are in [model.md](docs/model.md).

Both submitted models use an encoder-decoder Transformer with 8 attention heads, FFN width 2,048, PostNorm, GEGLU FFNs, and dropout probability 0.1 with the configured dropout switches enabled. Their common input representation and policy head are the same; the layer count differs:

| Submitted checkpoint | Encoder + decoder layers | Width | Trainable parameters |
|---|---:|---:|---:|
| `final_d2_33.pt` | 3 + 3 | 512 | 49,329,665 |
| `final_d2_44.pt` | 4 + 4 | 512 | 57,735,681 |

These specifications were checked against the checkpoint configurations and the model classes embedded in [submission.ipynb](submission.ipynb). The checkpoints contain FP16 tensors; the notebook instantiates FP32 models and loads the weights into them for inference. There is no value head in the submitted policy.

---

## Validation design

Randomly splitting individual decisions would place highly related states from the same game on both sides of the split. The training pipeline therefore assigns **whole replays** to a split. It first removes selected deck/archetype isolation replays, then reserves the latest date, then deterministically partitions older remaining replays into training and in-distribution validation. The selector also supports exact-deck and top-deck-archetype isolation groups. Validation uses winner actions; the training pool can include qualified loser actions.

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

The comparisons use W&B history rather than run names alone. For each paired table, I selected the **last optimizer step at which both runs recorded all four CE metrics**. A positive delta means the experimental variant had higher, worse CE. Runs are single configurations without repeated seeds; several early scale runs ended before their planned step count. The raw curves, exact values, and run links are available in the [experiment report](docs/solution-experiment-results.md).

| Variant − control | Shared step | Train CE Δ | In-distribution CE Δ | Latest-date CE Δ | Latest expert CE Δ |
|---|---:|---:|---:|---:|---:|
| Small − Tiny, full V1 split | 11,340 | −0.3064 | −0.0933 | −0.0060 | −0.0335 |
| Full history − no history | 9,720 | −0.0088 | −0.0216 | −0.0328 | −0.0204 |
| GEGLU − GELU | 9,720 | −0.0379 | +0.0021 | +0.0116 | +0.0024 |
| Full dropout − dropout off | 9,990 | +0.0880 | −0.0115 | −0.0253 | −0.0082 |
| Multitask − pure BC | 19,980 | −0.0011 | −0.0021 | −0.0037 | −0.0141 |
| effect/contextCard roles − no roles | 5,400 | −0.0015 | −0.0107 | +0.0026 | −0.0040 |

The rows are **separate experiments at different development stages**; their CE levels and deltas must not be ranked against each other as if they shared one training set and architecture. The table is a compact index to the matched run pairs, not a factorial ablation of the final submission.

### 1. Data coverage mattered more than fitting a small replay subset

I began with Tiny (128 width, 1+1 layers) and Small (512 width, 3+3 layers) V1 policies, using 1%, 10%, 30%, or 100% replay-selection fractions. The percentages describe replay selection, not exactly the same percentage of individual actions. Across fractions, epoch counts and learning-rate schedules differed, so the terminal values are descriptive rather than a clean scaling-law experiment.

![Data scaling: four CE metrics for Tiny and Small](docs/figures/01_data_scaling.png)

The most revealing run is the Small model trained on 1% of replays. Its logged training CE reached **0.3985**, while latest-date CE rose to **2.1403** by the observed endpoint. With the full replay split, the Small model's terminal training CE was **0.6312** and latest-date CE **0.9156**. The smaller pool made the training objective easier to fit, but its holdout behavior became much worse. That directly corrected my early intuition that a lower training loss alone would always signal a better policy. The [1% run](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/f1ecq0x8) stopped at 810 of 2,800 planned steps; the [full-split run](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/izjmqifz) stopped at 11,610 of 13,300, so I do not compare them as equal-compute endpoints.

### 2. Increasing capacity helped the trained V1 comparison, with limits

On the full V1 split, Small had lower CE than Tiny at the last common four-metric evaluation, **step 11,340**: training −0.3064, in-distribution −0.0933, latest-date −0.0060, and latest-date expert −0.0335. Tiny and Small both used batch size 4,096. This supported testing a larger policy, while the small latest-date CE difference cautioned against treating capacity as an unconditional gain.

![Model scaling: Tiny, Small, and Base](docs/figures/02_model_scaling.png)

The Base run used a different batch size (2,048) and ended after fewer processed actions. Its endpoint is informative about that run, but it cannot isolate the effect of scaling width and depth. The final models have the later V2 feature layout and should not be presented as direct outcomes of these early V1 curves.

### 3. Recent action history carried useful decision context

At step 9,720, the **full three-action history** variant had lower CE than history-off: training −0.0088, in-distribution −0.0216, latest-date −0.0328, and latest-date expert −0.0204. Relative to the Basic history variant, the same four differences were −0.0046, −0.0082, −0.0119, and −0.0157. The comparison supports including recent decisions when scoring the current choice; it does not prove which individual historical field is responsible.

![Three-action history encodings](docs/figures/03_action_history.png)

### 4. Activation and dropout involved a training–validation trade-off

At the shared step 9,720, GEGLU lowered training CE by **0.0379** relative to GELU, but its in-distribution, latest-date, and latest-date expert CE were higher by **0.0021**, **0.0116**, and **0.0024**. I therefore cannot claim that GEGLU won on every validation metric. The ReLU run lasted only 270 optimizer steps and is not an equally trained third comparison.

In the separate GEGLU dropout comparison at step 9,990, full dropout raised training CE by **0.0880** but lowered in-distribution, latest-date, and latest-date expert CE by **0.0115**, **0.0253**, and **0.0082**. This was a clearer regularization signal. The final checkpoint configuration uses GEGLU with dropout; its selection reflects the broader development process and submitted setup, not a claim that the GEGLU-only CE ablation unanimously favored it.

![Activation and dropout CE comparisons](docs/figures/04_activation_dropout.png)

### 5. Auxiliary tasks did not establish a decisive advantage

I explored multitask supervision to encourage a richer state representation without collecting new raw games. The actual logged run used a CLS representation and auxiliary targets for the next decision, opponent deck, and final own prize count, with weights **0.08, 0.08, and 0.14**. This is narrower than the full list of ideas in my competition notes; for example, an opponent-strength task was discussed but is not a confirmed head in this matched run.

Compared with matched pure BC at step 19,980, the multitask variant's training, in-distribution, latest-date, and latest-date expert CE differences were **−0.0011, −0.0021, −0.0037, and −0.0141**. These are small single-run differences. The training curves suggest a modest early or middle-stage benefit whose margin narrows later; they do not support my original shorthand that the main-task training CE had clearly become worse at the end. Multitask learning was an informative experiment, **not a component of the two submitted checkpoints**.

![Multitask versus pure behavior cloning](docs/figures/05_multitask.png)

### 6. More card identity features were not a uniform improvement

Adding both effect and context-card role IDs plus static-card projections yielded a mixed result at step 5,400. Relative to the matched variant without them, training CE changed by **−0.0015**, in-distribution CE by **−0.0107**, latest-date CE by **+0.0026**, and latest-date expert CE by **−0.0040**. This is not a clean argument that the extra fields either help or hurt every distribution. Because several inputs changed together, the comparison also cannot assign the effect to either card role alone.

![Effect and context-card feature comparison](docs/figures/07_effect_context.png)

My notes raised a possible mechanism: extra card IDs could encourage memorization of card identity rather than transferable card relationships. That remains a **hypothesis**. The current four-CE comparison does not demonstrate it, and the stronger archetype-isolation observation discussed in the notes needs its own explicitly matched analysis before being used as a quantitative claim here.

---

## Final training and inference

After development on holdout runs, I trained the final 3+3- and 4+4-layer policies with the project's full-data training mode. In that mode all cached dates are eligible for training selection, and training-time validation is skipped; winner filtering, eligible loser augmentation, and replay-selection settings still apply. The submitted checkpoints use the same main model configuration apart from Transformer depth. I do not infer an unseen-set score from their full-data training logs.

The [submission notebook](submission.ipynb) loads both checkpoints, reconstructs each architecture from its saved config, and scores the same enumerated legal actions. It applies a softmax to each model's FP32 logits, **averages the two probability vectors equally**, applies its enabled damage-counter KO eligibility mask, and takes the highest-scoring remaining action. The agent retains a three-action history between decisions and supplies a fixed 60-card **Festival Lead** deck. The notebook packages `main.py`, the deck, model manifest, both models, and the competition `cg` package into `submission.tar.gz`. There is no search loop in the submitted agent.

My notes record an earlier offline observation that ensembling helped across several validation sets. I could not recover a like-for-like evaluation of **these exact two final checkpoints** against their individual models, so I report the implemented ensemble without assigning it a numerical gain. Likewise, the project does not provide an independently verified Elo or battle win-rate ablation for the architectural changes above.

---

## What I would improve

The biggest missing piece is a stable **game-level evaluation**. Offline action CE helped me organize experiments, but I did not establish how much an improvement on each validation subgroup translated into match outcomes. I would build an opponent pool with varied decks and policies, repeat paired matchups, and review critical failures by game phase and deck before drawing stronger conclusions.

The experiment record also shows where I was too quick to explain a metric change. Small-data training CE could improve while holdout CE deteriorated. GEGLU and the effect/context-card features moved different validation distributions in different directions. The specialized fine-tuning, oversampling, and expert-identity attempts were exploratory and not sufficiently complete or controlled for the quantitative figure set in this write-up. They remain research leads, not evidence for the final method.

My broader lesson was to keep three questions separate: **what the training objective rewards, what a replay holdout measures, and what wins games against the current opponent population**. Better data coverage and recent decision context had the clearest support in the recoverable offline curves. Future work should connect those curves to actual play.

---

## Reproducibility and sources

- [Competition page](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/) and [my competition notes](https://app.notion.com/p/39f888c4bb6180d1a714f6a930465490).
- [Experiment figures, CE tables, and W&B run links](docs/solution-experiment-results.md); [figure index](docs/figures/README.md); [W&B project](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc).
- [Replay extraction and cache construction](docs/data-pipeline.md), [training and validation](docs/training-validation.md), [model architecture](docs/model.md), and [submission workflow](docs/submission.md).
- The comparison tables use the last **common** four-metric optimizer step for each pair, not each run's best point. Training CE is the recorded EMA; validation CE uses raw logged values. There is one run per configuration in these comparisons, no uncertainty interval, and no claim that a small CE delta is statistically significant.

The local W&B export used for this analysis contains 98 run records. Full histories were recovered for 59 candidate runs (82,706 logged rows). The archived [comparison values](outputs/solution_analysis/reviewed_comparisons.json), [plotted points](outputs/solution_analysis/chart_data.json), and [plotting script](outputs/solution_analysis/analyze_and_plot.py) make the charts inspectable when those local analysis outputs are available.
