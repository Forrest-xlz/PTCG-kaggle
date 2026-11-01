# Experiment branches and W&B runs

The [W&B project](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc) retains 22 runs that support the [solution write-up](../kaggle_writeup.md) and final submission. Run names start with the matching experiment number. A branch is a code snapshot; a run's config and recorded Git commit identify its exact training setup.

## E001: Replay and model scale

Branch: `exp/E001-data-model-scaling`.

- Tiny, 128 width and 1+1 layers: [1%](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/s5nua4w6), [10%](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/yolhcb0c), [30%](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/ohpa87d1), [100%](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/34onyy62) of selected replays.
- Small, 512 width and 3+3 layers: [1%](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/f1ecq0x8), [10%](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/fn7ifd32), [30%](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/wj6ygkkx), [100%](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/izjmqifz).
- Base, 768 width and 6+6 layers: [100%](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/erqfsfu1).

## E002: Effect and context-card features

Branch: `exp/E002-effect-context-card`. Runs: [control](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/no12vhhe) and [features enabled](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/8kju2961).

## E003: Previous-action history token

Branch: `exp/E003-three-action-history`. Runs: [off](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/msrhj27e), [basic](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/thosp9qs), [structural](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/4x62kkbo), and [full](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/hcfee8qy).

## E004: Activation and dropout

Branch: `exp/E004-geglu-dropout`. Runs: [GELU](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/xg77g879), [GEGLU without dropout](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/dmae120r), and [GEGLU with full dropout](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/8gj1gsxn).

## E005–E006: Auxiliary tasks against pure BC

- `exp/E005-bc-holdout-control`: [pure BC control](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/dxinbrdj).
- `exp/E006-multitask-auxiliary`: [next-decision, opponent-deck, and prize auxiliary tasks](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/8o9qul5p).

## E007: Final full-data training

Branch: `exp/E007-full-data-training`. Runs: [3+3 layers](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/owgh6x7l) and [4+4 layers](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/13nt1b23).

These two W&B runs record commit `63b022c...`, which is absent from the retained branch heads. E007 documents the available full-data workflow; [submission.ipynb](../submission.ipynb) defines the final checkpoints and ensemble.
