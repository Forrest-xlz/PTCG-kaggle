# W&B run curation

Project: [ptcg_bc](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc).
This inventory was made from the 98-run export; live W&B was checked at the same count before renaming.
The original metadata export is in `outputs/wandb_export/20261031_052357_571600/runs.jsonl`.
Complete histories for all 98 runs are backed up locally in `outputs/solution_analysis/history/`
and indexed by `outputs/solution_analysis/all_history_manifest.json`.

Keep 22 runs: the six plotted comparisons and the two final training runs.
Cleanup scope: 76 runs; 1 deleted, 75 still pending.
The remaining remote deletions require explicit approval of this exact list.
A `crashed` state alone was not used to decide whether to keep a run.
Run URLs use permanent IDs, so renaming does not break the write-up links.

## Keep: renamed key runs

| Experiment | Run ID | New W&B name | Previous name |
|---|---|---|---|
| E001 | [s5nua4w6](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/s5nua4w6) | `E001-data-scale-tiny-1pct` | `model_v1_data_tiny_model_tiny_pretrained` |
| E001 | [yolhcb0c](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/yolhcb0c) | `E001-data-scale-tiny-10pct` | `model_v1_data_small_model_tiny_pretrained` |
| E001 | [ohpa87d1](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/ohpa87d1) | `E001-data-scale-tiny-30pct` | `model_v1_data_base_model_tiny_pretrained` |
| E001 | [34onyy62](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/34onyy62) | `E001-data-scale-tiny-100pct` | `model_v1_data_full_model_tiny_pretrained` |
| E001 | [f1ecq0x8](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/f1ecq0x8) | `E001-data-scale-small-1pct` | `model_v1_data_tiny_model_small_pretrained` |
| E001 | [fn7ifd32](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/fn7ifd32) | `E001-data-scale-small-10pct` | `model_v1_data_samall_model_small_pretrained` |
| E001 | [wj6ygkkx](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/wj6ygkkx) | `E001-data-scale-small-30pct` | `model_v1_data_base_model_small_pretrained` |
| E001 | [izjmqifz](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/izjmqifz) | `E001-data-scale-small-100pct` | `model_v1_data_full_model_small_pretrained` |
| E001 | [erqfsfu1](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/erqfsfu1) | `E001-model-scale-base-100pct` | `model_v1_data_full_model_base_pretrained` |
| E002 | [no12vhhe](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/no12vhhe) | `E002-effect-context-control` | `ver_1.3.6_residual_false_mlp2` |
| E002 | [8kju2961](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/8kju2961) | `E002-effect-context-enabled` | `ver_1.4.0` |
| E003 | [msrhj27e](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/msrhj27e) | `E003-action-history-off` | `ver_1.4.2_mlp32` |
| E003 | [thosp9qs](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/thosp9qs) | `E003-action-history-basic` | `ver_1.4.3_basic_12` |
| E003 | [4x62kkbo](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/4x62kkbo) | `E003-action-history-structural` | `ver_1.4.3_structural_12` |
| E003 | [hcfee8qy](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/hcfee8qy) | `E003-action-history-full` | `ver_1.4.3_full_12` |
| E004 | [xg77g879](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/xg77g879) | `E004-activation-gelu` | `ver_1.4.5_gelu` |
| E004 | [dmae120r](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/dmae120r) | `E004-activation-geglu-no-dropout` | `ver_1.4.5_geglu` |
| E004 | [8gj1gsxn](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/8gj1gsxn) | `E004-geglu-full-dropout` | `ver_1.4.5_geglu_dropout_full` |
| E005 | [dxinbrdj](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/dxinbrdj) | `E005-pure-bc-multitask-control` | `ver_1.6.5_d3_33_baseline` |
| E006 | [8o9qul5p](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/8o9qul5p) | `E006-multitask-auxiliary` | `ver_1.13.0` |
| E007 | [owgh6x7l](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/owgh6x7l) | `E007-final-full-data-3x3` | `ver_final_0_d2_33` |
| E007 | [13nt1b23](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/13nt1b23) | `E007-final-full-data-4x4` | `ver_final_0_d2_44` |

## Proposed deletions

The table is the exact deletion scope for review. `Pending` rows remain on W&B.
`Latest CE` is the last recorded latest-date CE where present; it is provided to identify
a run, not as a quality threshold.

| Run ID | Current or former name | Status | State | Steps | Latest CE | Reason |
|---|---|---|---:|---:|---:|---|
| [c7rt74i8](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/c7rt74i8) | `ver_` | Deleted | finished | 30 | — | 30 steps; no validation CE |
| [fv7185bx](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/fv7185bx) | `ver_final_0_d1_33` | Pending | crashed | 22,220 | — | superseded full-data training run |
| [v2saapwi](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/v2saapwi) | `ver_final_0_d1_44` | Pending | crashed | 22,040 | — | superseded full-data training run |
| [onv84ct6](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/onv84ct6) | `ver_1.16.0` | Pending | crashed | 5,720 | 0.7356 | exploratory variant outside the retained comparisons |
| [rm0nyfg8](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/rm0nyfg8) | `value_v1` | Pending | crashed | 3,240 | — | incomplete run; no latest-date validation CE |
| [ccolfre1](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/ccolfre1) | `expert_classifier_1` | Pending | crashed | 1,460 | — | incomplete run; no latest-date validation CE |
| [1d5895of](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/1d5895of) | `ver_1.15.0` | Pending | crashed | 3,160 | 0.7515 | exploratory variant outside the retained comparisons |
| [a7dtzcqf](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/a7dtzcqf) | `ver_1.6.5_d3_33_15_0.15` | Pending | crashed | 19,060 | 0.7070 | exploratory variant outside the retained comparisons |
| [tos45t7i](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/tos45t7i) | `ver_final_0_33` | Pending | finished | 22,690 | — | superseded full-data training run |
| [6rwnymph](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/6rwnymph) | `ver_1.9.1` | Pending | finished | 15,890 | 0.6996 | exploratory variant outside the retained comparisons |
| [mi6k4dux](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/mi6k4dux) | `ver_1.6.4` | Pending | finished | 16,810 | 0.6909 | exploratory variant outside the retained comparisons |
| [fa1bsnvd](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/fa1bsnvd) | `ver_1.9.0` | Pending | crashed | 15,660 | 0.6986 | exploratory variant outside the retained comparisons |
| [vd8mlojl](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/vd8mlojl) | `ver_1.6.3_add_6.20_6.26` | Pending | finished | 20,560 | 0.6943 | exploratory variant outside the retained comparisons |
| [xeevh5ge](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/xeevh5ge) | `ver_1.11.0` | Pending | crashed | 550 | 0.8425 | exploratory variant outside the retained comparisons |
| [sfdmrkye](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/sfdmrkye) | `ver_1.6.3_lr_1.5e-4` | Pending | finished | 18,625 | 0.6860 | exploratory variant outside the retained comparisons |
| [4kafwse0](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/4kafwse0) | `ver_1.6.3_lr_6e-4` | Pending | crashed | 1,490 | 0.7901 | exploratory variant outside the retained comparisons |
| [8g316lpz](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/8g316lpz) | `ver_1.6.2_53_0.6` | Pending | crashed | 2,000 | 0.7540 | exploratory variant outside the retained comparisons |
| [g6aq5i0r](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/g6aq5i0r) | `ver_1.9.0` | Pending | crashed | 4,320 | 0.7293 | exploratory variant outside the retained comparisons |
| [z9cygoth](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/z9cygoth) | `finetune_deck1_indistribution_0.01` | Pending | finished | 595 | 0.9333 | exploratory variant outside the retained comparisons |
| [91kkpnoj](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/91kkpnoj) | `finetune_deck1` | Pending | crashed | 520 | 0.7209 | exploratory variant outside the retained comparisons |
| [ecywffe5](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/ecywffe5) | `ver_1.7.0_33_only_deck_expert_weights_0` | Pending | crashed | 12,360 | 0.6957 | exploratory variant outside the retained comparisons |
| [jm674nwr](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/jm674nwr) | `ver_1.7.0_33_gpt_baseline` | Pending | finished | 16,650 | 0.8029 | exploratory variant outside the retained comparisons |
| [jm6s3iho](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/jm6s3iho) | `ver_1.6.2_44_0.6` | Pending | finished | 18,625 | 0.7059 | exploratory variant outside the retained comparisons |
| [e64atih3](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/e64atih3) | `ver_1.6.2_44` | Pending | finished | 18,875 | 0.6976 | exploratory variant outside the retained comparisons |
| [javodeq0](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/javodeq0) | `ver_1.6.1_1.0` | Pending | crashed | 18,200 | 0.6908 | exploratory variant outside the retained comparisons |
| [aw3zs5gl](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/aw3zs5gl) | `ver_1.6.1_4_4` | Pending | crashed | 4,220 | 0.7312 | exploratory variant outside the retained comparisons |
| [xu3q8tz9](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/xu3q8tz9) | `ver_1.6.0_resume` | Pending | finished | 18,525 | 0.6924 | exploratory variant outside the retained comparisons |
| [kjg7umq8](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/kjg7umq8) | `ver_1.6.0` | Pending | crashed | 10,110 | 0.6904 | exploratory variant outside the retained comparisons |
| [d6rku2vf](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/d6rku2vf) | `ver_1.4.5_geglu_dropout_without_softmax` | Pending | finished | 12,160 | 0.6955 | exploratory variant outside the retained comparisons |
| [jjgf3gtf](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/jjgf3gtf) | `ver_1.4.5_geglu_dropout_all` | Pending | finished | 4,440 | 0.6942 | exploratory variant outside the retained comparisons |
| [23jk0uwa](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/23jk0uwa) | `ver_1.4.5` | Pending | finished | 270 | — | 270 steps; no validation CE |
| [80kxcquv](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/80kxcquv) | `ver_1.4.4_only_dynamic` | Pending | crashed | 2,680 | 0.6985 | exploratory variant outside the retained comparisons |
| [sw2e7jpo](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/sw2e7jpo) | `ver_1.4.4_only_pre_evol` | Pending | finished | 1,020 | 0.7776 | exploratory variant outside the retained comparisons |
| [c5fky1mi](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/c5fky1mi) | `ver_1.4.3_full_22` | Pending | crashed | 4,320 | 0.6930 | exploratory variant outside the retained comparisons |
| [46y842oz](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/46y842oz) | `ver_1.4.3_basic_13` | Pending | finished | 810 | 0.7883 | exploratory variant outside the retained comparisons |
| [7w0mlaja](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/7w0mlaja) | `ver_1.4.3_basic_22` | Pending | crashed | 890 | 0.7694 | exploratory variant outside the retained comparisons |
| [7s303sv6](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/7s303sv6) | `ver_1.4.2_only_position_and_index_ablation` | Pending | finished | 12,610 | 0.7396 | exploratory variant outside the retained comparisons |
| [24yjij7o](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/24yjij7o) | `ver_1.4.2_only_tool_energy_index_ablation` | Pending | finished | 210 | — | 210 steps; no validation CE |
| [fyyinx2t](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/fyyinx2t) | `ver_1.4.2_only_inplay_index_ablation` | Pending | finished | 270 | — | 270 steps; no validation CE |
| [7e4n6axe](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/7e4n6axe) | `ver_1.4.2_only_index_ablation` | Pending | finished | 270 | — | 270 steps; no validation CE |
| [tg3qoip3](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/tg3qoip3) | `ver_1.4.2_only_relative_position_ablation` | Pending | finished | 330 | 0.8509 | exploratory variant outside the retained comparisons |
| [0tkbxll6](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/0tkbxll6) | `ver_1.4.2_all_fasle_ablation` | Pending | finished | 190 | — | 190 steps; no validation CE |
| [mwhjy2ck](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/mwhjy2ck) | `ver_1.4.2_tool_energy_index_ablation` | Pending | finished | 270 | — | 270 steps; no validation CE |
| [9p2zz8ap](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/9p2zz8ap) | `ver_1.4.2_inplay_index_ablation` | Pending | finished | 310 | 0.8506 | exploratory variant outside the retained comparisons |
| [hp7euxcs](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/hp7euxcs) | `ver_1.4.2_index_ablation` | Pending | finished | 640 | 0.8004 | exploratory variant outside the retained comparisons |
| [549y93b8](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/549y93b8) | `ver_1.4.2_relative_position_ablation` | Pending | finished | 900 | 0.8013 | exploratory variant outside the retained comparisons |
| [3cwebo9b](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/3cwebo9b) | `ver_1.4.2_mlp31` | Pending | crashed | 7,110 | 0.6979 | exploratory variant outside the retained comparisons |
| [3jmz4ldw](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/3jmz4ldw) | `ver_1.4.2_mlp22` | Pending | finished | 850 | 0.7894 | exploratory variant outside the retained comparisons |
| [hu7k9nur](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/hu7k9nur) | `ver_1.4.1_mlp0` | Pending | finished | 450 | 0.8812 | exploratory variant outside the retained comparisons |
| [rgpmb6d0](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/rgpmb6d0) | `ver_1.4.1_mlp1` | Pending | finished | 270 | — | 270 steps; no validation CE |
| [asszmhwt](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/asszmhwt) | `ver_1.4.1_mlp2` | Pending | finished | 330 | 0.8902 | exploratory variant outside the retained comparisons |
| [atbfpt2y](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/atbfpt2y) | `ver_1.3.7_encoder2_option0` | Pending | crashed | 270 | — | 270 steps; no validation CE |
| [n2bht5j9](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/n2bht5j9) | `ver_1.3.7_encoder1_option0` | Pending | finished | 140 | — | 140 steps; no validation CE |
| [8yprl0rs](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/8yprl0rs) | `ver_1.3.6_residual_false_mlp2_option_mlp1` | Pending | crashed | 3,780 | 0.7283 | exploratory variant outside the retained comparisons |
| [526cghc4](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/526cghc4) | `ver_1.3.6_residual_false_mlp2_option` | Pending | finished | 1,140 | 0.7911 | exploratory variant outside the retained comparisons |
| [kit08kse](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/kit08kse) | `ver_1.3.6_residual_true_mlp4` | Pending | finished | 1,080 | 0.7916 | exploratory variant outside the retained comparisons |
| [3ivpqaoz](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/3ivpqaoz) | `ver_1.3.6_residual_false_mlp1` | Pending | finished | 1,370 | 0.7798 | exploratory variant outside the retained comparisons |
| [c0fcm9eu](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/c0fcm9eu) | `ver_1.3.6_residual_false_mlp3` | Pending | finished | 3,240 | 0.7524 | exploratory variant outside the retained comparisons |
| [kp780yny](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/kp780yny) | `ver_1.3.5_313_42` | Pending | crashed | 5,780 | 0.7554 | exploratory variant outside the retained comparisons |
| [x66lrjpm](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/x66lrjpm) | `ver_1.3.5_313_region_card_mlp` | Pending | finished | 6,480 | 0.7552 | exploratory variant outside the retained comparisons |
| [2fu5x5pa](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/2fu5x5pa) | `ver_1.3.5_313` | Pending | finished | 8,810 | 0.7490 | exploratory variant outside the retained comparisons |
| [butr1use](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/butr1use) | `ver_1.3.5_222` | Pending | finished | 10,260 | 0.7608 | exploratory variant outside the retained comparisons |
| [vv5mogjq](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/vv5mogjq) | `ver_1.3.5_no_54card` | Pending | finished | 8,810 | 0.7580 | exploratory variant outside the retained comparisons |
| [9t5til3g](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/9t5til3g) | `ver_1.3.4.1` | Pending | finished | 11,070 | 0.8058 | exploratory variant outside the retained comparisons |
| [5drw0va0](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/5drw0va0) | `ver_1.3.4` | Pending | finished | 1,950 | 0.7911 | exploratory variant outside the retained comparisons |
| [ru60136x](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/ru60136x) | `ver_1.3.3.2` | Pending | finished | 2,970 | 0.7699 | exploratory variant outside the retained comparisons |
| [05mv925p](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/05mv925p) | `ver_1.3.3.1` | Pending | finished | 3,310 | 0.7650 | exploratory variant outside the retained comparisons |
| [rp2jof8y](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/rp2jof8y) | `ver_1.3.3` | Pending | finished | 5,670 | 0.7539 | exploratory variant outside the retained comparisons |
| [2dnerc61](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/2dnerc61) | `ver_1.3.2.1` | Pending | crashed | 8,290 | 0.7509 | exploratory variant outside the retained comparisons |
| [etsmxplx](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/etsmxplx) | `ver_1.3.2` | Pending | crashed | 3,510 | 0.7673 | exploratory variant outside the retained comparisons |
| [1kqgfx6l](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/1kqgfx6l) | `ver1.3.1` | Pending | finished | 4,150 | 0.7804 | exploratory variant outside the retained comparisons |
| [5fq80uhn](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/5fq80uhn) | `ver_1.3.0` | Pending | crashed | 11,880 | 0.7911 | exploratory variant outside the retained comparisons |
| [0tgypsin](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/0tgypsin) | `ver_1.2.0` | Pending | crashed | 12,150 | 0.9112 | exploratory variant outside the retained comparisons |
| [mfpqtwwx](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/mfpqtwwx) | `ver_1.1_card_feature_ratio_0.7` | Pending | finished | 13,300 | 0.9260 | exploratory variant outside the retained comparisons |
| [8o0t4jxw](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/8o0t4jxw) | `ver_1.1_card_feature_ratio_0.5` | Pending | finished | 2,330 | 0.8835 | exploratory variant outside the retained comparisons |
| [yi0b5wu6](https://forge.coreweave.com/wandb/xlz1014361885-tokyo/ptcg_bc/runs/yi0b5wu6) | `ver_1.1_card_feature_ratio_1.0` | Pending | finished | 9,720 | 0.8811 | exploratory variant outside the retained comparisons |
