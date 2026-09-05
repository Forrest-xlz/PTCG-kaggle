"""Exact-deck coverage of the selected training pool (not epoch exposure)."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from analysis.deck_statistics import (
    CardCatalog, changed_slots, classify_deck, exact_deck_identity,
    load_deck_rows, weighted_jaccard,
)
from training.feature_cache import MmapFeatureDataset, parse_source_date, stable_deck_key, stable_episode_key
from training.expert_replays import load_expert_date_info, load_expert_loser_date_info
from validation.deck_names import parse_top_decks
from validation.isolation import load_isolation_replay_sets


COUNT_COLUMNS = ['raw_replays', 'expert_raw_replays', 'train_replays',
                 'expert_train_replays', 'train_action_samples', 'expert_train_action_samples']


def build_coverage_tables(rows, actions, experts, top_decks, names, catalog):
    """Count unique (date, episode) per exact deck; mirrors count once.

    Expert means an episode meeting the training expert-validation rule;
    it does not prove that the acting player was the high-scoring player.
    actions has one or more rows per date/episode/deck, with action_samples.
    """
    raw = rows[['date', 'episode_id', 'deck']].copy()
    raw['cards'] = raw.deck.map(lambda value: tuple(sorted(value)))
    cards = {stable_deck_key(deck): deck for deck in raw.cards.unique()}
    cards.update({stable_deck_key(deck): tuple(sorted(deck)) for deck in top_decks})
    card_keys = {deck: key for key, deck in cards.items()}
    raw['deck_key'] = raw.cards.map(card_keys)
    raw['date'] = raw.date.map(lambda date: parse_source_date(str(date)))
    raw['episode_key'] = raw.episode_id.map(stable_episode_key)
    raw = raw.drop_duplicates(['date', 'episode_key', 'deck_key'])
    actions = actions.copy()
    for frame in (raw, actions):
        frame['expert'] = pd.Series([int(key) in experts.get(date, ())
                           for date, key in zip(frame.date, frame.episode_key)], index=frame.index, dtype=bool)
    train_replays = actions.drop_duplicates(['date', 'episode_key', 'deck_key'])
    counts = pd.DataFrame(index=pd.Index(list(cards), name='deck_key', dtype='uint64'))
    for column, frame in [('raw_replays', raw), ('expert_raw_replays', raw[raw.expert]),
                          ('train_replays', train_replays),
                          ('expert_train_replays', train_replays[train_replays.expert])]:
        counts[column] = frame.groupby('deck_key').size()
    counts['train_action_samples'] = actions.groupby('deck_key').action_samples.sum()
    counts['expert_train_action_samples'] = actions[actions.expert].groupby('deck_key').action_samples.sum()
    counts = counts.fillna(0).astype('int64')
    unknown = set(actions.deck_key) - set(cards)
    if unknown:
        raise ValueError(f'{len(unknown)} training decks missing from deck CSVs; regenerate deck extraction for cache dates')
    metadata = {key: dict(deck_id=exact_deck_identity(deck).deck_id,
                          archetype=classify_deck(deck, catalog).archetype,
                          card_ids=json.dumps(list(deck))) for key, deck in cards.items()}
    summary, variants = [], []
    for name, deck in zip(names, top_decks):
        key = stable_deck_key(deck)
        summary.append(dict(name=name, **metadata[key], **counts.loc[key].to_dict()))
        for other, other_cards in cards.items():
            if metadata[other]['archetype'] == metadata[key]['archetype']:
                variants.append(dict(name=name, **metadata[other],
                                     changed_slots=changed_slots(deck, other_cards),
                                     weighted_jaccard=weighted_jaccard(deck, other_cards),
                                     **counts.loc[other].to_dict()))
    # Also preserve the useful full exact-deck census, without all-pairs similarity.
    census = counts.join(pd.DataFrame.from_dict(metadata, orient='index')).reset_index(drop=True)
    return pd.DataFrame(summary), pd.DataFrame(variants).sort_values(['name', 'changed_slots', 'raw_replays'], ascending=[True, True, False]), census.sort_values('raw_replays', ascending=False)


def selected_action_counts(dataset, indices):
    """Aggregate cache metadata shard-by-shard; no model inference or feature loading."""
    parts = []
    indices = np.sort(np.asarray(indices))
    for start, end, date, shard in zip(dataset.starts, dataset.ends, dataset.shard_dates, dataset.shards):
        local = indices[np.searchsorted(indices, start):np.searchsorted(indices, end)] - start
        for offset in range(0, len(local), 250_000):
            selected = local[offset:offset + 250_000]
            part = pd.DataFrame({column: shard.arrays[column][selected]
                                 for column in ('episode_key', 'deck_key')})
            part = part.groupby(['episode_key', 'deck_key']).size().rename('action_samples').reset_index()
            part['date'] = [date] * len(part)
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=['date', 'episode_key', 'deck_key', 'action_samples'])
    return pd.concat(parts, ignore_index=True).groupby(['date', 'episode_key', 'deck_key'], as_index=False).action_samples.sum()


def run_coverage(project_root, train_config):
    """Read the train YAML and reuse the real holdout/full-data selection methods."""
    from training.train import select_loser_augmentation_dates

    root = Path(project_root)
    settings = yaml.safe_load(Path(train_config).read_text(encoding='utf-8'))
    cfg = settings['train']
    def path(value):
        return (root / str(value).replace('${version_name}', str(settings['version_name']))).resolve()
    decks, names = parse_top_decks(cfg['top_decks'])
    if not decks:
        raise ValueError('train.top_decks must contain at least one deck')
    mode = cfg['data_selection_mode']
    if mode not in ('holdout', 'full_data'):
        raise ValueError('data_selection_mode must be holdout or full_data')
    isolation = cfg['isolation_validation']
    print('Loading deck CSVs...', flush=True)
    rows = load_deck_rows(sorted(path(isolation['deck_data']).glob('*.decks.csv')))
    dates = {parse_source_date(str(date)) for date in rows.date.unique()}
    with_dataset = MmapFeatureDataset(path(cfg['data']))
    try:
        missing = set(with_dataset.shard_dates) - dates
        if missing:
            raise ValueError(f'Deck CSVs missing cache dates: {sorted(missing)}')
        info = load_expert_date_info(path(cfg['replay_episodes']), dates, cfg['expert_validation_ratio'])
        experts = {date: item.expert_episode_keys for date, item in info.items()}
        loser_cfg = cfg['loser_augmentation']
        loser_keys = None
        if loser_cfg['enabled']:
            loser_dates = select_loser_augmentation_dates(with_dataset.shard_dates, loser_cfg['recent_dates'], mode)
            loser_keys = {date: item.eligible_episode_keys for date, item in load_expert_loser_date_info(
                path(cfg['replay_episodes']), loser_dates, loser_cfg['expert_ratio']).items()}
        kwargs = dict(train_replay_ratio=cfg['train_replay_ratio'], train_replay_seed=cfg['train_replay_seed'], loser_episode_keys=loser_keys)
        if mode == 'holdout':
            selections = {f'val_{name}': path(value) for name, value in isolation['selections'].items() if value is not None}
            isolated = load_isolation_replay_sets(deck_data_dir=path(isolation['deck_data']), selection_paths=selections, required_dates=with_dataset.shard_dates) if selections else None
            selection = with_dataset.build_splits(validation_ratio=cfg['validation_ratio'], validation_seed=cfg['validation_seed'],
                isolation_episode_keys=isolated.by_namespace if isolated else None, **kwargs)
        else:
            selection = with_dataset.build_training_indices(**kwargs)
        print(f'Counting {len(selection.train):,} selected training actions...', flush=True)
        actions = selected_action_counts(with_dataset, selection.train)
        cache_dates = sorted(set(with_dataset.shard_dates))
    finally:
        with_dataset.close()
    catalog = CardCatalog.from_csv(root.parent / 'pokemon_tcg_ai_battle/EN_Card_Data.csv')
    summary, variants, census = build_coverage_tables(rows, actions, experts, decks, names, catalog)
    context = dict(train_config=str(Path(train_config).resolve()), data_selection_mode=mode,
                   raw_dates=sorted(dates), cache_dates=cache_dates,
                   train_action_samples=int(actions.action_samples.sum()), max_samples_per_epoch=cfg.get('max_samples'),
                   expert_validation_ratio=cfg['expert_validation_ratio'],
                   note='Training pool, not epoch exposure. Expert refers to replay max participant score at daily participant percentile cutoff. Raw dates may exceed cache dates.',
                   train_selection_config={key: cfg[key] for key in ('validation_ratio', 'validation_seed', 'train_replay_ratio', 'train_replay_seed', 'loser_augmentation', 'isolation_validation', 'top_decks')})
    return summary, variants, census, context


def plot_coverage(summary, variants, output_dir):
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    summary.set_index('name')[['raw_replays', 'train_replays', 'expert_train_replays']].plot.bar(ax=ax)
    ax.set(ylabel='Unique replays', title='Exact top-deck coverage', xlabel='Top deck')
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / 'top_deck_coverage.png', dpi=160)
    figures = [fig]
    fig, axes = plt.subplots(len(summary), 1, figsize=(11, 3.5 * len(summary)), squeeze=False)
    for ax, name in zip(axes[:, 0], summary.name):
        data = variants[variants.name == name]
        for column, marker in [('raw_replays', 'o'), ('train_replays', 'x'), ('expert_train_replays', '+')]:
            ax.scatter(data.changed_slots, data[column], label=column, marker=marker, alpha=.65)
        ax.set(title=f'{name}: same-archetype variants', xlabel='Changed card slots from top deck', ylabel='Unique replays')
        ax.set_yscale('symlog', linthresh=1)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / 'top_deck_archetype_coverage.png', dpi=160)
    figures.append(fig)
    return figures
