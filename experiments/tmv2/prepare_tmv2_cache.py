from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build reusable Cat104 + fixed Trackman cache for TM-v2 experiments.')
    p.add_argument('--repo-root', default='.')
    p.add_argument('--train', required=True)
    p.add_argument('--trackman', required=True)
    p.add_argument('--mapping', required=True)
    p.add_argument('--work-dir', required=True)
    p.add_argument('--accepted-mapping-grades', nargs='+', default=['확정', '높음'])
    p.add_argument('--shrinkage', type=float, default=50.0)
    p.add_argument('--season-trend-min-history', type=int, default=4)
    p.add_argument('--trackman-shrinkage', type=float, default=100.0)
    p.add_argument('--context-smoothing', type=float, default=50.0)
    p.add_argument('--trackman-chunksize', type=int, default=250_000)
    p.add_argument('--rerun-trackman', action='store_true')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    from src.hgb_feature_selection_v2_pkg.common import _read_csv, build_main_features
    from src.hgb_feature_selection_v2_pkg.season_trend import add_season_trend_features
    from src.hgb_feature_selection_v2_pkg.trackman import (
        build_trackman_features,
        load_mapping,
        load_or_aggregate_trackman,
        merge_trackman,
    )

    work = Path(args.work_dir).resolve()
    cache_dir = work / 'native_handfix_run'
    cache_dir.mkdir(parents=True, exist_ok=True)
    trackman_cache = work / 'trackman_cache'
    trackman_cache.mkdir(parents=True, exist_ok=True)

    train = _read_csv(Path(args.train))
    main = build_main_features(train, shrinkage=args.shrinkage)
    main = add_season_trend_features(
        main,
        train,
        shrinkage=args.shrinkage,
        min_history_seasons=args.season_trend_min_history,
    )

    # IMPORTANT: train uses batter_hand codes 1/2 while Trackman context uses Left/Right.
    # Normalize the join key before build_trackman_features; otherwise contextual TM counts
    # are effectively unmatched. This fix improved Brier in 2023 and again in 2024.
    batter_hand = main.features.attrs.get('batter_hand')
    if batter_hand is None:
        raise RuntimeError('build_main_features did not preserve batter_hand join key')
    main.features.attrs['batter_hand'] = (
        batter_hand.astype('string')
        .str.replace(r'\.0$', '', regex=True)
        .replace({'1': 'Left', '2': 'Right', 'L': 'Left', 'R': 'Right'})
        .fillna('__MISSING__')
    )

    mapping = load_mapping(Path(args.mapping), tuple(args.accepted_mapping_grades))
    group_stats, context_stats, drift_stats = load_or_aggregate_trackman(
        Path(args.trackman),
        trackman_cache,
        max_rows=None,
        chunksize=args.trackman_chunksize,
        rerun=args.rerun_trackman,
    )
    tm_features, tm_blocks, tm_categorical = build_trackman_features(
        main,
        mapping,
        group_stats,
        context_stats,
        drift_stats,
        shrinkage_pitches=args.trackman_shrinkage,
        context_smoothing=args.context_smoothing,
    )
    engineered = merge_trackman(main, tm_features, tm_blocks, tm_categorical)
    if 'asof_pitcher_pitchmix_n' in engineered.features.columns:
        engineered.features.drop(columns=['asof_pitcher_pitchmix_n'], inplace=True)

    recent = set(engineered.blocks.get('recent_form', []))
    features_104 = [c for c in engineered.features.columns if c not in recent]
    trackman_features = {
        feature
        for block, columns in engineered.blocks.items()
        if block.startswith('trackman_')
        for feature in columns
    }
    tm15 = [f for f in features_104 if f in trackman_features]
    base89 = [f for f in features_104 if f not in trackman_features]

    if len(features_104) != 104:
        raise RuntimeError(f'Expected 104 baseline features, got {len(features_104)}')
    if len(base89) != 89 or len(tm15) != 15:
        raise RuntimeError(f'Expected 89 non-TM + 15 TM features, got {len(base89)} + {len(tm15)}')

    engineered.features.loc[:, base89].to_pickle(cache_dir / 'base89.pkl')
    engineered.features.loc[:, tm15].to_pickle(cache_dir / 'tm_fix.pkl')
    pd.DataFrame(
        {
            'row_id': engineered.row_ids.to_numpy(),
            'season': engineered.seasons.to_numpy(),
            'control_success': engineered.target.to_numpy(),
        }
    ).to_pickle(cache_dir / 'meta.pkl')

    (cache_dir / 'features_base89.txt').write_text('\n'.join(base89) + '\n', encoding='utf-8')
    (cache_dir / 'features_tm15.txt').write_text('\n'.join(tm15) + '\n', encoding='utf-8')
    print(f'SAVED {cache_dir}')
    print(f'base89={len(base89)}, tm15={len(tm15)}, total={len(features_104)}')

    del engineered, train, main, group_stats, context_stats, drift_stats
    gc.collect()


if __name__ == '__main__':
    main()
