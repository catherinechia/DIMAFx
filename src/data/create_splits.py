import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def load_df(path):
    """ Load a csv file, or return None if no path is given. """
    if path is None:
        return None
    return pd.read_csv(path)


def get_strata(df, label_cols):
    """ Combine one or more label columns into a single stratification label per sample. """
    return df[label_cols].astype(str).agg('_'.join, axis=1).values


def check_df(df, name, group_col, label_cols):
    """ Check that the required columns exist and that labels are consistent within each group. """
    missing = [c for c in [group_col] + label_cols if c not in df.columns]
    assert len(missing) == 0, f'Columns {missing} not found in {name}.'
    assert not df[[group_col] + label_cols].isna().any().any(), f'Missing values in the group/label columns of {name}.'
    #
    # Stratification works per sample; groups with mixed labels are allowed but make the class balance less exact
    num_unique = df.groupby(group_col)[label_cols].nunique().max(axis=1)
    if (num_unique > 1).any():
        warnings.warn(f'{(num_unique > 1).sum()} groups in {name} have more than one label value.')


def k_fold_indices(df, group_col, label_cols, k, seed):
    """ Stratified group k-fold. Returns a list of (train_idx, held_out_idx) per fold. """
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(sgkf.split(df, y=get_strata(df, label_cols), groups=df[group_col].values))


def holdout_split(df, group_col, label_cols, test_size, seed):
    """ Stratified group train/test split. The test size is approximate, as it is taken as one fold of round(1 / test_size) folds. """
    n_splits = round(1 / test_size)
    assert n_splits >= 2, f'test_size {test_size} is too large, it must be <= 0.5.'
    train_idx, test_idx = k_fold_indices(df, group_col, label_cols, n_splits, seed)[0]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def save_fold(out_dir, fold, dfs, suffixes):
    """ Save the dfs of a fold as {mode}.csv / {mode}_{suffix}.csv, matching MMDatasetBase.get_data_file_name. """
    fold_dir = os.path.join(out_dir, f'{fold}')
    os.makedirs(fold_dir, exist_ok=True)
    for mode, df in dfs.items():
        for suffix in suffixes:
            file_name = f'{mode}.csv' if suffix == 'none' else f'{mode}_{suffix}.csv'
            df.to_csv(os.path.join(fold_dir, file_name), index=False)


def summarize(fold, dfs, group_col, label_cols):
    """ Number of samples, groups and per-label counts for each set of a fold. """
    rows = []
    for mode, df in dfs.items():
        row = {'fold': fold, 'set': mode, 'n_samples': len(df), 'n_groups': df[group_col].nunique()}
        for label, count in pd.Series(get_strata(df, label_cols)).value_counts().sort_index().items():
            row[f'label_{label}'] = count
        rows.append(row)
    return rows


def create_splits(args):
    """ Create k-fold splits from df_train (and df_test). """
    df_train = load_df(args.df_train)
    df_test = load_df(args.df_test)
    #
    check_df(df_train, 'df_train', args.group_col, args.label_cols)
    if args.split_mode == 'holdout':
        # First split df_train into train and test, then continue as fixed_test
        assert df_test is None, "split_mode 'holdout' creates the test set from df_train, do not pass --df_test."
        df_train, df_test = holdout_split(df_train, args.group_col, args.label_cols, args.test_size, args.seed)
        print(f'Holdout: {len(df_train)} train samples, {len(df_test)} test samples ({len(df_test) / (len(df_train) + len(df_test)):.1%})')
    if df_test is not None:
        check_df(df_test, 'df_test', args.group_col, args.label_cols)
        overlap = set(df_train[args.group_col]) & set(df_test[args.group_col])
        if args.split_mode in ['fixed_test', 'holdout']:
            # A fixed test set must be fully independent of the training data
            assert len(overlap) == 0, f'{len(overlap)} groups occur in both df_train and df_test, e.g. {sorted(overlap)[:5]}.'
        elif len(overlap) > 0:
            print(f'Note: {len(overlap)} groups occur in both df_train and df_test; they are kept together when pooled.')
    #
    if args.split_mode == 'pooled':
        # Pool df_train and df_test, and use each fold once as the test set
        df_all = pd.concat([df for df in [df_train, df_test] if df is not None], ignore_index=True)
        df_split = df_all
    else:
        # Split only df_train; the held-out fold is saved as val, df_test is the test set of every fold
        assert df_test is not None, f"split_mode '{args.split_mode}' requires --df_test."
        df_split = df_train
    #
    # A: train/test, B: train/val/test. Default: A for pooled, B for fixed_test/holdout
    output_sets = getattr(args, 'output_sets', None)
    if output_sets is None:
        output_sets = 'train_test' if args.split_mode == 'pooled' else 'train_val_test'
    if args.split_mode == 'pooled' and output_sets == 'train_val_test':
        assert args.k >= 3, 'pooled with train_val_test requires k >= 3.'
    #
    folds = k_fold_indices(df_split, args.group_col, args.label_cols, args.k, args.seed)
    held_out_folds = [held_out_idx for _, held_out_idx in folds]
    #
    summary = []
    for fold, (train_idx, held_out_idx) in enumerate(folds):
        if args.split_mode == 'pooled':
            # The held-out fold is the test set
            dfs = {'test': df_split.iloc[held_out_idx]}
            if output_sets == 'train_val_test':
                # The next fold is the val set, the remaining k-2 folds are the train set
                val_idx = held_out_folds[(fold + 1) % args.k]
                train_idx = np.setdiff1d(train_idx, val_idx)
                dfs['val'] = df_split.iloc[val_idx]
            dfs['train'] = df_split.iloc[train_idx]
        else:
            # The held-out fold is the val set (only saved for train_val_test), df_test is the test set
            dfs = {'train': df_split.iloc[train_idx], 'test': df_test}
            if output_sets == 'train_val_test':
                dfs['val'] = df_split.iloc[held_out_idx]
        dfs = {mode: dfs[mode].reset_index(drop=True) for mode in ['train', 'val', 'test'] if mode in dfs}
        #
        # No group should occur in more than one set of a fold
        sets = [set(df[args.group_col]) for df in dfs.values()]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                assert len(sets[i] & sets[j]) == 0, f'Group leakage between sets in fold {fold}.'
        #
        save_fold(args.out_dir, fold, dfs, args.suffixes)
        summary.extend(summarize(fold, dfs, args.group_col, args.label_cols))
    #
    df_summary = pd.DataFrame(summary).fillna(0)
    df_summary.to_csv(os.path.join(args.out_dir, 'fold_distribution.csv'), index=False)
    print(df_summary.to_string(index=False))
    print(f'\nSaved {args.k} folds to {args.out_dir}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create stratified group k-fold splits for survival and classification')
    #
    parser.add_argument('--df_train', type=str, required=True, help='path to the train csv (one row per slide)')
    parser.add_argument('--df_test', type=str, default=None, help='path to the test csv. Optional for split_mode pooled')
    parser.add_argument('--out_dir', type=str, required=True, help='splits directory to write {fold}/train_*.csv etc. to (pass as --splits_dir to the main scripts)')
    parser.add_argument('--group_col', type=str, default='case_id', help='column with the group id (e.g. patient id); all rows of a group end up in the same set')
    parser.add_argument('--label_cols', type=str, nargs='+', required=True, help='column(s) to stratify on, e.g. the class label, or the censorship column for survival. Multiple columns are combined')
    parser.add_argument('--k', type=int, default=5, help='number of folds')
    parser.add_argument('--split_mode', type=str, default='pooled', choices=['pooled', 'fixed_test', 'holdout'],
                        help="pooled: pool df_train and df_test and k-fold over all. fixed_test: k-fold over df_train only (held-out fold saved as val.csv), df_test is used entirely as test set in every fold. "
                             "holdout: first split df_train into train/test (see --test_size), then continue as fixed_test")
    parser.add_argument('--output_sets', type=str, default=None, choices=['train_test', 'train_val_test'],
                        help="sets saved per fold. train_test: train_{suffix}.csv, test_{suffix}.csv. train_val_test: also val_{suffix}.csv. "
                             "For pooled with train_val_test, fold i is test, fold i+1 is val and the rest is train. "
                             "For fixed_test/holdout with train_test, the held-out fold of df_train is not saved. Default: train_test for pooled, train_val_test otherwise")
    parser.add_argument('--test_size', type=float, default=0.2, help='approximate fraction of df_train used as test set in split_mode holdout')
    parser.add_argument('--suffixes', type=str, nargs='+', default=['filtered'],
                        help="file name suffixes, matching --data_filter_type of the main scripts. 'none' gives train.csv/test.csv, 'filtered' gives train_filtered.csv/test_filtered.csv")
    parser.add_argument('--seed', type=int, default=1, help='random seed for the fold assignment')
    #
    args = parser.parse_args()
    create_splits(args)
