"""Compare multilabel genre classifiers on the compact and full feature sets.

Multilabel counterpart of compare_models.py (which must stay in the same
directory: seeds, logging and some constants are shared with it).

Labels
------
Each track can have several genres. Two label spaces are available (--genre-level):
* 'all' (default): every genre of the track, parents included (``genres_all`` in
  tracks.csv), which is where the multilabel structure really lives;
* 'top': the top-level genres of the track (the 16 genres of the single-label task).
--max-genres N keeps only the N genres with the most tracks in the development set
(0 = no limit). Tracks left without any of the kept genres are dropped.

Protocol (same as compare_models.py)
------------------------------------
* The official training and validation sets are pooled into one development set.
  Models and hyperparameters are compared with k-fold cross-validation on it.
* Folds keep all the tracks of an artist together (no artist leakage) AND are
  stratified on the genres. scikit-learn's StratifiedGroupKFold cannot do
  multilabel, so ``multilabel_group_folds`` generalises its greedy algorithm:
  artists are assigned one by one to the fold whose per-genre share of positives
  (and number of tracks) then deviates the least from a perfectly even split.
* The same folds are used for every model and feature set (paired comparison).
* The official test set is untouched unless --test is given: the best model of
  each feature set is then refit on the whole development set and evaluated once.
* Imputation and scaling are inside the Pipeline (fitted on training folds only).
* Fold models are discarded right after scoring. --save-models keeps the refit ones.
* --seeds repeats the whole cross-validation once per master seed (new folds and
  new model seeds each time).

Metrics
-------
Macro, micro, samples and weighted F1 are always computed and reported.
--score selects the one used to pick the best configuration of each model and the
best model of each feature set. Macro F1 gives every genre the same weight, micro F1
is dominated by the frequent genres, samples F1 is computed track by track, weighted
F1 weights each genre by its number of positives.

Usage
-----
    python compare_models_multilabel.py                                   # 16 genres, all models
    python compare_models_multilabel.py --max-genres 30 --score micro
    python compare_models_multilabel.py --genre-level top --feature-sets compact --models logreg rf
    python compare_models_multilabel.py --models svm hgb --seeds 0 1 2    # repeated runs for the finalists
    python compare_models_multilabel.py --save-models --test              # final step, once
"""
import argparse
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import ParameterGrid
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC

from compare_models import derive_seeds, format_value, scale_to_fold, setup_loggin
from feature_groups import FEATURE_SET_SPECS, build_feature_sets


DEFAULT_DATA_DIR = os.environ.get('MUSIC_DATA_DIR', '/run/media/thomas/Data/Documents/MusiClassi/fma_metadata')
SUBSET_NAME = 'medium'
FEATURES_CSV_FILE = 'myfeatures_'+SUBSET_NAME+'.csv'
# FEATURES_CSV_FILE = 'features.csv'


# Components that use randomness. Each one gets its own seed derived from --seed.
# (logreg/lbfgs, kNN and SVM are deterministic.)
SEED_KEYS = ('folds', 'scaler', 'rf', 'et', 'hgb', 'mlp', 'resampling')
MODEL_NAMES = ('logreg', 'knn', 'svm', 'rf', 'et', 'hgb', 'mlp')
SCORE_AVERAGES = ('macro', 'micro', 'samples', 'weighted')
SCORES = tuple(f'f1_{avg}' for avg in SCORE_AVERAGES)
DEFAULT_MAX_GENRES = 20



# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class Data(NamedTuple):
    """Development set (official training + validation, pooled) and held-out test set."""
    X_dev: pd.DataFrame
    Y_dev: np.ndarray        # (n_tracks, n_genres) 0/1 indicator matrix
    groups_dev: np.ndarray   # artist id of each development track
    X_test: pd.DataFrame
    Y_test: np.ndarray
    genre_ids: np.ndarray    # FMA genre id of each column of Y
    genre_names: list[str]   # genre name of each column of Y


def load_data(data_dir: str | os.PathLike, genre_level: str, max_genres: int) -> Data:
    """Load features and multilabel genres, split with the official FMA split.

    The most populated genres are chosen on the development set only (never on the
    test set). Tracks without any kept genre are dropped from both sets.

    Args:
        data_dir: Directory containing ``fma_metadata/``.
        genre_level: 'all' (every genre of the track, parents included) or 'top'
            (top-level genres only).
        max_genres: Keep this many genres, the most populated ones; 0 keeps all.

    Returns:
        The development and test sets.
    """
    import utils  # imported here because it pulls in librosa

    meta = Path(data_dir)
    logging.info(f"Reading data from {meta}...")
    tracks = utils.load(meta / 'tracks.csv')
    genres = utils.load(meta / 'genres.csv')
    features = utils.load(meta / FEATURES_CSV_FILE)

    features = features.replace([np.inf, -np.inf], np.nan)
    failed = features.index[features.isna().all(axis=1)]
    if len(failed) > 0:
        logging.info(f"Dropping {len(failed)} tracks whose feature extraction failed.")
    usable = features.index.difference(failed)

    # Set of genre ids of every track, at the requested level.
    if genre_level == 'top':
        top_of = genres['top_level'].to_dict()
        label_sets = tracks['track', 'genres'].map(
            lambda ids: {int(top_of[g]) for g in ids if g in top_of and pd.notna(top_of[g])})
    else:
        known = set(genres.index)
        label_sets = tracks['track', 'genres_all'].map(lambda ids: {int(g) for g in ids if g in known})

    subset = tracks['set', 'subset'] <= SUBSET_NAME
    split = tracks['set', 'split']
    artist = tracks['artist', 'id']

    def select(splits: tuple[str, ...]) -> pd.Index:
        mask = (subset & split.isin(splits)).to_numpy()
        ids = tracks.index[mask]
        return ids[ids.isin(usable)]

    dev_ids, test_ids = select(('training', 'validation')), select(('test',))

    missing_dev = artist.loc[dev_ids].isna()
    missing_test = artist.loc[test_ids].isna()

    logging.warning(
        f"Missing artist IDs: {missing_dev.sum()} development tracks "
        f"({missing_dev.mean():.2%}) and {missing_test.sum()} test tracks "
        f"({missing_test.mean():.2%})."
    )

    # The genres to keep are chosen on the development set (ties broken by genre id).
    counts = Counter(g for s in label_sets.loc[dev_ids] for g in s)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if max_genres:
        ranked = ranked[:max_genres]
    genre_ids = np.array([g for g, _ in ranked])
    titles = genres['title']
    names = [str(titles[g]) for g in genre_ids]
    names = [f"{n} [{g}]" if names.count(n) > 1 else n for n, g in zip(names, genre_ids)]
    if len(genre_ids) < 2:
        raise SystemExit(f"Only {len(genre_ids)} genre(s) available: multilabel needs at least 2.")
    column = {g: j for j, g in enumerate(genre_ids)}

    def indicator(ids: pd.Index) -> np.ndarray:
        Y = np.zeros((len(ids), len(genre_ids)), dtype=np.uint8)
        for i, s in enumerate(label_sets.loc[ids]):
            for g in s:
                if g in column:
                    Y[i, column[g]] = 1
        return Y

    Y_dev, Y_test = indicator(dev_ids), indicator(test_ids)
    keep_dev, keep_test = Y_dev.sum(1) > 0, Y_test.sum(1) > 0
    logging.info(f"Dropping {int((~keep_dev).sum())} development and {int((~keep_test).sum())} test tracks "
                 f"without any of the {len(genre_ids)} kept genres.")
    dev_ids, Y_dev = dev_ids[keep_dev], Y_dev[keep_dev]
    test_ids, Y_test = test_ids[keep_test], Y_test[keep_test]

    data = Data(X_dev=features.loc[dev_ids], Y_dev=Y_dev, groups_dev=artist.loc[dev_ids].to_numpy(),
                X_test=features.loc[test_ids], Y_test=Y_test, genre_ids=genre_ids, genre_names=names)

    shared = np.intersect1d(data.groups_dev, artist.loc[test_ids].to_numpy())
    n_nan = int(data.X_dev.isna().sum().sum())
    cardinality = Y_dev.sum(1)
    logging.info(f"{len(Y_dev)} development tracks ({len(np.unique(data.groups_dev))} artists), "
                 f"{len(Y_test)} test tracks, {len(genre_ids)} genres (level '{genre_level}').")
    logging.info(f"Labels per development track: mean {cardinality.mean():.2f}, max {cardinality.max()}; "
                 f"{(cardinality > 1).mean():.1%} of the tracks have several genres.")
    if cardinality.mean() < 1.05:
        logging.warning("WARNING - almost every track has a single genre: the problem is barely multilabel. "
                        "Try --genre-level all or a larger --max-genres.")
    logging.info("Genres (tracks in development / test sets):\n" + pd.DataFrame(
        {'genre': names, 'dev': Y_dev.sum(0), 'test': Y_test.sum(0)}).to_string(index=False))
    logging.info(f"{features.shape[1]} feature columns; {n_nan} NaN cells in the development set "
                 f"(imputed with the training median inside the pipeline).")
    if len(shared) > 0:
        raise ValueError(
            f"Artist leakage detected: {len(shared)} artists "
            "are present in both development and test sets."
        )
    logging.info("No artist leakage between development and test sets.")
    return data


# ---------------------------------------------------------------------------
# Folds: multilabel stratification with artist groups
# ---------------------------------------------------------------------------
def multilabel_group_folds(Y: np.ndarray, groups: np.ndarray, n_splits: int,
                           seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group k-fold stratified on a multilabel indicator matrix.

    Generalisation of the greedy algorithm of scikit-learn's StratifiedGroupKFold.
    Each group (artist) is summarised by its number of positives per genre plus its
    number of tracks. Groups are visited from the ones holding the largest share of
    any genre (or of the tracks) down to the smallest, ties being shuffled. Each
    group goes to the fold for which the spread, across folds, of the share of
    every genre (and of the tracks) is the smallest; ties go to the smallest fold.
    A group is never split, so no artist is in two folds.

    Args:
        Y: (n_tracks, n_genres) 0/1 indicator matrix.
        groups: Artist id of every track.
        n_splits: Number of folds.
        seed: Seed of the shuffling.

    Returns:
        List of (train_idx, val_idx) pairs of sorted indices.
    """
    rng = np.random.default_rng(seed)
    artists, inverse = np.unique(groups, return_inverse=True)
    # One column per genre, plus a last one counting the tracks (balances the fold sizes).
    per_group = np.zeros((len(artists), Y.shape[1] + 1))
    np.add.at(per_group, inverse, np.column_stack([Y, np.ones(len(Y))]))
    total = np.maximum(per_group.sum(0), 1)
    share = per_group / total

    order = rng.permutation(len(artists))
    order = order[np.argsort(-share[order].max(axis=1), kind='stable')]

    fold_share = np.zeros((n_splits, per_group.shape[1]))
    fold_of_group = np.empty(len(artists), dtype=np.int64)
    for g in order:
        best, best_spread, best_size = 0, np.inf, np.inf
        for k in range(n_splits):
            fold_share[k] += share[g]
            spread = np.std(fold_share, axis=0).mean()
            fold_share[k] -= share[g]
            size = fold_share[k, -1]
            if spread < best_spread - 1e-12 or (abs(spread - best_spread) <= 1e-12 and size < best_size):
                best, best_spread, best_size = k, spread, size
        fold_share[best] += share[g]
        fold_of_group[g] = best

    fold_of_track = fold_of_group[inverse]
    return [(np.flatnonzero(fold_of_track != k), np.flatnonzero(fold_of_track == k))
            for k in range(n_splits)]


def log_fold_balance(Y: np.ndarray, groups: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]],
                     names: list[str]) -> None:
    """Log how well the folds are balanced, and check that no artist is in both parts of a fold."""
    n_folds = len(folds)
    sizes = [len(val) for _, val in folds]
    share = np.array([Y[val].sum(0) for _, val in folds]) / np.maximum(Y.sum(0), 1)  # (fold, genre)
    leaks = sum(len(np.intersect1d(groups[tr], groups[val])) for tr, val in folds)
    logging.info(f"Fold sizes (validation): {sizes}. Share of each genre's positives per fold: "
                 f"min {share.min():.3f}, max {share.max():.3f} (perfect: {1 / n_folds:.3f}).")
    logging.info(f"Artists present in both the training and validation part of a fold: {leaks} (must be 0).")
    n_artists = [len(np.unique(groups[Y[:, j] == 1])) for j in range(Y.shape[1])]
    few = [f"{names[j]} ({n_artists[j]})" for j in range(Y.shape[1]) if n_artists[j] < n_folds]
    if few:
        logging.warning(f"WARNING - genres with fewer artists than folds, some validation folds "
                        f"have no positive for them: {', '.join(few)}")


def check_fold_labels(Y, folds, names):
    for fold_id, (train_idx, val_idx) in enumerate(folds):

        train_counts = Y[train_idx].sum(axis=0)
        val_counts = Y[val_idx].sum(axis=0)
        missing_train = [names[j] for j in np.flatnonzero(train_counts == 0)]

        if missing_train:
            raise ValueError(
                f"Fold {fold_id}: no positive training examples "
                f"for genres: {missing_train}"
            )

# ---------------------------------------------------------------------------
# Resampling (multilabel versions of the under / oversampling)
# ---------------------------------------------------------------------------
def undersample_indices(idx: np.ndarray, Y: np.ndarray, max_per_genre: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Remove tracks until no genre has more than ``max_per_genre`` positive tracks.

    Genres are processed from the most to the least populated. To bring a genre down
    to the cap, the tracks removed are those whose genres are the most frequent (the
    track score is the sum over its genres of 1 / current genre count, lowest score
    removed first, ties at random), so the tracks carrying rare genres are kept.
    Removing tracks can lower other genres below the cap; genres at or under the cap
    are never touched on their own account.

    Args:
        idx: Indices (into ``Y``) of the tracks to undersample.
        Y: Indicator matrix of the development set.
        max_per_genre: Maximum number of positive tracks per genre.
        rng: Random generator.

    Returns:
        The sorted indices of the kept tracks.
    """
    idx = np.asarray(idx)
    Ysub = Y[idx].astype(np.int64)
    keep = np.ones(len(idx), dtype=bool)
    counts = Ysub.sum(0)
    while counts.max() > max_per_genre:
        c = int(np.argmax(counts))
        members = np.flatnonzero(keep & (Ysub[:, c] == 1))
        score = (Ysub[members] / np.maximum(counts, 1)).sum(axis=1)
        removed = members[np.lexsort((rng.random(len(members)), score))[:counts[c] - max_per_genre]]
        keep[removed] = False
        counts = Ysub[keep].sum(0)
    return np.sort(idx[keep])


def oversample_indices(idx: np.ndarray, Y: np.ndarray, min_per_genre: int,
                       rng: np.random.Generator) -> np.ndarray:
    """Duplicate tracks until every genre has at least ``min_per_genre`` positive rows.

    Genres are processed from the rarest to the most frequent. For a genre below the
    floor, its positive tracks receive the missing copies as evenly as possible (the
    remainder goes to tracks drawn at random). The copies also count for the other
    genres of those tracks. Genres with no positive track cannot be oversampled.

    Args:
        idx: Indices (into ``Y``) of the tracks to oversample.
        Y: Indicator matrix of the development set.
        min_per_genre: Minimum number of positive rows per genre.
        rng: Random generator.

    Returns:
        The sorted indices, with duplicates, of the oversampled tracks.
    """
    idx = np.asarray(idx)
    Ysub = Y[idx].astype(np.int64)
    copies = np.ones(len(idx), dtype=np.int64)
    for c in np.argsort(Ysub.sum(0), kind='stable'):
        members = np.flatnonzero(Ysub[:, c] == 1)
        missing = min_per_genre - int(copies @ Ysub[:, c])
        if missing <= 0 or len(members) == 0:
            continue
        extra = np.full(len(members), missing // len(members))
        extra[rng.choice(len(members), size=missing % len(members), replace=False)] += 1
        copies[members] += extra
    return np.sort(np.repeat(idx, copies))


def resample_indices(idx: np.ndarray, Y: np.ndarray, undersampling: int | None,
                     oversampling: int | None, rng: np.random.Generator) -> np.ndarray:
    """Undersample (genres above ``undersampling``), then oversample (genres below ``oversampling``).

    Either can be None. The result may contain duplicates (oversampling), which is
    fine to index the training data.
    """
    if undersampling is not None:
        idx = undersample_indices(idx, Y, undersampling, rng)
    if oversampling is not None:
        idx = oversample_indices(idx, Y, oversampling, rng)
    return idx


def log_resampling(Y: np.ndarray, before: np.ndarray, after: np.ndarray, names: list[str],
                   title: str) -> None:
    """Log the number of positive tracks per genre before / after the resampling."""
    rows = {'(all tracks)': {'rows_before': len(before), 'rows_after': len(after),
                             'distinct_after': len(np.unique(after))}}
    for j, name in enumerate(names):
        a = after[Y[after, j] == 1]
        rows[name] = {'rows_before': int(Y[before, j].sum()), 'rows_after': len(a),
                      'distinct_after': len(np.unique(a))}
    logging.info(f"{title}:\n{pd.DataFrame.from_dict(rows, orient='index').to_string()}")


def describe_resampling(undersampling: int | None, oversampling: int | None) -> str:
    """One-line description of the resampling, for the logs."""
    parts = []
    if undersampling is not None:
        parts.append(f"undersampling to at most {undersampling} positive tracks per genre")
    if oversampling is not None:
        parts.append(f"oversampling (duplication) to at least {oversampling} positive tracks per genre")
    return ' then '.join(parts)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_models(n_features: int, seeds: dict[str, int],
                 scaler: str) -> dict[str, tuple[Pipeline, dict[str, list]]]:
    """Build the multilabel pipelines and their hyperparameter grids.

    Models without native multilabel support (logreg, svm, hgb) are wrapped in
    OneVsRestClassifier: one binary model per genre, so ``class_weight='balanced'``
    acts genre by genre. kNN, random forest, extra trees and the MLP handle
    multilabel targets natively. Their ``class_weight='balanced'`` is not used
    for the forests: with several outputs scikit-learn multiplies the weights of
    all the genres, which is not a meaningful balancing.

    Args:
        n_features: Number of input columns (the SVM ``gamma`` grid depends on it).
        seeds: Seeds from :func:`derive_seeds`.
        scaler: 'standard' or 'quantile'; applied before the scale-sensitive models.

    Returns:
        A dict mapping each name of ``MODEL_NAMES`` to ``(pipeline, grid)``.
        The grid keys are Pipeline-style ('clf__estimator__C' for wrapped models).
    """

    def pipe(clf, scale: bool) -> Pipeline:
        steps = [('impute', SimpleImputer(strategy='median'))]
        if scale:
            steps.append(('scale', QuantileTransformer(output_distribution='normal',
                                                       random_state=seeds['scaler'])
                          if scaler == 'quantile' else StandardScaler()))
        steps.append(('clf', clf))
        return Pipeline(steps)

    gammas = [m / n_features for m in (0.5, 0.75, 1)]

    return {
        'logreg': (pipe(OneVsRestClassifier(LogisticRegression(max_iter=2000)), True),
                   {'clf__estimator__C': [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01,0.03],
                    'clf__estimator__class_weight': [None, 'balanced']}),
        'knn': (pipe(KNeighborsClassifier(), True),
                {'clf__n_neighbors': [1, 5, 10, 20],
                 'clf__weights': ['uniform', 'distance'],
                 'clf__p': [1, 2]}),
        'svm': (pipe(OneVsRestClassifier(SVC(kernel='rbf')), True),
                {'clf__estimator__C': [1, 2, 3],
                 'clf__estimator__gamma': gammas,
                 'clf__estimator__class_weight': ['balanced']}),
        'rf': (pipe(RandomForestClassifier(n_jobs=1, n_estimators=300,
                                           random_state=seeds['rf']), False),
               {'clf__max_features': ["sqrt", 0.1,0.3, 0.6],
                'clf__min_samples_leaf': [1, 5],
                'clf__min_samples_split': [2]}),
        'et': (pipe(ExtraTreesClassifier(n_jobs=1, n_estimators=300,
                                         random_state=seeds['et']), False),
               {'clf__max_features': [0.6, 0.8, 1],
                'clf__min_samples_leaf': [1, 5],
                'clf__min_samples_split': [2, 5]}),
        'hgb': (pipe(OneVsRestClassifier(HistGradientBoostingClassifier(
                    early_stopping=False, max_iter=100, min_samples_leaf=1,
                    random_state=seeds['hgb'])), False),
                {'clf__estimator__learning_rate': [0.3, 1],
                 'clf__estimator__max_leaf_nodes': [5],
                 'clf__estimator__min_samples_leaf': [1],
                 'clf__estimator__l2_regularization': [0, 1],
                 'clf__estimator__class_weight': ['balanced']}),
        'mlp': (pipe(MLPClassifier(max_iter=200, early_stopping=False,
                                   random_state=seeds['mlp']), True),
                {'clf__hidden_layer_sizes': [(128, 64), (128, 64, 32), (256,), (256, 128), (256, 128, 64), (512,), (512, 256), (1024,)],
                 'clf__alpha': [0.3, 0.6, 1]}),
    }


def _clean_key(key: str) -> str:
    """Hyperparameter name without the 'clf__' / 'estimator__' pipeline prefixes."""
    return key.removeprefix('clf__').removeprefix('estimator__')


def format_params(params: dict) -> str:
    """Readable one-line description of a hyperparameter dict."""
    return ', '.join(f"{_clean_key(key)}={format_value(value)}" for key, value in params.items())


def format_grid(grid: dict[str, list]) -> str:
    """Readable multi-line description of a hyperparameter grid, with the number of configurations."""
    n_configs = int(np.prod([len(values) for values in grid.values()]))
    lines = [f"    {_clean_key(key)}: [{', '.join(format_value(v) for v in values)}]"
             for key, values in grid.items()]
    return '\n'.join(lines + [f"    -> {n_configs} configurations"])


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def multilabel_scores(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict[str, float]:
    """Macro, micro, samples and weighted F1 (undefined F1 counts as 0)."""
    return {f'f1_{avg}': float(f1_score(Y_true, Y_pred, average=avg, zero_division=0))
            for avg in SCORE_AVERAGES}


def _n_iter(clf) -> float:
    """Mean number of iterations run by the iterative models (HGB, MLP); NaN for the others."""
    members = getattr(clf, 'estimators_', [clf])  # OneVsRestClassifier holds one model per genre
    values = [float(np.mean(m.n_iter_)) for m in members
              if isinstance(m, (HistGradientBoostingClassifier, MLPClassifier))]
    return float(np.mean(values)) if values else np.nan


def _fit_predict(pipeline: Pipeline, params: dict, X: np.ndarray, Y: np.ndarray,
                 train_idx: np.ndarray, val_idx: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit one configuration on one training fold.

    Returns:
        (0/1 predictions of shape (n_val, n_genres), fit time in seconds, number of
        iterations for the iterative models, NaN otherwise). The model is discarded.
    """
    model = clone(pipeline).set_params(**params)
    t0 = time.perf_counter()
    model.fit(X[train_idx], Y[train_idx])
    fit_time = time.perf_counter() - t0
    pred = model.predict(X[val_idx])
    if hasattr(pred, 'toarray'):
        pred = pred.toarray()
    pred = np.asarray(pred).reshape(len(val_idx), -1).astype(np.uint8)
    return pred, fit_time, _n_iter(model[-1])


def unpack_predictions(packed: np.ndarray, n_genres: int) -> np.ndarray:
    """Inverse of the bit-packing used for the out-of-fold predictions."""
    return np.unpackbits(packed, axis=-1, count=n_genres)


def cross_validate_grid(pipeline: Pipeline, grid: dict[str, list], X: np.ndarray, Y: np.ndarray,
                        folds: list[tuple[np.ndarray, np.ndarray]],
                        n_jobs: int) -> tuple[list[dict], list[dict], np.ndarray]:
    """Cross-validate every configuration of ``grid``.

    All (configuration, fold) fits are independent, so they are distributed over
    worker processes. Only scores and predictions are returned.

    Args:
        pipeline: Unfitted pipeline.
        grid: Hyperparameter grid (Pipeline-style keys).
        X: Feature matrix of the development set.
        Y: Indicator matrix of the genres.
        folds: List of (train_idx, val_idx) pairs.
        n_jobs: Number of worker processes (-1 = all cores).

    Returns:
        configs: the hyperparameter dicts, indexed by config_id.
        rows: one dict per fit (config_id, config, fold, the four F1 scores, fit_time, n_iter).
        oof: out-of-fold predictions, shape (n_configs, n_samples, ceil(n_genres / 8)),
            bit-packed along the genre axis to save memory (see :func:`unpack_predictions`).
    """
    configs = list(ParameterGrid(grid))
    tasks = [(c, f) for c in range(len(configs)) for f in range(len(folds))]
    outputs = Parallel(n_jobs=n_jobs)(
        delayed(_fit_predict)(pipeline, configs[c], X, Y, *folds[f]) for c, f in tasks)

    oof = np.zeros((len(configs), len(Y), (Y.shape[1] + 7) // 8), dtype=np.uint8)
    rows = []
    for (c, f), (pred, fit_time, n_iter) in zip(tasks, outputs):
        val_idx = folds[f][1]
        oof[c, val_idx] = np.packbits(pred, axis=1)
        rows.append({'config_id': c, 'config': format_params(configs[c]), 'fold': f,
                     **multilabel_scores(Y[val_idx], pred),
                     'fit_time': fit_time, 'n_iter': n_iter})
    return configs, rows, oof


SUMMARY_KEYS = ['feature_set', 'n_features', 'model', 'config_id', 'config']


def summarize(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the scores: one row per (feature set, model, configuration).

    For each score ``s``: ``s_mean`` is the mean over all (seed, fold) fits, ``s_std``
    the standard deviation of the single-fit scores, and ``s_seed_std`` the standard
    deviation, across seeds, of the per-seed mean (NaN with a single seed).
    ``n_iter`` is the mean number of iterations (HGB and MLP only).
    """
    agg = {}
    for s in SCORES:
        agg[f'{s}_mean'] = (s, 'mean')
        agg[f'{s}_std'] = (s, 'std')
    summary = (fold_scores.groupby(SUMMARY_KEYS, sort=False)
               .agg(**agg, fit_time_s=('fit_time', 'mean'), n_iter=('n_iter', 'mean'))
               .reset_index())
    per_seed = fold_scores.groupby(SUMMARY_KEYS + ['seed'], sort=False)[list(SCORES)].mean()
    seed_std = (per_seed.groupby(level=SUMMARY_KEYS, sort=False).std()
                .add_suffix('_seed_std'))
    columns = SUMMARY_KEYS + [f'{s}_{stat}' for s in SCORES for stat in ('mean', 'std', 'seed_std')] \
        + ['fit_time_s', 'n_iter']
    return summary.join(seed_std, on=SUMMARY_KEYS)[columns]


def best_configs(summary: pd.DataFrame, score: str) -> pd.DataFrame:
    """Best configuration (highest mean of ``score``) of each (feature set, model)."""
    idx = summary.groupby(['feature_set', 'model'], sort=False)[f'{score}_mean'].idxmax()
    return summary.loc[idx]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report_winner(Y: np.ndarray, pred: np.ndarray, names: list[str], title: str,
                  path: Path, test: bool = False) -> pd.DataFrame:
    """Log a per-genre report and save a per-genre precision / recall / F1 bar chart.

    Returns:
        The per-genre scores (also written next to the chart as a CSV file).
    """
    kind = 'real predictions' if test else 'out-of-fold predictions'
    logging.info(f"\n--- {title} ({kind}) ---")
    logging.info(classification_report(Y, pred, target_names=names, digits=3, zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(Y, pred, zero_division=0)
    per_genre = pd.DataFrame({'genre': names, 'precision': precision, 'recall': recall,
                              'f1': f1, 'support': support}).sort_values('support', ascending=False)
    per_genre.to_csv(path.with_suffix('.csv'), index=False)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(names))))
    pos = np.arange(len(per_genre))
    for offset, column in zip((-0.27, 0, 0.27), ('precision', 'recall', 'f1')):
        ax.barh(pos + offset, per_genre[column], height=0.27, label=column)
    ax.set_yticks(pos, [f"{g} ({s})" for g, s in zip(per_genre.genre, per_genre.support)])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('score (genre name and number of positive tracks)')
    ax.set_title(title)
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return per_genre


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
def run_comparison(data: Data, fsets: dict[str, pd.MultiIndex], model_names: list[str],
                   out_dir: Path, seeds: list[int], scaler: str, n_folds: int, n_jobs: int,
                   do_test: bool, save_models: bool, score: str = 'f1_macro',
                   undersampling: int | None = None, oversampling: int | None = None) -> None:
    """Cross-validate every model on every feature set, once per seed, and write the results to ``out_dir``.

    Args:
        data: Development and test sets.
        fsets: Feature-set name -> selected columns.
        model_names: Names from ``MODEL_NAMES`` to run.
        out_dir: Output directory (created if needed).
        seeds: Master seeds (see :func:`derive_seeds`). The whole cross-validation is
            repeated for each one, with a different fold split and different model seeds.
            The reports and the final refit use the first seed.
        scaler: 'standard' or 'quantile'.
        n_folds: Number of cross-validation folds.
        n_jobs: Number of worker processes (-1 = all cores).
        do_test: Evaluate the best model of each feature set on the test set.
        save_models: Refit the best configuration of every (feature set, model)
            on the whole development set and save it.
        score: One of ``SCORES``; used to pick the best configuration of each model
            and the best model of each feature set.
        undersampling: If set, maximum number of positive tracks per genre *in the whole
            development set*; the training part of every fold is undersampled to
            ``undersampling * (n_folds - 1) / n_folds`` (see :func:`undersample_indices`).
            The final refit uses ``undersampling`` as it is.
        oversampling: If set, minimum number of positive tracks per genre in the whole
            development set (scaled for the folds like ``undersampling``, applied after
            the undersampling if both are set). Validation folds and test set are never
            resampled, so the scores stay comparable.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'genre_id': data.genre_ids, 'genre': data.genre_names,
                  'dev_tracks': data.Y_dev.sum(0), 'test_tracks': data.Y_test.sum(0)}
                 ).to_csv(out_dir / 'genres_used.csv', index=False)
    n_genres = len(data.genre_names)
    logging.info(f"Model selection score: {score}.")

    fold_rows: list[dict] = []
    configs_by_key: dict[tuple[str, str], list[dict]] = {}
    oof_by_key: dict[tuple[str, str], np.ndarray] = {}  # out-of-fold predictions, first seed only

    for seed in seeds:
        component_seeds = derive_seeds(seed)

        # Folds are built once per seed and reused for every model / feature set (paired comparison).
        folds = multilabel_group_folds(data.Y_dev, data.groups_dev, n_folds, component_seeds['folds'])
        check_fold_labels(data.Y_dev, folds, data.genre_names)
        if seed == seeds[0]:
            log_fold_balance(data.Y_dev, data.groups_dev, folds, data.genre_names)

        if undersampling is not None or oversampling is not None:
            # Only the training part of each fold is resampled (validation folds keep the real
            # distribution). Done once per seed: all models / configurations see the same data.
            fold_under, fold_over = scale_to_fold(undersampling, n_folds), scale_to_fold(oversampling, n_folds)
            rng = np.random.default_rng(component_seeds['resampling'])
            resampled_folds = [(resample_indices(train_idx, data.Y_dev, fold_under, fold_over, rng), val_idx)
                               for train_idx, val_idx in folds]
            if seed == seeds[0]:
                log_resampling(data.Y_dev, folds[0][0], resampled_folds[0][0], data.genre_names,
                               f"Resampling of the training part of fold 0, "
                               f"{describe_resampling(fold_under, fold_over)} "
                               f"(requested for the whole set: {undersampling} / {oversampling}, "
                               f"scaled by (n_folds - 1) / n_folds = {n_folds - 1}/{n_folds}; "
                               f"the other folds are similar)")
            logging.info(f"[seed {seed}] Mean training size per fold: "
                         f"{np.mean([len(t) for t, _ in folds]):.0f} -> "
                         f"{np.mean([len(t) for t, _ in resampled_folds]):.0f} rows.")
            folds = resampled_folds

        for fs_name, cols in fsets.items():
            X = data.X_dev.loc[:, cols].to_numpy(np.float32)
            models = build_models(len(cols), component_seeds, scaler)

            for name in model_names:
                pipeline, grid = models[name]
                if seed == seeds[0]:  # the grid does not depend on the seed
                    logging.info(f"Grid for {name} ({fs_name}, {len(cols)} feats):\n{format_grid(grid)}")
                t0 = time.perf_counter()
                configs, rows, oof = cross_validate_grid(pipeline, grid, X, data.Y_dev, folds, n_jobs)
                elapsed = time.perf_counter() - t0

                rows = [{'seed': seed, 'feature_set': fs_name, 'n_features': len(cols),
                         'model': name, **r} for r in rows]
                fold_rows.extend(rows)
                configs_by_key[(fs_name, name)] = configs
                if seed == seeds[0]:
                    oof_by_key[(fs_name, name)] = oof

                best = best_configs(summarize(pd.DataFrame(rows)), score).iloc[0]
                scores_txt = '  '.join(f"{avg}-F1={best[f'f1_{avg}_mean']:.3f}" for avg in SCORE_AVERAGES)
                logging.info(f"[seed {seed} | {fs_name:7s} | {len(cols):4d} feats] {name:6s} "
                             f"{scores_txt}  (best by {score}; {elapsed:.0f}s, "
                             f"{len(configs)} configs x {len(folds)} folds)  {best.config}")

                # Safety copy: a crash during a long run does not lose the finished models.
                pd.DataFrame(fold_rows).to_csv(out_dir / 'fold_scores.csv', index=False)

    # Raw per-fit scores and per-configuration summary: every trained model is in there.
    fold_scores = pd.DataFrame(fold_rows)
    summary = summarize(fold_scores)
    summary.to_csv(out_dir / 'config_scores.csv', index=False)

    best = best_configs(summary, score)
    spread = summary.groupby(['feature_set', 'model'], sort=False)[f'{score}_mean'].agg(
        worst_config='min', n_configs='size')
    table = best.join(spread, on=['feature_set', 'model'])
    other = [f'{s}_mean' for s in SCORES if s != score]
    logging.info(f"\n=== Best configuration per model, selected on {score} "
                 f"(cross-validated; worst_config = worst config of that model) ===")
    logging.info(table.sort_values(['feature_set', f'{score}_mean'], ascending=[True, False])
                 [['feature_set', 'n_features', 'model', f'{score}_mean', f'{score}_std', f'{score}_seed_std',
                   'worst_config', *other, 'n_configs', 'fit_time_s', 'n_iter', 'config']]
                 .dropna(axis=1, how='all')
                 .to_string(index=False, float_format='%.3f'))
    table.to_csv(out_dir / 'best_per_model.csv', index=False)

    winners = best.loc[best.groupby('feature_set', sort=False)[f'{score}_mean'].idxmax()]
    for row in winners.itertuples():
        oof_pred = unpack_predictions(oof_by_key[(row.feature_set, row.model)][row.config_id], n_genres)
        report_winner(data.Y_dev, oof_pred, data.genre_names,
                      f'{row.feature_set} / {row.model} (seed {seeds[0]})',
                      out_dir / f'per_genre_{row.feature_set}_{row.model}.png')

    if not (save_models or do_test):
        return

    # Refit on the whole development set: all best configurations to save them,
    # otherwise only the winner of each feature set (for the test evaluation).
    refit_seeds = derive_seeds(seeds[0])
    models_dir = out_dir / 'models'
    if save_models:
        models_dir.mkdir(exist_ok=True)
    refit_idx = np.arange(len(data.Y_dev))
    if undersampling is not None or oversampling is not None:
        refit_idx = resample_indices(refit_idx, data.Y_dev, undersampling, oversampling,
                                     np.random.default_rng(refit_seeds['resampling']))
        logging.info(f"Final refit on {len(refit_idx)} resampled rows (instead of {len(data.Y_dev)}).")
    for row in (best if save_models else winners).itertuples():
        cols = fsets[row.feature_set]
        pipeline, _ = build_models(len(cols), refit_seeds, scaler)[row.model]
        params = configs_by_key[(row.feature_set, row.model)][row.config_id]
        model = clone(pipeline).set_params(**params).fit(
            data.X_dev.loc[:, cols].to_numpy(np.float32)[refit_idx], data.Y_dev[refit_idx])

        if save_models:
            path = models_dir / f'{row.feature_set}_{row.model}.joblib'
            dump({'pipeline': model, 'columns': list(cols), 'genres': list(data.genre_names),
                  'genre_ids': [int(g) for g in data.genre_ids]}, path, compress=3)
            logging.info(f"Saved {path}")
        if do_test and row.Index in winners.index:
            pred = model.predict(data.X_test.loc[:, cols].to_numpy(np.float32))
            if hasattr(pred, 'toarray'):
                pred = pred.toarray()
            pred = np.asarray(pred).reshape(len(data.Y_test), -1)
            test_scores = multilabel_scores(data.Y_test, pred)
            logging.info(f"TEST  {row.feature_set}/{row.model}: "
                         + '  '.join(f"{avg}-F1={test_scores[f'f1_{avg}']:.3f}" for avg in SCORE_AVERAGES))
            report_winner(data.Y_test, pred, data.genre_names, f"TEST  {row.feature_set}/{row.model}",
                          out_dir / f'per_genre_{row.feature_set}_{row.model}_test.png', test=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                    help="The base directory containing the fma_metadata/ which contains the features, "
                         "tracks.csv and genres.csv")
    ap.add_argument('--out-dir', type=Path, default=Path('results_multilabel'))
    ap.add_argument('--feature-sets', nargs='+', choices=list(FEATURE_SET_SPECS),
                    default=list(FEATURE_SET_SPECS))
    ap.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    ap.add_argument('--scaler', choices=['standard', 'quantile'], default='standard',
                    help="scaler for logreg/knn/svm/mlp (quantile is more robust to skew/kurtosis outliers)")
    ap.add_argument('--seeds', type=int, nargs='+', default=[0],
                    help="master seeds; the whole cross-validation is repeated for each one "
                         "(different folds and model seeds), see derive_seeds()")
    ap.add_argument('--n-folds', type=int, default=5)
    ap.add_argument('--n-jobs', type=int, default=-1,
                    help="worker processes over the (configuration, fold) fits; -1 = all cores")
    ap.add_argument('--genre-level', choices=['all', 'top'], default='all',
                    help="'all': every genre of a track, parents included (genres_all); "
                         "'top': only its top-level genres")
    ap.add_argument('--max-genres', type=int, default=DEFAULT_MAX_GENRES, metavar='N',
                    help="keep only the N most populated genres (counted on the development set); "
                         f"tracks without any kept genre are dropped. 0 = keep all. Default: {DEFAULT_MAX_GENRES}")
    ap.add_argument('--score', choices=SCORE_AVERAGES, default='macro',
                    help="F1 average used to pick the best configuration of each model and the best "
                         "model of each feature set (all four are always reported). Default: macro")
    ap.add_argument('--test', action='store_true',
                    help="evaluate the best model of each feature set on the test set (do this once, at the end)")
    ap.add_argument('--save-models', action='store_true',
                    help="refit the best configuration of each (feature set, model) on the whole development set and save it")
    ap.add_argument('--undersampling', type=int, default=None, metavar='N',
                    help="cap every genre at N positive tracks, N being relative to the whole development "
                         "set. Tracks are removed from the most populated genres first, preferring tracks "
                         "that carry only frequent genres. Each training fold is capped at "
                         "N * (n_folds - 1) / n_folds (e.g. 1600 for N=2000 and 5 folds); the final refit "
                         "uses N. Validation folds and test set are never resampled. Default: none")
    ap.add_argument('--oversampling', type=int, default=None, metavar='N',
                    help="duplicate tracks so that every genre with fewer than N positive tracks reaches N, "
                         "N being relative to the whole development set, like --undersampling (each training "
                         "fold: N * (n_folds - 1) / n_folds; final refit: N). If both options are given, the "
                         "undersampling is done first. Default: none")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_loggin(args.out_dir / 'logs.txt')
    if len(set(args.seeds)) != len(args.seeds):
        ap.error("--seeds must not contain duplicates")
    if args.max_genres < 0 or args.max_genres == 1:
        ap.error("--max-genres must be 0 (no limit) or at least 2")
    if args.undersampling == 0: args.undersampling = None
    if args.oversampling == 0: args.oversampling = None
    if args.undersampling is not None and args.undersampling < 1:
        ap.error("--undersampling must be a positive number of tracks")
    if args.oversampling is not None and args.oversampling < 1:
        ap.error("--oversampling must be a positive number of tracks")
    if None not in (args.undersampling, args.oversampling) and args.oversampling > args.undersampling:
        ap.error("--oversampling must not be larger than --undersampling")

    logging.info(f"The code is run with: {args}.")
    logging.info(f"Seeds used: {args.seeds}.")
    logging.info(f"Results will be saved in {args.out_dir}.")
    logging.info(f"The scaler used: {args.scaler}.")
    logging.info(f"Feature sets: {FEATURE_SET_SPECS}.")

    data = load_data(args.data_dir, args.genre_level, args.max_genres)
    all_sets = build_feature_sets(data.X_dev.columns)
    fsets = {name: all_sets[name] for name in args.feature_sets}
    logging.info(f"Columns per feature set: { {name: len(cols) for name, cols in fsets.items()} }")
    run_comparison(data, fsets, args.models, args.out_dir, seeds=args.seeds, scaler=args.scaler,
                   n_folds=args.n_folds, n_jobs=args.n_jobs, do_test=args.test,
                   save_models=args.save_models, score=f'f1_{args.score}',
                   undersampling=args.undersampling, oversampling=args.oversampling)


if __name__ == '__main__':
    main()
