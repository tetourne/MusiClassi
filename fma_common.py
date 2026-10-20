"""Shared utilities for the FMA feature analyses.

Used by ``run_ablation.py``, ``run_permutation.py`` and ``run_correlation.py``.
Everything that is configurable lives in ``config.yaml``. This module provides:

* configuration and output-path helpers,
* loading of the feature table and labels (:func:`load_dataset`),
* feature selectors, feature sets and feature groups,
* model construction from the configuration,
* metrics, artist-grouped train/eval splits and paired statistics,
* a small horizontal bar plot helper.
"""
from __future__ import annotations

import importlib
import warnings
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn import metrics as skm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, QuantileTransformer,
                                   RobustScaler, StandardScaler)

Config = dict[str, Any]
"""The parsed ``config.yaml``."""

Groups = dict[str, dict[str, np.ndarray]]
"""``{level: {group_name: positions of the group's columns in X}}``."""

LEVELS: tuple[str, str, str] = ("feature", "statistics", "number")
SUBSET_ORDER: list[str] = ["small", "medium", "large"]


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def load_config(path: str | Path,
                feature_set: str | None = None,
                output_dir: str | None = None) -> Config:
    """Load the YAML configuration, with optional command-line overrides.

    Args:
        path: Path to the YAML configuration file.
        feature_set: If given, overrides ``cfg["feature_set"]``.
        output_dir: If given, overrides ``cfg["paths"]["output_dir"]``.

    Returns:
        The configuration as a nested dictionary.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if feature_set:
        cfg["feature_set"] = feature_set
    if output_dir:
        cfg["paths"]["output_dir"] = output_dir
    return cfg


def output_path(cfg: Config, name: str) -> Path:
    """Return (and create) the output directory ``<output_dir>/<feature_set>/<name>/``.

    Args:
        cfg: The configuration.
        name: Name of the analysis sub-directory (e.g. ``"ablation"``).

    Returns:
        The existing output directory.
    """
    out = Path(cfg["paths"]["output_dir"]).expanduser() / cfg["feature_set"] / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def as_list(x: Any) -> list[Any]:
    """Wrap a scalar into a list; convert a list or tuple into a list.

    Args:
        x: A scalar, list or tuple.

    Returns:
        ``x`` as a list.
    """
    return list(x) if isinstance(x, (list, tuple)) else [x]


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@dataclass
class Dataset:
    """Feature matrix, labels and metadata of the selected tracks.

    Attributes:
        X: Feature matrix of shape ``(n_tracks, n_columns)``, float64, may contain NaN.
        y: Integer genre labels, indexing into ``classes``.
        columns: Column MultiIndex with levels ``(feature, statistics, number)``;
            its length equals ``X.shape[1]``.
        classes: Genre names, sorted.
        split: Official FMA split (``training``/``validation``/``test``) of each track.
        artist: Artist id of each track, or None if absent from ``tracks.csv``.
        track_ids: FMA track id of each row.
    """

    X: np.ndarray
    y: np.ndarray
    columns: pd.MultiIndex
    classes: list[str]
    split: np.ndarray
    artist: np.ndarray | None
    track_ids: np.ndarray


def selector_mask(columns: pd.MultiIndex, sel: dict[str, Any], strict: bool = False) -> np.ndarray:
    """Return a boolean mask of the columns matching a selector.

    A selector is a dict with any of the keys ``feature``, ``statistics``
    (or ``statistic``), ``number`` (each a name or a list of names; all given
    keys must match), ``columns_csv`` (path to a CSV with columns ``feature``,
    ``statistics``, ``number`` listing the columns to keep) and ``exclude``
    (a nested selector whose matches are removed).

    Args:
        columns: Column MultiIndex of the feature table.
        sel: The selector.
        strict: If True, raise on names that do not exist in ``columns`` (typo guard).

    Returns:
        Boolean array of length ``len(columns)``.

    Raises:
        ValueError: If the selector is empty, or if ``strict`` and a name is unknown.
    """
    mask = np.ones(len(columns), dtype=bool)
    used = False
    for key, level in (("feature", "feature"), ("statistics", "statistics"),
                       ("statistic", "statistics"), ("number", "number")):
        if sel.get(key) is None:
            continue
        used = True
        vals = [str(v).zfill(2) if level == "number" else str(v) for v in as_list(sel[key])]
        present = set(columns.get_level_values(level))
        unknown = [v for v in vals if v not in present]
        if strict and unknown:
            raise ValueError(f"Selector {sel}: unknown {level} value(s) {unknown}")
        mask &= columns.get_level_values(level).isin(vals)
    if sel.get("columns_csv") is not None:   # CSV with columns feature, statistics, number
        used = True
        kept = pd.read_csv(Path(sel["columns_csv"]).expanduser(), dtype=str)
        kept_idx = pd.MultiIndex.from_arrays(
            [kept["feature"], kept["statistics"], kept["number"].str.zfill(2)])
        mask &= columns.isin(kept_idx)
    if sel.get("exclude"):
        used = True
        mask &= ~selector_mask(columns, sel["exclude"], strict)
    if not used:
        raise ValueError(f"Empty selector: {sel}")
    return mask


def check_families(cfg: Config, features: Iterable[str]) -> None:
    """Validate the ``families`` block of the config against the features of the CSV.

    Features that belong to no family are omitted from every family-level
    analysis (they only appear at the feature level), so a forgotten or
    misspelled name silently changes the results. This check makes it visible.

    * A feature listed in more than one family is always an error (ambiguous).
    * A feature of the CSV that is in no family, or a name listed in ``families``
      that is not in the CSV (typo?), is an error when ``cfg["strict_families"]``
      is true (the default) and a warning otherwise.

    Args:
        cfg: The configuration (``families`` and ``strict_families``).
        features: Names of all the features (level 0) of the full features CSV,
            before any feature set restricts them.

    Raises:
        ValueError: On duplicated assignments, or on unassigned/unknown names when strict.
    """
    fams = cfg.get("families") or {}
    if not fams:
        return
    owners: dict[str, list[str]] = defaultdict(list)
    for fam, flist in fams.items():
        for f in as_list(flist or []):
            owners[f].append(fam)
    dup = {f: o for f, o in owners.items() if len(o) > 1}
    if dup:
        raise ValueError(f"Features listed in several families (ambiguous): {dup}")

    present = list(dict.fromkeys(features))
    unknown = sorted(set(owners) - set(present))
    orphan = [f for f in present if f not in owners]
    problems = []
    if unknown:
        problems.append(f"listed in `families` but absent from the features CSV (typo?): {unknown}")
    if orphan:
        problems.append("in the features CSV but in no family (they would be omitted from every "
                        f"family-level analysis): {orphan}")
    if problems:
        msg = "families check: " + "; ".join(problems)
        if cfg.get("strict_families", True):
            raise ValueError(msg + ". Fix `families` in config.yaml, or set `strict_families: false` "
                                   "to continue with a warning.")
        warnings.warn(msg)


def load_dataset(cfg: Config) -> Dataset:
    """Load the features and labels of the active feature set.

    Reads the feature CSV (3-level column header) and ``tracks.csv``, keeps the
    tracks of the configured subset that have a top genre and at least one valid
    feature, and restricts the columns to ``cfg["feature_set"]``.

    Args:
        cfg: The configuration. Uses ``paths``, ``data.subset``, ``feature_set``
            and ``feature_sets``.

    Returns:
        The dataset (all official splits; splitting is done by :func:`make_splits`).

    Raises:
        ValueError: If ``cfg["feature_set"]`` is not defined in ``feature_sets``, if a
            selector of the feature set references unknown names, or if the
            ``families`` block is inconsistent (see :func:`check_families`).
    """
    paths = cfg["paths"]
    feats = pd.read_csv(Path(paths["features_csv"]).expanduser(), index_col=0, header=[0, 1, 2])
    cols = pd.MultiIndex.from_arrays(
        [feats.columns.get_level_values(i).astype(str) for i in range(3)], names=LEVELS)
    feats.columns = cols.set_levels(
        cols.levels[2].str.zfill(2), level=2)  # type: ignore[call-overload]  # pandas-stubs
    check_families(cfg, feats.columns.get_level_values("feature"))

    tracks = pd.read_csv(Path(paths["tracks_csv"]).expanduser(),
                         index_col=0, header=[0, 1], low_memory=False)
    subset = cfg.get("data", {}).get("subset", "small")
    allowed = SUBSET_ORDER[: SUBSET_ORDER.index(subset) + 1]
    tracks = tracks[tracks["set", "subset"].isin(allowed)]
    tracks = tracks[tracks["track", "genre_top"].notna()]

    # failed extractions = rows that are entirely NaN
    feats = feats.loc[~feats.isna().all(axis=1)]
    common = feats.index.intersection(tracks.index)
    feats, tracks = feats.loc[common], tracks.loc[common]

    # ---- active feature set --------------------------------------------
    fs_name = cfg["feature_set"]
    if fs_name not in cfg["feature_sets"]:
        raise ValueError(f"feature_set '{fs_name}' not in feature_sets: {list(cfg['feature_sets'])}")
    fs = cfg["feature_sets"][fs_name]
    if fs in (None, "all"):
        keep = np.ones(feats.shape[1], dtype=bool)
    else:
        keep = np.zeros(feats.shape[1], dtype=bool)
        for sel in as_list(fs):
            keep |= selector_mask(cast(pd.MultiIndex, feats.columns), sel, strict=True)
    feats = feats.loc[:, keep]

    labels = tracks["track", "genre_top"].astype(str)
    classes = sorted(labels.unique())
    y = labels.map({c: i for i, c in enumerate(classes)}).to_numpy()
    artist = tracks["artist", "id"].to_numpy() if ("artist", "id") in tracks.columns else None

    ds = Dataset(X=feats.to_numpy(dtype=np.float64), y=y, columns=cast(pd.MultiIndex, feats.columns),
                 classes=classes, split=tracks["set", "split"].astype(str).to_numpy(),
                 artist=artist, track_ids=feats.index.to_numpy())
    n_nan = int(np.isnan(ds.X).sum())
    print(f"[data] feature_set='{fs_name}': {ds.X.shape[0]} tracks x {ds.X.shape[1]} columns, "
          f"{len(classes)} genres, {n_nan} NaN cells")
    return ds


# --------------------------------------------------------------------------
# Feature groups
# --------------------------------------------------------------------------
def _groupby_positions(df: pd.DataFrame, keys: list[str]) -> dict[str, np.ndarray]:
    """Group column positions by one or several metadata columns.

    Args:
        df: Frame with the column metadata and a ``pos`` column (position in X).
        keys: Metadata columns to group by.

    Returns:
        ``{group_name: positions}``; multi-key names are joined with ``"/"``.
    """
    out: dict[str, np.ndarray] = {}
    for name, g in df.groupby(keys, sort=True):
        label = "/".join(map(str, name)) if isinstance(name, tuple) else str(name)
        out[label] = g["pos"].to_numpy()
    return out


def _select(groups: dict[str, np.ndarray], spec: Any, level: str) -> dict[str, np.ndarray]:
    """Apply a config spec (``"all"``, a list of names or null) to a dict of groups.

    Names missing from ``groups`` (e.g. features absent from the active feature
    set) are skipped with a warning.

    Args:
        groups: All the available groups of this level.
        spec: ``"all"``/True for every group, None/False for none, or a list of names.
        level: Name of the level, used in warnings.

    Returns:
        The selected groups.
    """
    if spec is None or spec is False:
        return {}
    if spec is True or spec == "all":
        return groups
    out = {}
    for name in as_list(spec):
        if name in groups:
            out[name] = groups[name]
        else:
            warnings.warn(f"[{level}] '{name}' not available in the active feature set "
                          f"(skipped). Available: {list(groups)}")
    return out


def resolve_groups(columns: pd.MultiIndex, cfg: Config, groups_cfg: dict[str, Any]) -> Groups:
    """Build the feature groups requested in the configuration.

    Supported levels: ``feature``, ``statistic``, ``feature_statistic``,
    ``column``, ``family`` (from ``cfg["families"]``) and ``custom``
    (name -> selector or list of selectors). Empty levels are dropped.

    Args:
        columns: Column MultiIndex of the (already restricted) feature matrix.
        cfg: The configuration (for ``families``).
        groups_cfg: The ``groups`` block: for each level ``all``, a list of
            names, or null to skip the level.

    Returns:
        ``{level: {group_name: positions in X}}``.
    """
    df = columns.to_frame(index=False)
    df["pos"] = np.arange(len(df))

    by_feature = _groupby_positions(df, ["feature"])
    resolved: Groups = {}

    if "feature" in groups_cfg:
        resolved["feature"] = _select(by_feature, groups_cfg["feature"], "feature")
    if "statistic" in groups_cfg:
        resolved["statistic"] = _select(_groupby_positions(df, ["statistics"]),
                                        groups_cfg["statistic"], "statistic")
    if "feature_statistic" in groups_cfg:
        resolved["feature_statistic"] = _select(_groupby_positions(df, ["feature", "statistics"]),
                                                groups_cfg["feature_statistic"], "feature_statistic")
    if "column" in groups_cfg:
        cols = {"/".join(c): np.array([i]) for i, c in enumerate(columns)}
        resolved["column"] = _select(cols, groups_cfg["column"], "column")

    if "family" in groups_cfg:
        fams = {}
        for fam, feats in (cfg.get("families") or {}).items():
            pos = [by_feature[f] for f in feats if f in by_feature]
            if pos:
                fams[fam] = np.concatenate(pos)
        for fam, fam_pos in fams.items():
            n_feat = sum(1 for f in (cfg["families"][fam] or []) if f in by_feature)
            print(f"[families] {fam}: {n_feat} feature(s), {len(fam_pos)} column(s)")
        resolved["family"] = _select(fams, groups_cfg["family"], "family")

    custom = groups_cfg.get("custom") or {}
    if custom:
        cust = {}
        for name, sel in custom.items():
            mask = np.zeros(len(columns), dtype=bool)
            for s in as_list(sel):
                mask |= selector_mask(columns, s)
            if mask.any():
                cust[name] = np.flatnonzero(mask)
            else:
                warnings.warn(f"[custom] group '{name}' is empty in this feature set (skipped)")
        resolved["custom"] = cust

    return {lvl: g for lvl, g in resolved.items() if g}


def groups_table(columns: pd.MultiIndex, groups: Groups) -> pd.DataFrame:
    """List the columns of every group, for traceability.

    Args:
        columns: Column MultiIndex of the feature matrix.
        groups: Groups as returned by :func:`resolve_groups`.

    Returns:
        A frame with columns ``level``, ``group`` and ``column`` (``feature/statistic/number``).
    """
    rows = [(lvl, name, "/".join(columns[p]))
            for lvl, g in groups.items() for name, pos in g.items() for p in pos]
    return pd.DataFrame(rows, columns=["level", "group", "column"])


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
ALIASES: dict[str, str] = {
    "RandomForestClassifier": "sklearn.ensemble.RandomForestClassifier",
    "ExtraTreesClassifier": "sklearn.ensemble.ExtraTreesClassifier",
    "GradientBoostingClassifier": "sklearn.ensemble.GradientBoostingClassifier",
    "HistGradientBoostingClassifier": "sklearn.ensemble.HistGradientBoostingClassifier",
    "SVC": "sklearn.svm.SVC",
    "LogisticRegression": "sklearn.linear_model.LogisticRegression",
    "KNeighborsClassifier": "sklearn.neighbors.KNeighborsClassifier",
    "MLPClassifier": "sklearn.neural_network.MLPClassifier",
    "LinearDiscriminantAnalysis": "sklearn.discriminant_analysis.LinearDiscriminantAnalysis",
    "XGBClassifier": "xgboost.XGBClassifier",
    "LGBMClassifier": "lightgbm.LGBMClassifier",
}
SCALERS: dict[str, type] = {"standard": StandardScaler, "robust": RobustScaler,
                            "minmax": MinMaxScaler, "quantile": QuantileTransformer}


def build_model(cfg: Config, seed: int | None = None) -> Pipeline:
    """Build a fresh ``(imputer -> scaler ->) classifier`` pipeline from the config.

    The classifier is given by ``cfg["model"]["class"]`` (short alias or dotted
    path) and ``cfg["model"]["params"]``. If it has a ``random_state`` parameter
    that is not set explicitly, it is set to ``seed``.

    Args:
        cfg: The configuration (``model`` block).
        seed: Random seed for the classifier, or None to leave it untouched.

    Returns:
        An unfitted scikit-learn pipeline; its last step is named ``"model"``.
    """
    mcfg = cfg["model"]
    path = ALIASES.get(mcfg["class"], mcfg["class"])
    module, name = path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), name)
    params = dict(mcfg.get("params") or {})
    est = cls(**params)
    if seed is not None and "random_state" in est.get_params() and "random_state" not in params:
        est.set_params(random_state=seed)

    pre = mcfg.get("preprocessing") or {}
    steps = []
    imp = pre.get("imputer", "median")
    if imp and imp != "none":
        steps.append(("imputer", SimpleImputer(strategy=imp)))
    sc = pre.get("scaler", "none")
    if sc and sc != "none":
        steps.append(("scaler", SCALERS[sc]()))
    steps.append(("model", est))
    return Pipeline(steps)


# --------------------------------------------------------------------------
# Metrics (all "higher is better")
# --------------------------------------------------------------------------
def _f1(avg: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Return an F1-score function with the given averaging.

    Args:
        avg: Averaging mode of :func:`sklearn.metrics.f1_score` (e.g. ``"macro"``).

    Returns:
        A function ``(y_true, y_pred) -> float``.
    """
    return lambda y, p: skm.f1_score(y, p, average=avg, zero_division=0)


METRICS: dict[str, tuple[str, Callable[..., float]]] = {
    "accuracy": ("pred", skm.accuracy_score),
    "balanced_accuracy": ("pred", skm.balanced_accuracy_score),
    "f1_macro": ("pred", _f1("macro")),
    "f1_weighted": ("pred", _f1("weighted")),
    "neg_log_loss": ("proba", lambda y, P, labels: -skm.log_loss(y, P, labels=labels)),
}
"""Metric name -> (``"pred"`` or ``"proba"``, function). The kind tells whether the
function takes predicted labels or predicted probabilities (plus the class labels)."""


def compute_metrics(model: Pipeline, X: np.ndarray, y: np.ndarray, names: list[str]) -> dict[str, float]:
    """Score a fitted model on ``(X, y)`` with several metrics, predicting only once.

    Args:
        model: A fitted pipeline. It needs ``predict_proba`` if a probability
            metric (``neg_log_loss``) is requested.
        X: Feature matrix.
        y: True integer labels.
        names: Metric names, keys of :data:`METRICS`.

    Returns:
        ``{metric_name: value}``; every metric is "higher is better".
    """
    kinds = {METRICS[n][0] for n in names}
    pred = model.predict(X)
    proba = model.predict_proba(X) if "proba" in kinds else None
    out = {}
    for n in names:
        kind, fn = METRICS[n]
        out[n] = float(fn(y, pred) if kind == "pred" else fn(y, proba, model.classes_))
    return out


# --------------------------------------------------------------------------
# Evaluation splits
# --------------------------------------------------------------------------
@dataclass
class Split:
    """One train/evaluation split, as positional indices into the dataset.

    Attributes:
        train: Indices of the training tracks.
        test: Indices of the evaluation tracks.
        repeat: Index of the CV repeat this split belongs to.
        fold: Index of the fold within the repeat.
    """

    train: np.ndarray
    test: np.ndarray
    repeat: int
    fold: int


def make_splits(cfg: Config, ds: Dataset) -> list[Split]:
    """Build the train/evaluation splits described in ``cfg["evaluation"]``.

    With ``scheme: cv``, runs repeated stratified K-fold on the pooled official
    splits listed in ``evaluation.cv.splits``, grouped by artist when
    ``group_by_artist`` is true (``StratifiedGroupKFold``), so that no artist is
    in both the training and the evaluation part. With ``scheme: holdout``,
    returns a single split defined by the official FMA splits.

    Args:
        cfg: The configuration (``evaluation`` block).
        ds: The dataset.

    Returns:
        The list of splits (a single one for ``holdout``).

    Raises:
        ValueError: If artist grouping is requested but artist ids are unavailable.
    """
    ev = cfg["evaluation"]
    seed = ev.get("seed", 42)

    if ev["scheme"] == "holdout":
        h = ev["holdout"]
        tr = np.flatnonzero(np.isin(ds.split, h["train_splits"]))
        te = np.flatnonzero(np.isin(ds.split, h["eval_splits"]))
        print(f"[splits] holdout: train={len(tr)} eval={len(te)} (single split: no variance estimate)")
        return [Split(tr, te, 0, 0)]

    cv = ev["cv"]
    pool = np.flatnonzero(np.isin(ds.split, cv["splits"]))
    use_groups = cv.get("group_by_artist", True)
    artist = ds.artist
    if use_groups and artist is None:
        raise ValueError("group_by_artist=true but ('artist','id') is missing from tracks.csv")
    splits = []
    for rep in range(cv["n_repeats"]):
        rs = seed + rep
        if use_groups:
            assert artist is not None
            it = StratifiedGroupKFold(cv["n_splits"], shuffle=True, random_state=rs).split(
                pool, ds.y[pool], artist[pool])
        else:
            it = StratifiedKFold(cv["n_splits"], shuffle=True, random_state=rs).split(pool, ds.y[pool])
        for fold, (a, b) in enumerate(it):
            splits.append(Split(pool[a], pool[b], rep, fold))
    print(f"[splits] {cv['n_repeats']}x{cv['n_splits']}-fold "
          f"({'artist-grouped, ' if use_groups else ''}stratified) on {len(pool)} tracks "
          f"-> {len(splits)} train/eval splits")
    return splits


# --------------------------------------------------------------------------
# Stats and plots
# --------------------------------------------------------------------------
def paired_summary(diff: np.ndarray | pd.Series | list[float], n_train: float, n_test: float
                   ) -> tuple[float, float, float]:
    """Summarise paired differences with a Nadeau-Bengio corrected resampled t-test.

    The correction accounts for the overlap between training sets across
    folds/repeats; the plain paired t-test is far too optimistic here. Treat the
    p-value as a rough guide, especially when screening many groups.

    Args:
        diff: Per-split differences (e.g. ablated score minus baseline score).
        n_train: Mean number of training samples per split.
        n_test: Mean number of evaluation samples per split.

    Returns:
        ``(mean, std, p_value)``. ``std`` is NaN with fewer than 2 values and
        ``p_value`` is NaN with fewer than 3 (H0: the mean difference is 0).
    """
    d = np.asarray(diff, dtype=float)
    k = len(d)
    mean = d.mean()
    std = d.std(ddof=1) if k > 1 else np.nan
    p = np.nan
    if k >= 3:
        var = d.var(ddof=1)
        if var > 0:
            t = mean / np.sqrt((1.0 / k + n_test / n_train) * var)
            p = 2 * stats.t.sf(abs(t), k - 1)
        else:
            p = 1.0 if mean == 0 else 0.0
    return mean, std, p


def barh_plot(df: pd.DataFrame, label: str, value: str, err: str, title: str,
              path: str | Path, xlabel: str,
              vlines: list[tuple[float, str, str]] | None = None, top_n: int = 40) -> None:
    """Save a horizontal bar chart with error bars (first row of ``df`` on top).

    Args:
        df: Rows already sorted from most to least important.
        label: Column of ``df`` with the bar labels.
        value: Column with the bar lengths.
        err: Column with the error-bar half-lengths (NaN is treated as 0).
        title: Figure title.
        path: Output image path.
        xlabel: Label of the x axis.
        vlines: Optional reference lines as ``(x, legend_text, linestyle)``.
        top_n: Maximum number of bars drawn.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.28 * len(df) + 1.5)))
    ax.barh(df[label], df[value], xerr=df[err].fillna(0), capsize=2, color="#4C72B0")
    for x, text, style in (vlines or []):
        ax.axvline(x, color="k", linestyle=style, linewidth=1, label=text)
    if vlines:
        ax.legend(loc="lower right")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
