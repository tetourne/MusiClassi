"""Correlation analysis of the features (redundancy between columns, features and families).

Outputs (per method, under ``results/<feature_set>/correlation/<method>/``):

* ``corr_matrix.csv``: full column x column matrix (optional).
* ``heatmap_full_{grouped,clustered}.png``: full matrix, ordered by family/feature or by clustering.
* ``top_pairs.csv`` / ``top_pairs_cross_feature.csv``: most correlated pairs of columns.
* ``pair_categories.csv``: how many pairs exceed each threshold, by type of pair.
* ``{all,stat_<s>}_{feature,family}_{meanabs,maxabs}.csv/png``: feature- and family-level matrices.
* ``statistic_matched_meanabs.csv/png``: redundancy between the statistics
  (same feature and coefficient).
* ``stat_<s>_heatmap_{grouped,clustered}.png``: sub-matrix restricted to one statistic (e.g. ``mean``).
* ``vif.csv``, ``feature_redundancy.csv/png``: how well each column is explained by ALL the others.
* ``redundancy_curve.csv/png``: number of redundancy clusters versus |r| threshold.
* ``clusters.csv``, ``kept_columns.csv``, ``pruning_by_feature.csv``: suggested pruning at one threshold.

Usage:
    python run_correlation.py --config config.yaml [--feature-set compact] [--output-dir results]
"""
from __future__ import annotations

import argparse
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

import fma_common as fc

STAT_ORDER: list[str] = ["mean", "std", "skew", "kurtosis", "median", "min", "max", "10th", "90th", "value"]
"""Canonical order of the statistics (``value`` is the single statistic of scalar features)."""

CATEGORIES: list[str] = [
    "same feature, same statistic (other coefficient/bin)",
    "same feature, same coefficient, other statistic",
    "same feature, other coefficient and statistic",
    "different features, same family",
    "different families",
]
"""Types of column pairs; the position in this list is the category code."""

CROSS_FEATURE: tuple[int, int] = (3, 4)
"""Category codes of the pairs involving two different features."""


# --------------------------------------------------------------------------
# Column metadata and ordering
# --------------------------------------------------------------------------
def column_info(columns: pd.MultiIndex, cfg: fc.Config) -> pd.DataFrame:
    """Build a metadata table with one row per column.

    Args:
        columns: Column MultiIndex ``(feature, statistics, number)``.
        cfg: The configuration (``families`` block).

    Returns:
        A frame with columns ``feature``, ``statistics``, ``number``, ``family``
        (``"(none)"`` if the feature is in no family) and ``name``
        (``feature/statistic/number``).
    """
    fam_of = {f: fam for fam, fl in (cfg.get("families") or {}).items() for f in fl}
    df = columns.to_frame(index=False)
    df["family"] = df["feature"].map(fam_of).fillna("(none)")
    df["name"] = df["feature"] + "/" + df["statistics"] + "/" + df["number"]
    return df


def grouped_order(info: pd.DataFrame, cfg: fc.Config) -> np.ndarray:
    """Order columns by family, feature (config order), statistic and coefficient.

    Args:
        info: Column metadata from :func:`column_info`.
        cfg: The configuration (``families`` block, defines the family/feature order).

    Returns:
        Permutation of ``range(len(info))`` giving the grouped order.
    """
    fams = list(cfg.get("families") or {})
    feat_rank = {f: i for i, f in enumerate(f for fl in (cfg.get("families") or {}).values() for f in fl)}
    fam_r = info["family"].map({f: i for i, f in enumerate(fams)}).fillna(len(fams)).to_numpy()
    feat_r = info["feature"].map(feat_rank).fillna(10_000).to_numpy()
    name_r = pd.factorize(info["feature"], sort=True)[0]
    stat_r = info["statistics"].map({s: i for i, s in enumerate(STAT_ORDER)}).fillna(99).to_numpy()
    num_r = info["number"].astype(int).to_numpy()
    return np.lexsort((num_r, stat_r, name_r, feat_r, fam_r))


def runs(labels: list[str]) -> list[tuple[int, int, str]]:
    """Split a sequence of labels into contiguous runs of equal labels.

    Args:
        labels: The labels.

    Returns:
        ``[(start, stop, label)]`` with ``stop`` exclusive.
    """
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i, labels[start]))
            start = i
    return out


# --------------------------------------------------------------------------
# Core computations
# --------------------------------------------------------------------------
def prepare(X: np.ndarray, info: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Median-impute the (few) NaN and drop columns that are empty or constant.

    This is a descriptive, label-free step computed on all the tracks given (not
    per CV fold): it is only used to compute correlations, never to fit or
    evaluate a model. Model evaluation always imputes/scales inside the training
    fold, through the pipeline of :func:`fma_common.build_model`.

    Args:
        X: Feature matrix of shape ``(n_tracks, n_columns)``.
        info: Column metadata, one row per column of ``X``.

    Returns:
        The cleaned matrix and the matching (re-indexed) metadata.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), med, X)
    keep = np.isfinite(X).all(axis=0) & (X.std(axis=0) > 0)
    if not keep.all():
        print(f"[prep] dropped {int((~keep).sum())} empty/constant column(s): "
              f"{list(info.loc[~keep, 'name'][:10])}")
    return X[:, keep], info.loc[keep].reset_index(drop=True)


def corr_matrix(X: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix of the columns of ``X``, via one matrix product.

    Args:
        X: Matrix without NaN or constant columns.

    Returns:
        Symmetric ``(n_columns, n_columns)`` matrix with a unit diagonal.
    """
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    R = Z.T @ Z / len(Z)
    np.clip(R, -1, 1, out=R)
    np.fill_diagonal(R, 1.0)
    return R


def block_aggregate(A: np.ndarray, groups: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean and max of |r| between every pair of groups of columns.

    On the diagonal the value is computed within the group, excluding the
    self-correlation of each column (NaN for single-column groups).

    Args:
        A: Matrix of absolute correlations.
        groups: ``{group_name: positions of its columns in A}``.

    Returns:
        ``(mean_abs, max_abs)``, two square frames indexed by group name.
    """
    names = list(groups)
    mean = np.full((len(names), len(names)), np.nan)
    mx = mean.copy()
    for a, na in enumerate(names):
        for b in range(a, len(names)):
            ia, ib = groups[na], groups[names[b]]
            if a == b:
                sub = A[np.ix_(ia, ia)]
                vals = sub[~np.eye(len(ia), dtype=bool)]
            else:
                vals = A[np.ix_(ia, ib)].ravel()
            if vals.size:
                mean[a, b] = mean[b, a] = vals.mean()
                mx[a, b] = mx[b, a] = vals.max()
    return pd.DataFrame(mean, index=names, columns=names), pd.DataFrame(mx, index=names, columns=names)


def statistic_matched(A: np.ndarray, info: pd.DataFrame) -> pd.DataFrame:
    """Mean |r| between statistics, over the (feature, coefficient) pairs having both.

    Args:
        A: Matrix of absolute correlations.
        info: Column metadata.

    Returns:
        Square frame indexed by statistic (scalar ``value`` columns excluded).
    """
    pos: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for i, (f, n, s) in enumerate(zip(info["feature"], info["number"], info["statistics"])):
        pos[(f, n)][s] = i
    acc: dict[frozenset, list[float]] = defaultdict(list)
    for d in pos.values():
        for (sa, ia), (sb, ib) in combinations(d.items(), 2):
            acc[frozenset((sa, sb))].append(A[ia, ib])
    present = [s for s in STAT_ORDER if s in set(info["statistics"]) and s != "value"]
    M = pd.DataFrame(np.nan, index=present, columns=present)
    for s in present:
        M.loc[s, s] = 1.0
    for key, vals in acc.items():
        if len(key) == 2 and all(k in present for k in key):
            sa, sb = tuple(key)
            M.loc[sa, sb] = M.loc[sb, sa] = float(np.mean(vals))
    return M


def pair_tables(R: np.ndarray, info: pd.DataFrame, thresholds: list[float], pair_thr: float,
                max_pairs: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Classify all column pairs and list the most correlated ones.

    Args:
        R: Correlation matrix.
        info: Column metadata.
        thresholds: |r| thresholds used for the per-category counts.
        pair_thr: Minimum |r| for a pair to be listed.
        max_pairs: Maximum number of listed pairs per table.

    Returns:
        ``(counts, top_pairs, top_pairs_cross_feature)``: the number of pairs per
        category and threshold (over all pairs), the listed pairs sorted by |r|,
        and the same restricted to pairs of two different features.
    """
    n = len(info)
    iu, ju = np.triu_indices(n, 1)
    r = R[iu, ju]
    f, s, num, fam = (pd.factorize(info[c])[0] for c in ("feature", "statistics", "number", "family"))
    same_f, same_s = f[iu] == f[ju], s[iu] == s[ju]
    same_n = num[iu] == num[ju]
    same_fam = (fam[iu] == fam[ju]) & (info["family"].to_numpy()[iu] != "(none)")
    cat = np.select([same_f & same_s, same_f & same_n & ~same_s, same_f,
                     ~same_f & same_fam], [0, 1, 2, 3], default=4)

    # counts per category and threshold (all pairs, not only the saved ones)
    absr = np.abs(r)
    rows = []
    for c, label in enumerate(CATEGORIES):
        m = cat == c
        row: dict[str, Any] = {"category": label, "n_pairs": int(m.sum())}
        for t in thresholds:
            row[f"|r|>={t}"] = int((absr[m] >= t).sum())
        rows.append(row)
    counts = pd.DataFrame(rows)

    def build(mask: np.ndarray) -> pd.DataFrame:
        """List the pairs of ``mask`` above ``pair_thr``, sorted by decreasing |r|."""
        sel = np.flatnonzero(mask & (absr >= pair_thr))
        sel = sel[np.argsort(-absr[sel])][:max_pairs]
        a, b = iu[sel], ju[sel]
        return pd.DataFrame({
            "r": r[sel], "abs_r": absr[sel],
            "column_a": info["name"].to_numpy()[a], "column_b": info["name"].to_numpy()[b],
            "family_a": info["family"].to_numpy()[a], "family_b": info["family"].to_numpy()[b],
            "category": [CATEGORIES[c] for c in cat[sel]]})

    return counts, build(np.ones(len(r), bool)), build(np.isin(cat, CROSS_FEATURE))


def vif_table(R: np.ndarray, info: pd.DataFrame, ridge: float = 1e-6) -> pd.DataFrame:
    """Variance inflation factor and R2 of each column against all the others.

    ``VIF_i = [R^-1]_ii`` and ``R2_i = 1 - 1/VIF_i`` is the share of the variance
    of column i explained linearly by all other columns. A tiny ridge keeps the
    inversion stable when columns are (almost) exactly collinear; exactly
    collinear columns therefore show a VIF of about ``1/ridge`` instead of infinity.

    Args:
        R: Correlation matrix.
        info: Column metadata.
        ridge: Value added to the diagonal before inversion.

    Returns:
        The metadata with extra columns ``vif`` and ``r2_with_others``.
    """
    vif = np.clip(np.diag(np.linalg.inv(R + ridge * np.eye(len(R)))), 1.0, None)
    out = info[["name", "feature", "statistics", "number", "family"]].copy()
    out["vif"] = vif
    out["r2_with_others"] = 1.0 - 1.0 / vif
    return out


def cluster_columns(A: np.ndarray) -> np.ndarray:
    """Hierarchical clustering (average linkage) of columns with distance ``1 - |r|``.

    Args:
        A: Matrix of absolute correlations.

    Returns:
        The SciPy linkage matrix.
    """
    D = 1.0 - A
    np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2, 0, None)
    return linkage(squareform(D, checks=False), method="average")


def prune(Z: np.ndarray, A: np.ndarray, info: pd.DataFrame, tau: float, mode: str,
          relevance: np.ndarray | None = None) -> pd.DataFrame:
    """Cut the dendrogram at ``|r| = tau`` and keep one representative per cluster.

    Args:
        Z: Linkage matrix from :func:`cluster_columns`.
        A: Matrix of absolute correlations.
        info: Column metadata.
        tau: Correlation threshold; the dendrogram is cut at distance ``1 - tau``.
        mode: ``"medoid"`` keeps the column with the highest mean |r| to the other
            members (label-free); ``"relevance"`` keeps the one with the highest
            ``relevance`` value.
        relevance: Per-column relevance scores, required for ``mode="relevance"``.

    Returns:
        One row per column with ``cluster``, ``cluster_size``,
        ``mean_abs_r_in_cluster``, ``anova_f`` (if relevance given) and ``keep``,
        sorted by decreasing cluster size.
    """
    labels = fcluster(Z, t=1.0 - tau + 1e-12, criterion="distance")
    keep = np.zeros(len(info), bool)
    mean_in = np.full(len(info), np.nan)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) == 1:
            keep[idx[0]] = True
            continue
        sub = A[np.ix_(idx, idx)]
        score = (sub.sum(axis=1) - 1) / (len(idx) - 1)   # mean |r| to the other members
        mean_in[idx] = score
        if mode == "relevance":
            assert relevance is not None, "representative='relevance' needs relevance scores"
            rep = idx[np.nanargmax(relevance[idx])]
        else:
            rep = idx[np.argmax(score)]
        keep[rep] = True
    out = info[["name", "feature", "statistics", "number", "family"]].copy()
    out["cluster"] = labels
    out["cluster_size"] = out.groupby("cluster")["name"].transform("size")
    out["mean_abs_r_in_cluster"] = mean_in
    if relevance is not None:
        out["anova_f"] = relevance
    out["keep"] = keep
    return out.sort_values(["cluster_size", "cluster"], ascending=[False, True])


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _plt() -> Any:
    """Import matplotlib with the non-interactive backend and return ``pyplot``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def corr_heatmap(R: np.ndarray, info: pd.DataFrame, path: str | Path, title: str, mode: str) -> None:
    """Save a signed correlation heatmap.

    Args:
        R: Correlation matrix, already in the display order.
        info: Column metadata in the same order as ``R``.
        path: Output image path.
        title: Figure title.
        mode: ``"grouped"`` for columns ordered by family/feature (block lines and
            feature labels, labels that would overlap are omitted), or
            ``"clustered"`` for a data-driven order (family colour strip, plus
            column labels when there are at most 200 columns).
    """
    plt = _plt()
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    n = len(info)
    size = 14 if n > 300 else 10
    fams = list(pd.unique(info["family"]))
    colors = plt.get_cmap("tab10")(np.arange(len(fams)) % 10)

    if mode == "grouped":
        fig, ax = plt.subplots(figsize=(size, size))
        im = ax.imshow(R, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
        gap = max(1, round(0.011 * n))   # minimum spacing (in columns) between two labels

        def spaced(rs: list[tuple[int, int, str]]) -> list[tuple[float, str]]:
            """Keep only the runs whose centre is far enough from the previous kept label."""
            kept: list[tuple[float, str]] = []
            last = float("-inf")
            for a, b, lab in rs:
                c = (a + b - 1) / 2
                if c - last >= gap:
                    kept.append((c, lab))
                    last = c
            return kept

        fr = runs(info["feature"].tolist())
        lab = spaced(fr)
        fs = 6 if len(fr) <= 45 else 4
        ax.set_xticks([c for c, _ in lab])
        ax.set_xticklabels([l for _, l in lab], rotation=90, fontsize=fs)
        ax.set_yticks([c for c, _ in lab])
        ax.set_yticklabels([l for _, l in lab], fontsize=fs)
        for a, _, _ in fr:
            ax.axhline(a - 0.5, lw=0.25, color="k")
            ax.axvline(a - 0.5, lw=0.25, color="k")
        fam_runs = runs(info["family"].tolist())
        for a, _, _ in fam_runs:
            ax.axhline(a - 0.5, lw=1.4, color="k")
            ax.axvline(a - 0.5, lw=1.4, color="k")
        for a, b, fam in fam_runs:       # rotated text needs a tall enough block
            if b - a >= 0.10 * n:
                ax.text(-0.17, (a + b - 1) / 2, fam, transform=ax.get_yaxis_transform(), rotation=90,
                        va="center", ha="center", fontsize=9, fontweight="bold")
        note = "families (top to bottom): " + " | ".join(f for _, _, f in fam_runs)
        if len(lab) < len(fr):
            note += "\n(labels of very small features are omitted; see the feature-level matrices)"
        ax.set_xlabel(note, fontsize=7)
    else:
        fig, (axs, ax) = plt.subplots(1, 2, figsize=(size + 1, size),
                                      gridspec_kw=dict(width_ratios=[0.025, 1], wspace=0.01))
        codes = pd.factorize(info["family"])[0]
        axs.imshow(codes[:, None], cmap=ListedColormap(colors), aspect="auto", interpolation="nearest")
        axs.set_xticks([])
        axs.set_yticks([])
        im = ax.imshow(R, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
        ax.set_yticks([])
        if n <= 200:
            ax.set_xticks(range(n))
            ax.set_xticklabels(info["name"], rotation=90, fontsize=3.5)
            ax.set_yticks(range(n))
            ax.set_yticklabels(info["name"], fontsize=3.5)
            ax.yaxis.tick_right()
        else:
            ax.set_xticks([])
        fig.legend(handles=[Patch(color=colors[i], label=f) for i, f in enumerate(fams)],
                   loc="lower left", fontsize=7, ncol=min(len(fams), 6))
    fig.colorbar(im, ax=ax, shrink=0.6, label="correlation")
    fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def block_heatmap(df: pd.DataFrame, path: str | Path, title: str, cmap: str = "viridis",
                  vmin: float = 0.0, vmax: float = 1.0, family_lines: list[int] | None = None) -> None:
    """Save an annotated heatmap of a small square table (features, families, statistics).

    Args:
        df: Square frame; NaN cells are left blank.
        path: Output image path.
        title: Figure title.
        cmap: Matplotlib colormap name.
        vmin: Lower colour limit.
        vmax: Upper colour limit.
        family_lines: Row/column positions at which white separator lines are drawn.
    """
    plt = _plt()
    n = len(df)
    vals = df.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(5, 0.38 * n + 2), max(4.5, 0.38 * n + 1.5)))
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n))
    ax.set_xticklabels(df.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df.index, fontsize=7)
    if n <= 40:
        for i in range(n):
            for j in range(n):
                v = vals[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.5,
                            color="w" if v < 0.6 * (vmax - vmin) + vmin else "k")
    for b in family_lines or []:
        ax.axhline(b - 0.5, color="w", lw=1.5)
        ax.axvline(b - 0.5, color="w", lw=1.5)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# One analysis (per method), in steps
# --------------------------------------------------------------------------
def aggregate_outputs(R: np.ndarray, info: pd.DataFrame, cfg: fc.Config, out: Path, tag: str,
                      plots: bool) -> pd.DataFrame | None:
    """Write the feature- and family-level matrices of mean |r| and max |r|.

    As in the ablation and permutation scripts, features that belong to no family
    are omitted from the family-level matrices (they remain in the feature-level
    ones).

    Args:
        R: Correlation matrix of the given columns.
        info: Column metadata of the same columns.
        cfg: The configuration (for the display order).
        out: Output directory.
        tag: Prefix of the output files (e.g. ``"all"`` or ``"stat_mean"``).
        plots: Whether to also save the heatmaps.

    Returns:
        The family-level matrix of mean |r|, or None if no feature has a family.
    """
    A = np.abs(R)
    order = grouped_order(info, cfg)
    info_o, A_o = info.iloc[order].reset_index(drop=True), A[np.ix_(order, order)]
    feats = {f: np.flatnonzero(info_o["feature"] == f) for f in pd.unique(info_o["feature"])}
    fams = {f: np.flatnonzero(info_o["family"] == f) for f in pd.unique(info_o["family"])
            if f != "(none)"}
    fam_of_feat = info_o.drop_duplicates("feature").set_index("feature")["family"]
    bounds = [i for i, _, _ in runs(fam_of_feat.tolist())][1:]

    fam_mean = None
    for level, groups, lines in (("feature", feats, bounds), ("family", fams, None)):
        if not groups:
            continue
        mean, mx = block_aggregate(A_o, groups)
        if level == "family":
            fam_mean = mean
        mean.to_csv(out / f"{tag}_{level}_meanabs.csv", float_format="%.4f")
        mx.to_csv(out / f"{tag}_{level}_maxabs.csv", float_format="%.4f")
        if plots:
            block_heatmap(mean, out / f"{tag}_{level}_meanabs.png",
                          f"Mean |r| between {level}s ({tag}; diagonal = within {level})",
                          family_lines=lines)
            block_heatmap(mx, out / f"{tag}_{level}_maxabs.png",
                          f"Max |r| between {level}s ({tag}; diagonal = within {level})",
                          family_lines=lines)
    return fam_mean


def step_full_matrix(R: np.ndarray, info: pd.DataFrame, Z: np.ndarray, method: str, cfg: fc.Config,
                     out: Path) -> None:
    """Save the full correlation matrix and its two heatmaps (grouped and clustered).

    Args:
        R: Correlation matrix.
        info: Column metadata.
        Z: Linkage matrix of the columns (gives the clustered order).
        method: ``"spearman"`` or ``"pearson"`` (used in titles).
        cfg: The configuration (``correlation`` and ``output`` blocks).
        out: Output directory.
    """
    ccfg, plots = cfg["correlation"], cfg["output"].get("plots", True)
    order = grouped_order(info, cfg)
    leaves = leaves_list(Z)
    if ccfg.get("save_full_matrix", True):
        names = info["name"].to_numpy()[order]
        pd.DataFrame(R[np.ix_(order, order)], index=names, columns=names).to_csv(
            out / "corr_matrix.csv", float_format="%.4f")
    if plots and ccfg.get("full_heatmap", True):
        corr_heatmap(R[np.ix_(order, order)], info.iloc[order].reset_index(drop=True),
                     out / "heatmap_full_grouped.png", f"{method} correlation, grouped by family/feature",
                     "grouped")
        corr_heatmap(R[np.ix_(leaves, leaves)], info.iloc[leaves].reset_index(drop=True),
                     out / "heatmap_full_clustered.png", f"{method} correlation, hierarchical clustering",
                     "clustered")


def step_pairs(R: np.ndarray, info: pd.DataFrame, cfg: fc.Config, out: Path) -> None:
    """Save and print the pair tables (counts per category, most correlated pairs).

    Args:
        R: Correlation matrix.
        info: Column metadata.
        cfg: The configuration (``correlation`` and ``output`` blocks).
        out: Output directory.
    """
    ccfg = cfg["correlation"]
    thresholds = ccfg.get("thresholds", [0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    top_console = cfg["output"].get("top_n_console", 15)
    counts, top, top_cross = pair_tables(R, info, thresholds, ccfg.get("pair_threshold", 0.9),
                                         ccfg.get("max_pairs_saved", 100_000))
    counts.to_csv(out / "pair_categories.csv", index=False)
    top.to_csv(out / "top_pairs.csv", index=False, float_format="%.4f")
    top_cross.to_csv(out / "top_pairs_cross_feature.csv", index=False, float_format="%.4f")
    print("\nPairs of columns above each |r| threshold, by type of pair:")
    print(counts.to_string(index=False))
    n_dup = int((top["abs_r"] >= 0.999).sum())
    print(f"\n{len(top)} pairs with |r| >= {ccfg.get('pair_threshold', 0.9)} "
          f"({len(top_cross)} between different features); {n_dup} (near-)duplicates with |r| >= 0.999.")
    print(f"\nTop {top_console} pairs between DIFFERENT features:")
    print(top_cross.head(top_console)[["r", "column_a", "column_b", "category"]].to_string(index=False))


def step_aggregates(R: np.ndarray, info: pd.DataFrame, method: str, cfg: fc.Config, out: Path) -> None:
    """Feature/family-level matrices, statistic-vs-statistic matrix and per-statistic sub-matrices.

    Args:
        R: Correlation matrix.
        info: Column metadata.
        method: ``"spearman"`` or ``"pearson"`` (used in titles).
        cfg: The configuration (``correlation`` and ``output`` blocks).
        out: Output directory.
    """
    ccfg, plots = cfg["correlation"], cfg["output"].get("plots", True)
    A = np.abs(R)
    fam_mean = aggregate_outputs(R, info, cfg, out, "all", plots)
    if fam_mean is not None:
        print("\nMean |r| between families (diagonal = within family):")
        print(fam_mean.to_string(float_format=lambda v: f"{v:.2f}"))

    matched = statistic_matched(A, info)
    matched.to_csv(out / "statistic_matched_meanabs.csv", float_format="%.4f")
    if plots:
        block_heatmap(matched, out / "statistic_matched_meanabs.png",
                      "Mean |r| between statistics (same feature & coefficient)")
    print("\nMean |r| between statistics of the same feature & coefficient:")
    print(matched.to_string(float_format=lambda v: f"{v:.2f}"))

    for s in ccfg.get("per_statistic", ["mean"]):
        keep_stats = [s, "value"] if ccfg.get("include_scalars", True) else [s]
        idx = np.flatnonzero(info["statistics"].isin(keep_stats))
        if len(idx) < 3:
            print(f"[per_statistic] '{s}': not enough columns, skipped")
            continue
        sub_info = info.iloc[idx].reset_index(drop=True)
        sub_R = R[np.ix_(idx, idx)]
        aggregate_outputs(sub_R, sub_info, cfg, out, f"stat_{s}", plots)
        if plots:
            o = grouped_order(sub_info, cfg)
            corr_heatmap(sub_R[np.ix_(o, o)], sub_info.iloc[o].reset_index(drop=True),
                         out / f"stat_{s}_heatmap_grouped.png", f"{method} correlation, '{s}' columns only",
                         "grouped")
            lv = leaves_list(cluster_columns(np.abs(sub_R)))
            corr_heatmap(sub_R[np.ix_(lv, lv)], sub_info.iloc[lv].reset_index(drop=True),
                         out / f"stat_{s}_heatmap_clustered.png",
                         f"{method} correlation, '{s}' columns only, clustered", "clustered")


def step_vif(R: np.ndarray, info: pd.DataFrame, n_rows: int, method: str, cfg: fc.Config,
             out: Path) -> None:
    """Compute VIF / R2 of each column against all the others and summarise per feature.

    Prints a warning when there are too few tracks for the number of columns,
    because the correlation matrix is then (nearly) singular and the values are
    meaningless.

    Args:
        R: Correlation matrix.
        info: Column metadata.
        n_rows: Number of tracks the correlations were computed on.
        method: ``"spearman"`` or ``"pearson"`` (used in titles).
        cfg: The configuration (``correlation`` and ``output`` blocks).
        out: Output directory.
    """
    n = len(info)
    top_console = cfg["output"].get("top_n_console", 15)
    if n_rows < 3 * n:
        print(f"[warning] only {n_rows} tracks for {n} columns: the correlation matrix is (nearly) "
              f"singular, VIF/R2 values are meaningless (all ~1). Use more tracks or fewer columns.")
    vt = vif_table(R, info)
    vt.sort_values("vif", ascending=False).to_csv(out / "vif.csv", index=False, float_format="%.4g")
    fr = (vt.groupby(["family", "feature"])["r2_with_others"].agg(["median", "min", "max", "size"])
            .reset_index().sort_values("median", ascending=False))
    fr.to_csv(out / "feature_redundancy.csv", index=False, float_format="%.4f")
    print("\nShare of each column's variance explained linearly by all the others (R2), "
          "median per feature:")
    print(fr.head(top_console).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if cfg["output"].get("plots", True):
        fc.barh_plot(fr.assign(zero=0.0), "feature", "median", "zero",
                     f"Redundancy per feature ({method}): median R2 with all other columns",
                     out / "feature_redundancy.png", "R2 (1 = fully explained by the others)",
                     top_n=cfg["output"].get("top_n_plot", 40))


def step_clusters(Z: np.ndarray, A: np.ndarray, info: pd.DataFrame, X_imp: np.ndarray, y: np.ndarray,
                  method: str, cfg: fc.Config, out: Path) -> None:
    """Redundancy curve and suggested pruning of correlated columns.

    Args:
        Z: Linkage matrix of the columns.
        A: Matrix of absolute correlations.
        info: Column metadata.
        X_imp: Cleaned feature matrix (only used for ``representative: relevance``).
        y: Integer labels (only used for ``representative: relevance``).
        method: ``"spearman"`` or ``"pearson"`` (used in titles).
        cfg: The configuration (``correlation`` and ``output`` blocks).
        out: Output directory.
    """
    ccfg = cfg["correlation"]
    thresholds = ccfg.get("thresholds", [0.5, 0.7, 0.8, 0.9, 0.95, 0.99])
    taus = np.round(np.concatenate([np.arange(0.5, 1.0, 0.01), [0.995, 0.999]]), 3)
    curve = pd.DataFrame({"threshold": taus,
                          "n_clusters": [len(np.unique(fcluster(Z, 1.0 - t + 1e-12, "distance")))
                                         for t in taus]})
    curve.to_csv(out / "redundancy_curve.csv", index=False)
    if cfg["output"].get("plots", True):
        plt = _plt()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(curve["threshold"], curve["n_clusters"], marker=".")
        ax.set_xlabel("|r| threshold (average linkage)")
        ax.set_ylabel("number of clusters = effective number of columns")
        ax.set_title(f"Redundancy curve ({method})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "redundancy_curve.png", dpi=150)
        plt.close(fig)
    print("\nNumber of columns left if each cluster of mutually correlated columns is reduced to one:")
    print(curve[curve["threshold"].isin(list(thresholds))].to_string(index=False))

    cl = ccfg.get("clustering", {})
    tau, mode = cl.get("threshold", 0.9), cl.get("representative", "medoid")
    relevance = None
    if mode == "relevance":
        from sklearn.feature_selection import f_classif
        relevance = np.nan_to_num(f_classif(X_imp, y)[0], nan=-1.0)
    table = prune(Z, A, info, tau, mode, relevance)
    table.to_csv(out / "clusters.csv", index=False, float_format="%.4f")
    table[table["keep"]][["feature", "statistics", "number"]].sort_values(
        ["feature", "statistics", "number"]).to_csv(out / "kept_columns.csv", index=False)
    by_feat = (table.groupby(["family", "feature"])["keep"].agg(n_columns="size", n_kept="sum")
                    .reset_index())
    by_feat["pct_kept"] = 100 * by_feat["n_kept"] / by_feat["n_columns"]
    by_feat.to_csv(out / "pruning_by_feature.csv", index=False, float_format="%.1f")
    print(f"\nPruning at |r| >= {tau} (average linkage, representative = {mode}): "
          f"{len(info)} -> {int(table['keep'].sum())} columns")
    print(by_feat.groupby("family")[["n_columns", "n_kept"]].sum().to_string())


def analyse(R: np.ndarray, info: pd.DataFrame, X_imp: np.ndarray, y: np.ndarray, method: str,
            cfg: fc.Config, out: Path, rows_label: str) -> None:
    """Run every analysis step for one correlation matrix and write the results.

    Args:
        R: Correlation matrix (Pearson on raw values or on ranks, i.e. Spearman).
        info: Column metadata, one row per column of ``R``.
        X_imp: Cleaned feature matrix the correlations were computed from.
        y: Integer labels of the same tracks.
        method: ``"spearman"`` or ``"pearson"``.
        cfg: The configuration.
        out: Output directory of this method (created if needed).
        rows_label: Description of the tracks used, printed in the header.
    """
    out.mkdir(parents=True, exist_ok=True)
    A = np.abs(R)
    print(f"\n{'=' * 78}\n{method.upper()} correlation | {len(info)} columns | {rows_label}\n{'=' * 78}")

    Z = cluster_columns(A)
    step_full_matrix(R, info, Z, method, cfg, out)
    step_pairs(R, info, cfg, out)
    step_aggregates(R, info, method, cfg, out)
    if cfg["correlation"].get("vif", True):
        step_vif(R, info, len(X_imp), method, cfg, out)
    step_clusters(Z, A, info, X_imp, y, method, cfg, out)


def main() -> None:
    """Run the correlation analysis described in the configuration."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--feature-set", default=None, help="override cfg['feature_set']")
    ap.add_argument("--output-dir", default=None, help="override cfg['paths']['output_dir']")
    args = ap.parse_args()

    cfg = fc.load_config(args.config, args.feature_set, args.output_dir)
    ccfg = cfg["correlation"]
    out = fc.output_path(cfg, "correlation")

    ds = fc.load_dataset(cfg)
    rows = np.flatnonzero(np.isin(ds.split, ccfg.get("splits", ["training", "validation"])))
    X, y = ds.X[rows], ds.y[rows]
    info = column_info(ds.columns, cfg)
    X, info = prepare(X, info)
    rows_label = f"{len(rows)} tracks from splits {ccfg.get('splits', ['training', 'validation'])}"
    with open(out / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    for method in fc.as_list(ccfg.get("methods", ["spearman", "pearson"])):
        Xm = stats.rankdata(X, axis=0) if method == "spearman" else X
        analyse(corr_matrix(Xm), info, X, y, method, cfg, out / method, rows_label)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
