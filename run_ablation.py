"""Feature ablation: retrain the model with groups of features removed / isolated.

Three kinds of ablation, which can be combined (see the ``ablation`` block of config.yaml):

* One at a time (``modes``). For every group (feature, family, statistic, ...):

  * ``drop``: train on all features EXCEPT the group (does the model miss it?).
  * ``keep``: train on ONLY the group (is it informative alone?).

* Cumulative (``cumulative``). Remove things successively, in the order given:
  full set, then minus A, then minus A and B, then minus A, B and C, and so on.
  Each step is compared with the full set and with the previous step, which shows
  where performance starts to degrade even when no single removal matters (as
  happens with strongly correlated features).

* Configurations (``configurations``). Arbitrary named sets of removals, each one the
  full feature set minus its own list, so that a feature can be removed in one
  configuration and kept in another (for example: minus A; minus A and B; minus A and
  C; minus A, B and C). Every configuration is compared with the full set and with every
  other configuration (paired), which shows which features substitute for each other.

Every variant is evaluated on exactly the same train/eval splits and with the same
model seed as the baseline (all features), so differences are paired.

Usage:
    python run_ablation.py --config config.yaml [--feature-set compact] [--output-dir results]
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

import fma_common as fc

TaskKey = tuple[str, str, str, int, int]
"""``(mode, level, group, n_group_features, split_index)``; mode is ``"baseline"``,
``"drop"``, ``"keep"``, ``"cumulative"`` or ``"configuration"`` (for the last two,
``n_group_features`` is the number of columns removed)."""

Task = tuple[TaskKey, fc.Split, np.ndarray]
"""A model fit to run: its key, the split, and the column positions used."""


def fit_eval(X: np.ndarray, y: np.ndarray, split: fc.Split, cols: np.ndarray, cfg: fc.Config,
             seed: int, metrics: list[str], key: TaskKey) -> tuple[TaskKey, dict[str, float]]:
    """Fit the model on the training part of a split and score it on the evaluation part.

    Args:
        X: Full feature matrix.
        y: Integer labels.
        split: The train/evaluation split.
        cols: Positions of the columns the model may use.
        cfg: The configuration (``model`` block).
        seed: Random seed of the classifier (same for all variants, so they are paired).
        metrics: Names of the metrics to compute.
        key: Identifier of the task, returned unchanged to identify the result.

    Returns:
        ``(key, {metric_name: value})``.
    """
    model = fc.build_model(cfg, seed)
    model.fit(X[np.ix_(split.train, cols)], y[split.train])
    res = fc.compute_metrics(model, X[np.ix_(split.test, cols)], y[split.test], metrics)
    return key, res


# --------------------------------------------------------------------------
# One-at-a-time ablation (drop / keep)
# --------------------------------------------------------------------------
def build_tasks(n_columns: int, groups: fc.Groups, splits: list[fc.Split], modes: list[str]) -> list[Task]:
    """List every model fit of the ablation (baseline plus one per group, mode and split).

    Args:
        n_columns: Total number of feature columns.
        groups: Groups to ablate, as returned by :func:`fma_common.resolve_groups`.
        splits: Train/evaluation splits.
        modes: ``"drop"`` and/or ``"keep"`` (may be empty).

    Returns:
        The tasks. Variants that would leave no column are skipped.
    """
    all_cols = np.arange(n_columns)
    tasks: list[Task] = []
    for si, sp in enumerate(splits):
        tasks.append((("baseline", "all", "all", n_columns, si), sp, all_cols))
    for lvl, g in groups.items():
        for name, pos in g.items():
            for mode in modes:
                cols = np.setdiff1d(all_cols, pos) if mode == "drop" else pos
                if len(cols) == 0:
                    continue  # dropping everything is meaningless
                for si, sp in enumerate(splits):
                    tasks.append(((mode, lvl, name, len(pos), si), sp, cols))
    return tasks


def summarise(raw: pd.DataFrame, splits: list[fc.Split]) -> pd.DataFrame:
    """Compare every one-at-a-time variant with the baseline, split by split.

    Args:
        raw: Long table with columns ``mode, level, group, n_features, split, metric, value``.
        splits: The splits (for the training/evaluation sizes used by the corrected test).

    Returns:
        One row per ``(mode, level, group, metric)`` with the mean/std of the score,
        the mean/std of the paired difference to the baseline (``delta``, negative
        means the variant is worse), its relative size, a corrected p-value and the
        delta per feature of the group. Cumulative rows are not included here.
    """
    n_train = float(np.mean([len(s.train) for s in splits]))
    n_test = float(np.mean([len(s.test) for s in splits]))
    base = raw[raw["mode"] == "baseline"].set_index(["metric", "split"])["value"]

    rows: list[dict[str, Any]] = []
    for (mode, lvl, name, metric), g in raw[raw["mode"].isin(["drop", "keep"])].groupby(
            ["mode", "level", "group", "metric"]):
        v = g["value"].to_numpy()
        b = base.loc[str(metric)].loc[g["split"]].to_numpy()
        d_mean, d_std, p = fc.paired_summary(v - b, n_train, n_test)
        size = int(g["n_features"].iloc[0])
        rows.append(dict(mode=mode, level=lvl, group=name, metric=metric, n_features=size,
                         score_mean=v.mean(), score_std=v.std(ddof=1) if len(v) > 1 else np.nan,
                         baseline_mean=b.mean(), delta_mean=d_mean, delta_std=d_std,
                         delta_pct_of_baseline=100 * d_mean / b.mean(), p_value_corrected=p,
                         delta_per_feature=d_mean / size))
    return pd.DataFrame(rows)


def report(summary: pd.DataFrame, baseline: pd.Series, n_classes: int, n_columns: int,
           modes: list[str], levels: list[str], cfg: fc.Config, out: Path) -> None:
    """Print the ranked results of each mode and level, and save the bar plots.

    Args:
        summary: Output of :func:`summarise`.
        baseline: Baseline scores of the primary metric, one per split.
        n_classes: Number of genres (for the chance line).
        n_columns: Number of columns of the baseline model.
        modes: Modes that were run.
        levels: Group levels that were run.
        cfg: The configuration (``evaluation.primary_metric`` and ``output``).
        out: Output directory.
    """
    primary = cfg["evaluation"]["primary_metric"]
    print(f"\n=== Baseline ({n_columns} features), {primary}: "
          f"{baseline.mean():.4f} +/- {baseline.std(ddof=1):.4f}")
    chance = 1.0 / n_classes
    show = ["group", "n_features", "score_mean", "delta_mean", "delta_std", "p_value_corrected"]
    for mode in modes:
        for lvl in levels:
            t = summary[(summary["mode"] == mode) & (summary["level"] == lvl)
                        & (summary["metric"] == primary)]
            if t.empty:
                continue
            if mode == "drop":   # most harmful removal first
                t = t.sort_values("delta_mean")
            else:                # most informative alone first
                t = t.sort_values("score_mean", ascending=False)
            print(f"\n--- {mode} | {lvl} | {primary} ---")
            print(t[show].head(cfg["output"].get("top_n_console", 15)).to_string(
                index=False, float_format=lambda x: f"{x:.4f}"))
            if cfg["output"].get("plots", True):
                top = cfg["output"].get("top_n_plot", 40)
                if mode == "drop":
                    fc.barh_plot(t, "group", "delta_mean", "delta_std",
                                 f"Drop-group ablation ({lvl}) - change in {primary}",
                                 out / f"ablation_drop_{lvl}_{primary}.png",
                                 f"{primary} change vs baseline (negative = group mattered)", top_n=top)
                else:
                    vl = [(baseline.mean(), "all features", "-")]
                    if primary in ("accuracy", "balanced_accuracy"):
                        vl.append((chance, "chance", ":"))
                    fc.barh_plot(t, "group", "score_mean", "score_std",
                                 f"Keep-only-group ablation ({lvl}) - {primary}",
                                 out / f"ablation_keep_{lvl}_{primary}.png", primary, vlines=vl, top_n=top)


# --------------------------------------------------------------------------
# Cumulative ablation
# --------------------------------------------------------------------------
@dataclass
class CumStep:
    """One step of the cumulative (successive) ablation.

    Attributes:
        label: Unique, sortable label (``"01_<name>"``) used as group name in the raw table.
        name: Human-readable name of what this step removes.
        removed_now: Number of columns removed by this step.
        removed_total: Number of columns removed since the full set, this step included.
        remaining: Positions of the columns still used after this step.
    """

    label: str
    name: str
    removed_now: int
    removed_total: int
    remaining: np.ndarray


def removal_mask(columns: pd.MultiIndex, cfg: fc.Config, sel: dict[str, Any]) -> np.ndarray:
    """Boolean mask of the columns matched by one removal selector.

    The selector accepts the usual keys ``feature``, ``statistics``, ``number``,
    ``exclude`` and ``columns_csv`` (see :func:`fma_common.selector_mask`), plus
    ``family`` (a family name or list of names from ``cfg["families"]``). All the
    given keys must match.

    Args:
        columns: Column MultiIndex of the active feature matrix.
        cfg: The configuration (``families`` block).
        sel: The selector.

    Returns:
        Boolean array of length ``len(columns)``.

    Raises:
        ValueError: On an empty selector, an unknown family, or (typo guard) a
            feature/statistic/number that is not in the active feature set.
    """
    if not isinstance(sel, dict):
        raise ValueError(f"A removal selector must be a mapping such as {{feature: rms}}, got {sel!r}")
    sel = dict(sel)
    fam = sel.pop("family", None)
    if fam is None and not sel:
        raise ValueError("Empty removal selector")
    mask = np.ones(len(columns), dtype=bool)
    if fam is not None:
        families = cfg.get("families") or {}
        feats: list[str] = []
        for name in fc.as_list(fam):
            if name not in families:
                raise ValueError(f"Unknown family '{name}'. Available: {list(families)}")
            feats += list(families[name] or [])
        mask &= np.asarray(columns.get_level_values("feature").isin(feats))
    if sel:
        mask &= fc.selector_mask(columns, sel, strict=True)
    return mask


def _describe(sel: dict[str, Any]) -> str:
    """Short text for a selector, used as default step name."""
    return ", ".join(f"{k}={'/'.join(map(str, fc.as_list(v)))}" for k, v in sel.items())


def resolve_cumulative(columns: pd.MultiIndex, cfg: fc.Config,
                       steps_cfg: list[dict[str, Any]]) -> list[CumStep]:
    """Turn the ``ablation.cumulative`` config into successive column sets.

    Step k removes everything removed by steps 1..k from the full feature set.

    Args:
        columns: Column MultiIndex of the active feature matrix.
        cfg: The configuration.
        steps_cfg: List of steps, each ``{name: optional text, remove: selector or list of
            selectors}`` (a list means the union of the selectors).

    Returns:
        The steps, in order (the full set itself is the baseline, not a step).

    Raises:
        ValueError: If a step has no ``remove`` entry, removes no column that is still
            present (typo or duplicate), or if the steps would remove every column.
    """
    n = len(columns)
    removed = np.zeros(n, dtype=bool)
    steps: list[CumStep] = []
    for i, step in enumerate(steps_cfg, 1):
        if "remove" not in step:
            raise ValueError(f"Cumulative step {i} has no 'remove' entry: {step}")
        sels = fc.as_list(step["remove"])
        mask = np.zeros(n, dtype=bool)
        for sel in sels:
            mask |= removal_mask(columns, cfg, sel)
        name = str(step.get("name") or " + ".join(_describe(s) for s in sels))
        new = mask & ~removed
        if not new.any():
            raise ValueError(f"Cumulative step {i} ('{name}') removes no column that is still "
                             f"present: check the names, or whether an earlier step already removed it.")
        removed |= mask
        if removed.all():
            raise ValueError(f"Cumulative steps up to {i} ('{name}') would remove every column")
        steps.append(CumStep(label=f"{i:02d}_{name}", name=name, removed_now=int(new.sum()),
                             removed_total=int(removed.sum()), remaining=np.flatnonzero(~removed)))
    return steps


def build_cumulative_tasks(steps: list[CumStep], splits: list[fc.Split]) -> list[Task]:
    """List the model fits of the cumulative ablation (one per step and split).

    Args:
        steps: Steps from :func:`resolve_cumulative`.
        splits: Train/evaluation splits.

    Returns:
        The tasks (the baseline fits are created by :func:`build_tasks`).
    """
    return [(("cumulative", "cumulative", st.label, st.removed_total, si), sp, st.remaining)
            for st in steps for si, sp in enumerate(splits)]


def summarise_cumulative(raw: pd.DataFrame, steps: list[CumStep], splits: list[fc.Split],
                         n_columns: int) -> pd.DataFrame:
    """Compare each cumulative step with the full set and with the previous step.

    Args:
        raw: Long table of all results (needs the baseline and cumulative rows).
        steps: Steps from :func:`resolve_cumulative`.
        splits: The splits (for the sizes used by the corrected test).
        n_columns: Number of columns of the full feature set.

    Returns:
        One row per ``(metric, step)``, with step 0 being the full set: columns removed
        and remaining, score mean/std over splits, and the paired differences
        ``delta_vs_baseline`` and ``delta_vs_previous`` (mean, std, corrected p-value;
        negative = worse).
    """
    n_train = float(np.mean([len(s.train) for s in splits]))
    n_test = float(np.mean([len(s.test) for s in splits]))
    base_all = raw[raw["mode"] == "baseline"].set_index(["metric", "split"])["value"]
    cum = raw[raw["mode"] == "cumulative"]

    rows: list[dict[str, Any]] = []
    for metric in sorted(cum["metric"].unique()):
        b = base_all.loc[str(metric)].sort_index()
        nan = float("nan")
        rows.append(dict(metric=metric, step=0, removed="(full feature set)", cols_removed_step=0,
                         cols_removed_total=0, n_remaining=n_columns, score_mean=b.mean(),
                         score_std=b.std(ddof=1) if len(b) > 1 else nan,
                         delta_vs_baseline_mean=0.0, delta_vs_baseline_std=nan, p_vs_baseline=nan,
                         delta_vs_previous_mean=nan, delta_vs_previous_std=nan, p_vs_previous=nan))
        prev = b
        for i, st in enumerate(steps, 1):
            v = cum[(cum["metric"] == metric) & (cum["group"] == st.label)].set_index(
                "split")["value"].sort_index()
            db_mean, db_std, db_p = fc.paired_summary((v - b).to_numpy(), n_train, n_test)
            dp_mean, dp_std, dp_p = fc.paired_summary((v - prev).to_numpy(), n_train, n_test)
            rows.append(dict(metric=metric, step=i, removed=st.name, cols_removed_step=st.removed_now,
                             cols_removed_total=st.removed_total,
                             n_remaining=n_columns - st.removed_total, score_mean=v.mean(),
                             score_std=v.std(ddof=1) if len(v) > 1 else nan,
                             delta_vs_baseline_mean=db_mean, delta_vs_baseline_std=db_std,
                             p_vs_baseline=db_p, delta_vs_previous_mean=dp_mean,
                             delta_vs_previous_std=dp_std, p_vs_previous=dp_p))
            prev = v
    return pd.DataFrame(rows)


def plot_cumulative(table: pd.DataFrame, metric: str, path: Path) -> None:
    """Save the cumulative ablation plot: score per step, and paired change vs the full set.

    The top panel shows the score after each step (mean +/- std over splits); that
    spread mostly reflects how hard each fold is, and it is shared by all steps. The
    bottom panel shows the paired difference to the full set (mean +/- std of the
    per-split differences), which is the quantity to judge.

    Args:
        table: Rows of :func:`summarise_cumulative` for one metric.
        metric: Metric name, for the axis labels.
        path: Output image path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True, figsize=(max(6.0, 0.95 * len(table) + 2), 8),
                                  gridspec_kw=dict(height_ratios=[2, 1.3]))
    x = table["step"].to_numpy()
    ax.errorbar(x, table["score_mean"], yerr=table["score_std"].fillna(0), marker="o", capsize=3)
    ax.axhline(float(table["score_mean"].iloc[0]), color="k", linestyle="--", linewidth=1,
               label="full feature set")
    ax.set_ylabel(f"{metric}\n(mean +/- std over splits)")
    ax.set_title("Cumulative ablation: each step removes one more group")
    ax.legend()
    ax.grid(alpha=0.3)

    ax2.errorbar(x, table["delta_vs_baseline_mean"], yerr=table["delta_vs_baseline_std"].fillna(0),
                 marker="o", capsize=3, color="#C44E52")
    ax2.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax2.set_ylabel(f"change in {metric} vs full set\n(paired, mean +/- std)")
    ax2.grid(alpha=0.3)
    labels = [("full" if r.step == 0 else f"- {r.removed}") + f"\n({r.n_remaining} cols)"
              for r in table.itertuples()]
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report_cumulative(table: pd.DataFrame, cfg: fc.Config, out: Path) -> None:
    """Print the cumulative table for the primary metric and save its plot.

    Args:
        table: Output of :func:`summarise_cumulative`.
        cfg: The configuration (``evaluation.primary_metric`` and ``output``).
        out: Output directory.
    """
    primary = cfg["evaluation"]["primary_metric"]
    t = table[table["metric"] == primary]
    show = ["step", "removed", "n_remaining", "score_mean", "delta_vs_baseline_mean", "p_vs_baseline",
            "delta_vs_previous_mean", "p_vs_previous"]
    print(f"\n--- cumulative ablation | {primary} (delta < 0 = worse; p-values are approximate) ---")
    print(t[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if cfg["output"].get("plots", True):
        plot_cumulative(t, primary, out / f"ablation_cumulative_{primary}.png")


def steps_table(columns: pd.MultiIndex, steps: list[CumStep], n_columns: int) -> pd.DataFrame:
    """List the columns removed at each cumulative step, for traceability.

    Args:
        columns: Column MultiIndex of the active feature matrix.
        steps: Steps from :func:`resolve_cumulative`.
        n_columns: Number of columns of the full feature set.

    Returns:
        A frame with columns ``step``, ``label`` and ``column`` (``feature/statistic/number``).
    """
    rows = []
    prev = np.arange(n_columns)
    for i, st in enumerate(steps, 1):
        for p in np.setdiff1d(prev, st.remaining):
            rows.append((i, st.label, "/".join(columns[p])))
        prev = st.remaining
    return pd.DataFrame(rows, columns=["step", "label", "column"])


# --------------------------------------------------------------------------
# Named configurations (arbitrary removal sets, compared pairwise)
# --------------------------------------------------------------------------
@dataclass
class Configuration:
    """One named configuration: the full feature set minus a set of removed columns.

    Attributes:
        label: Unique, sortable label (``"01_<name>"``) used as group name in the raw table.
        name: Name given in the config.
        removed_total: Number of columns removed from the full set.
        remaining: Positions of the columns used by this configuration.
    """

    label: str
    name: str
    removed_total: int
    remaining: np.ndarray


def _item_mask(columns: pd.MultiIndex, cfg: fc.Config, item: Any, removals: dict[str, Any]) -> np.ndarray:
    """Mask of one item of a ``remove`` list: a name from ``ablation.removals`` or an inline selector.

    Args:
        columns: Column MultiIndex of the active feature matrix.
        cfg: The configuration (``families`` block).
        item: A string (name defined in ``removals``) or a selector mapping.
        removals: The ``ablation.removals`` block (name -> selector or list of selectors).

    Returns:
        Boolean array of length ``len(columns)``.

    Raises:
        ValueError: If a name is not defined in ``removals`` or a selector is invalid.
    """
    if isinstance(item, str):
        if item not in removals:
            raise ValueError(f"Unknown removal '{item}'. Defined in ablation.removals: {list(removals)}")
        mask = np.zeros(len(columns), dtype=bool)
        for sel in fc.as_list(removals[item]):
            mask |= removal_mask(columns, cfg, sel)
        return mask
    return removal_mask(columns, cfg, item)


def resolve_configurations(columns: pd.MultiIndex, cfg: fc.Config, configs_cfg: list[dict[str, Any]],
                           removals: dict[str, Any]) -> list[Configuration]:
    """Turn the ``ablation.configurations`` config into column sets.

    Each configuration is independent: it is the full feature set minus the union
    of the columns matched by its own ``remove`` list.

    Args:
        columns: Column MultiIndex of the active feature matrix.
        cfg: The configuration.
        configs_cfg: List of ``{name: optional text, remove: item or list of items}``, where an
            item is a name defined in ``removals`` or an inline selector.
        removals: The ``ablation.removals`` block (name -> selector or list of selectors).

    Returns:
        The configurations, in order (the full set itself is the baseline, run automatically).

    Raises:
        ValueError: If a configuration has no ``remove`` entry, is named ``full`` or has a
            duplicated name, removes no column or every column, or uses an unknown name,
            family or feature.
    """
    n = len(columns)
    out: list[Configuration] = []
    names: set[str] = set()
    removed_sets: dict[bytes, str] = {}
    for i, conf in enumerate(configs_cfg, 1):
        if "remove" not in conf:
            raise ValueError(f"Configuration {i} has no 'remove' entry: {conf}")
        items = fc.as_list(conf["remove"])
        mask = np.zeros(n, dtype=bool)
        for item in items:
            mask |= _item_mask(columns, cfg, item, removals)
        name = str(conf.get("name") or " + ".join(it if isinstance(it, str) else _describe(it) for it in items))
        if name == "full":
            raise ValueError("'full' is the baseline and is always run automatically: "
                             "give this configuration another name.")
        if name in names:
            raise ValueError(f"Duplicate configuration name '{name}'")
        if not mask.any():
            raise ValueError(f"Configuration '{name}' removes no column from the active feature set "
                             f"(check the names).")
        if mask.all():
            raise ValueError(f"Configuration '{name}' would remove every column")
        key = mask.tobytes()
        if key in removed_sets:
            warnings.warn(f"Configurations '{removed_sets[key]}' and '{name}' remove exactly the "
                          f"same columns; one of them is redundant.")
        names.add(name)
        removed_sets[key] = name
        out.append(Configuration(label=f"{i:02d}_{name}", name=name, removed_total=int(mask.sum()),
                                 remaining=np.flatnonzero(~mask)))
    return out


def build_configuration_tasks(configs: list[Configuration], splits: list[fc.Split]) -> list[Task]:
    """List the model fits of the configurations (one per configuration and split).

    Args:
        configs: Configurations from :func:`resolve_configurations`.
        splits: Train/evaluation splits.

    Returns:
        The tasks (the baseline fits are created by :func:`build_tasks`).
    """
    return [(("configuration", "configuration", c.label, c.removed_total, si), sp, c.remaining)
            for c in configs for si, sp in enumerate(splits)]


def summarise_configurations(raw: pd.DataFrame, configs: list[Configuration], splits: list[fc.Split],
                             n_columns: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare configurations with the full set and with each other, split by split.

    Args:
        raw: Long table of all results (needs the baseline and configuration rows).
        configs: Configurations from :func:`resolve_configurations`.
        splits: The splits (for the sizes used by the corrected test).
        n_columns: Number of columns of the full feature set.

    Returns:
        ``(table, pairwise)``. ``table`` has one row per ``(metric, configuration)``, the
        full set included: columns removed and remaining, score mean/std over splits and the
        paired difference to the full set (mean, std, corrected p-value; negative = worse).
        ``pairwise`` has one row per ``(metric, pair of configurations)``, the full set
        included: ``delta_a_minus_b`` is the paired score of ``config_a`` minus that of ``config_b``.
    """
    n_train = float(np.mean([len(s.train) for s in splits]))
    n_test = float(np.mean([len(s.test) for s in splits]))
    base_all = raw[raw["mode"] == "baseline"].set_index(["metric", "split"])["value"]
    conf = raw[raw["mode"] == "configuration"]
    nan = float("nan")

    table_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for metric in sorted(conf["metric"].unique()):
        series: dict[str, pd.Series] = {"full": base_all.loc[str(metric)].sort_index()}
        removed = {"full": 0}
        for c in configs:
            series[c.name] = conf[(conf["metric"] == metric) & (conf["group"] == c.label)].set_index(
                "split")["value"].sort_index()
            removed[c.name] = c.removed_total
        for name, v in series.items():
            if name == "full":
                d_mean, d_std, p = 0.0, nan, nan
            else:
                d_mean, d_std, p = fc.paired_summary((v - series["full"]).to_numpy(), n_train, n_test)
            table_rows.append(dict(metric=metric, config=name, cols_removed=removed[name],
                                   n_remaining=n_columns - removed[name], score_mean=v.mean(),
                                   score_std=v.std(ddof=1) if len(v) > 1 else nan,
                                   delta_vs_full_mean=d_mean, delta_vs_full_std=d_std, p_vs_full=p))
        for a, b in combinations(series, 2):
            d_mean, d_std, p = fc.paired_summary((series[a] - series[b]).to_numpy(), n_train, n_test)
            pair_rows.append(dict(metric=metric, config_a=a, config_b=b, delta_a_minus_b_mean=d_mean,
                                  delta_a_minus_b_std=d_std, p_value_corrected=p))
    return pd.DataFrame(table_rows), pd.DataFrame(pair_rows)


def report_configurations(table: pd.DataFrame, pairwise: pd.DataFrame, cfg: fc.Config, out: Path) -> None:
    """Print the tables of the configurations for the primary metric and save the plot.

    Args:
        table: First output of :func:`summarise_configurations`.
        pairwise: Second output of :func:`summarise_configurations`.
        cfg: The configuration (``evaluation.primary_metric`` and ``output``).
        out: Output directory.
    """
    primary = cfg["evaluation"]["primary_metric"]
    t = table[table["metric"] == primary]
    print(f"\n--- configurations vs the full set | {primary} (delta < 0 = worse; p-values approximate) ---")
    print(t.drop(columns="metric").to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    pw = pairwise[pairwise["metric"] == primary].drop(columns="metric")
    print(f"\n--- all pairs of configurations | {primary} (delta = config_a minus config_b) ---")
    print(pw.head(60).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if len(pw) > 60:
        print(f"... {len(pw) - 60} more pairs in ablation_configurations_pairwise.csv")
    if cfg["output"].get("plots", True):
        p = t[t["config"] != "full"].copy()
        p["label"] = p["config"] + " (" + p["n_remaining"].astype(str) + " cols)"
        fc.barh_plot(p, "label", "delta_vs_full_mean", "delta_vs_full_std",
                     f"Configurations vs full feature set - change in {primary}",
                     out / f"ablation_configurations_{primary}.png",
                     f"{primary} change vs full set (negative = worse; paired mean +/- std)",
                     top_n=len(p))


def configurations_table(columns: pd.MultiIndex, configs: list[Configuration], n_columns: int) -> pd.DataFrame:
    """List the columns removed by each configuration, for traceability.

    Args:
        columns: Column MultiIndex of the active feature matrix.
        configs: Configurations from :func:`resolve_configurations`.
        n_columns: Number of columns of the full feature set.

    Returns:
        A frame with columns ``config`` and ``column`` (``feature/statistic/number``).
    """
    rows = [(c.name, "/".join(columns[p]))
            for c in configs for p in np.setdiff1d(np.arange(n_columns), c.remaining)]
    return pd.DataFrame(rows, columns=["config", "column"])


# --------------------------------------------------------------------------
def main() -> None:
    """Run the ablation described in the configuration and save the results."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--feature-set", default=None, help="override cfg['feature_set']")
    ap.add_argument("--output-dir", default=None, help="override cfg['paths']['output_dir']")
    args = ap.parse_args()

    cfg = fc.load_config(args.config, args.feature_set, args.output_dir)
    ev, acfg = cfg["evaluation"], cfg["ablation"]
    metrics, primary = ev["metrics"], ev["primary_metric"]
    seed = ev.get("seed", 42)
    raw_modes = acfg.get("modes", ["drop", "keep"])
    modes = fc.as_list(raw_modes) if raw_modes else []    # `modes: []` skips the one-at-a-time part
    steps_cfg = acfg.get("cumulative") or []
    configs_cfg = acfg.get("configurations") or []
    if not modes and not steps_cfg and not configs_cfg:
        raise SystemExit("Nothing to do: set ablation.modes, ablation.cumulative and/or "
                         "ablation.configurations in the config.")
    out = fc.output_path(cfg, "ablation")

    ds = fc.load_dataset(cfg)
    n_columns = ds.X.shape[1]
    splits = fc.make_splits(cfg, ds)

    groups: fc.Groups = {}
    if modes:
        groups = fc.resolve_groups(ds.columns, cfg, acfg.get("groups", cfg["groups"]))
        if not groups:
            raise SystemExit("No groups to ablate: check the 'groups' section of the config.")
        for lvl, g in groups.items():
            print(f"[groups] {lvl}: {len(g)} group(s)")
    cum_steps = resolve_cumulative(ds.columns, cfg, steps_cfg) if steps_cfg else []
    for i, st in enumerate(cum_steps, 1):
        print(f"[cumulative] step {i}: remove '{st.name}' ({st.removed_now} columns) "
              f"-> {n_columns - st.removed_total} columns left")

    configs = (resolve_configurations(ds.columns, cfg, configs_cfg, acfg.get("removals") or {})
               if configs_cfg else [])
    for c in configs:
        print(f"[configuration] '{c.name}': remove {c.removed_total} columns "
              f"-> {n_columns - c.removed_total} columns left")

    tasks = (build_tasks(n_columns, groups, splits, modes) + build_cumulative_tasks(cum_steps, splits)
             + build_configuration_tasks(configs, splits))
    print(f"[run] {len(tasks)} model fits, n_jobs={cfg['runtime']['n_jobs']}")
    gen = Parallel(n_jobs=cfg["runtime"]["n_jobs"], return_as="generator_unordered")(
        delayed(fit_eval)(ds.X, ds.y, sp, cols, cfg, seed, metrics, key) for key, sp, cols in tasks)
    rows = []
    for (mode, lvl, name, size, si), res in tqdm(gen, total=len(tasks)):
        for m, v in res.items():
            rows.append(dict(mode=mode, level=lvl, group=name, n_features=size,
                             split=si, metric=m, value=v))
    raw = pd.DataFrame(rows).sort_values(["mode", "level", "group", "metric", "split"])
    raw.to_csv(out / "ablation_raw.csv", index=False)

    if modes:
        summary = summarise(raw, splits)
        summary.to_csv(out / "ablation_summary.csv", index=False)
        fc.groups_table(ds.columns, groups).to_csv(out / "groups.csv", index=False)
    with open(out / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    baseline = raw[raw["mode"] == "baseline"].set_index(["metric", "split"])["value"].loc[primary]
    if modes:
        report(summary, baseline, len(ds.classes), n_columns, modes, list(groups), cfg, out)
    else:
        print(f"\n=== Baseline ({n_columns} features), {primary}: "
              f"{baseline.mean():.4f} +/- {baseline.std(ddof=1):.4f}")
    if cum_steps:
        cum_table = summarise_cumulative(raw, cum_steps, splits, n_columns)
        cum_table.to_csv(out / "ablation_cumulative.csv", index=False)
        steps_table(ds.columns, cum_steps, n_columns).to_csv(out / "cumulative_steps.csv", index=False)
        report_cumulative(cum_table, cfg, out)
    if configs:
        conf_table, pairwise = summarise_configurations(raw, configs, splits, n_columns)
        conf_table.to_csv(out / "ablation_configurations.csv", index=False)
        pairwise.to_csv(out / "ablation_configurations_pairwise.csv", index=False)
        configurations_table(ds.columns, configs, n_columns).to_csv(
            out / "configurations_removed_columns.csv", index=False)
        report_configurations(conf_table, pairwise, cfg, out)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
