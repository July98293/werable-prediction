"""Static PNG report: model statistics + per-athlete risk trajectories.

Produces, into reports/:
  01_pr_auc_by_strategy.png   PR-AUC across model x imbalance-strategy, for the
                              deployed window config (input=3, output=7 days),
                              recomputed live via proper GroupKFold CV.
  02_feature_importance.png   Top-15 XGBoost feature importances from the
                              exported (deployed) pipeline.
  03_confusion_matrix.png     Aggregate confusion matrix (summed across folds)
                              for the deployed config, at each fold's tuned
                              threshold.
  04_pr_curve.png             Precision-recall curve over concatenated
                              out-of-fold predictions for the deployed config.
  05_risk_trajectory_*.png    Predicted risk over time for a few real players
                              (using the exported, all-data-fit pipeline),
                              with real reported-injury dates marked.

Colors/marks follow the dataviz skill's reference palette (references/palette.md):
sequential blue for magnitude, status red only for the injury-event marker.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

from src.config import load_config, with_overrides
from src.cv.splitters import iter_splits, train_val_split_for_threshold
from src.data import raw_loaders
from src.eval.metrics import compute_metrics, tune_threshold
from src.eval.report import aggregate_fold_metrics
from src.models.pipeline import build_flat_pipeline
from src.windowing.windows import build_windows
from scripts.run_experiment import build_daily_df, load_raw_frames

REPO = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO / "reports"
ARTIFACT_DIR = REPO / "model_artifact"
CACHE_PATH = REPORTS_DIR / "_cv_comparison_cache.npz"
REPORTS_DIR.mkdir(exist_ok=True)

# ---- palette (dataviz skill reference palette, references/palette.md) ----
BLUE = "#2a78d6"
BLUE_LIGHT_100 = "#cde2fb"
BLUE_LIGHT_250 = "#86b6ef"
GRAY_DEEMPH = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "text.color": TEXT_PRIMARY,
    "axes.labelcolor": TEXT_SECONDARY,
    "axes.edgecolor": GRIDLINE,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 11,
})


def style_ax(ax, grid_axis="y"):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.tick_params(length=0)
    ax.grid(axis=grid_axis, linewidth=0.8, color=GRIDLINE)
    ax.set_axisbelow(True)


# =====================================================================
# 1) PR-AUC by model x imbalance strategy (deployed window config)
# =====================================================================
def run_deployed_config_cv(base_cfg, raw_frames, injury_path, model_name, imbalance_strategy):
    """One (model, imbalance) combo at input=3/output=7: returns
    (fold_metrics, concatenated_y_true, concatenated_y_prob) over out-of-fold
    test predictions.
    """
    cfg = with_overrides(base_cfg, window__input_sessions=3, window__output_days=7,
                          model__name=model_name, imbalance__strategy=imbalance_strategy)
    daily_df = build_daily_df(raw_frames, injury_path, cfg.data, cfg.target)
    windows = build_windows(daily_df, cfg.window)
    X, y, groups = windows.X_flat.to_numpy(), windows.y, windows.groups

    fold_metrics, all_y_true, all_y_prob = [], [], []
    for train_idx, test_idx in iter_splits(X, y, groups, cfg.cv):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(set(y_train)) < 2:
            continue
        X_train, X_test = X[train_idx], X[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        threshold = 0.5
        try:
            inner_tr, inner_val = train_val_split_for_threshold(X_train, y_train, groups_train, cfg.cv)
            if len(set(y_train[inner_tr])) >= 2:
                pipe = build_flat_pipeline(cfg.model, cfg.imbalance, cfg.cv.random_seed, y_train[inner_tr])
                pipe.fit(X_train[inner_tr], y_train[inner_tr])
                prob_val = pipe.predict_proba(X_train[inner_val])[:, 1]
                threshold = tune_threshold(y_train[inner_val], prob_val, cfg.threshold_metric)
        except ValueError:
            pass

        pipe = build_flat_pipeline(cfg.model, cfg.imbalance, cfg.cv.random_seed, y_train)
        pipe.fit(X_train, y_train)
        prob_test = pipe.predict_proba(X_test)[:, 1]

        fold_metrics.append(compute_metrics(y_test, prob_test, groups_test, threshold))
        all_y_true.append(y_test)
        all_y_prob.append(prob_test)

    if not fold_metrics:
        return None, None, None
    return fold_metrics, np.concatenate(all_y_true), np.concatenate(all_y_prob)


def plot_pr_auc_by_strategy(rows, deployed_key):
    rows = sorted(rows, key=lambda r: r["pr_auc"], reverse=True)
    labels = [f"{r['model']} + {r['imbalance']}" for r in rows]
    values = [r["pr_auc"] for r in rows]
    colors = [BLUE if (r["model"], r["imbalance"]) == deployed_key else GRAY_DEEMPH for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(rows)), values, color=colors, width=0.62)
    for rect, r in zip(bars, rows):
        if (r["model"], r["imbalance"]) == deployed_key:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + max(values) * 0.02,
                    f"{r['pr_auc']:.4f} (deployed)", ha="center", va="bottom",
                    fontsize=9.5, color=TEXT_PRIMARY, fontweight="bold")

    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=9.5, rotation=30, ha="right")
    ax.set_xlim(-0.7, len(rows) - 0.3)
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_ylabel("PR-AUC (average precision)")
    ax.set_title("PR-AUC by model x imbalance strategy — input=3 sessions, output=7 days\n"
                  "GroupKFold(5), player-level holdout", fontsize=12, color=TEXT_PRIMARY, pad=12)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "01_pr_auc_by_strategy.png", dpi=160)
    plt.close(fig)


# =====================================================================
# 2) Feature importance (deployed pipeline)
# =====================================================================
def plot_feature_importance(pipeline, flat_columns, top_n=15):
    importances = pipeline.named_steps["classifier"].feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    names = [flat_columns[i] for i in order][::-1]
    values = [importances[i] for i in order][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(names)), values, color=BLUE, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel("XGBoost feature importance (gain-based)")
    ax.set_title("Top 15 features — deployed model", fontsize=12, color=TEXT_PRIMARY, pad=12)
    style_ax(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "02_feature_importance.png", dpi=160)
    plt.close(fig)


# =====================================================================
# 3) Confusion matrix (deployed config, aggregate across folds)
# =====================================================================
def plot_confusion_matrix(cm: np.ndarray, pr_auc: float):
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    cm = np.array(cm)
    vmax = cm.max()
    ax.imshow(cm, cmap=_blue_cmap(), vmin=0, vmax=vmax)

    labels = ["No injury", "Injury (next 7d)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / vmax if vmax else 0
            text_color = "#ffffff" if frac > 0.55 else TEXT_PRIMARY
            ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center",
                    fontsize=15, fontweight="bold", color=text_color)

    ax.set_title(f"Confusion matrix — deployed config\n(sum over 5 folds; per-fold mean PR-AUC {pr_auc:.4f})",
                 fontsize=11.5, color=TEXT_PRIMARY, pad=12)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "03_confusion_matrix.png", dpi=160)
    plt.close(fig)


def _blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_seq", [BLUE_LIGHT_100, BLUE])


# =====================================================================
# 4) PR curve (deployed config, out-of-fold concatenated predictions)
# =====================================================================
def plot_pr_curve(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(recall, precision, color=BLUE, linewidth=2)
    ax.axhline(baseline, color=GRAY_DEEMPH, linewidth=1.5, linestyle=(0, (1, 0)))
    ax.text(0.98, baseline, f"no-skill baseline ({baseline:.4f})", ha="right", va="bottom",
            fontsize=9, color=TEXT_MUTED, transform=ax.get_yaxis_transform())

    ax.set_xlim(0, 1); ax.set_ylim(0, max(precision.max(), baseline) * 1.15)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall curve — deployed config\n"
                 f"Pooled average precision = {ap:.4f} (concatenated out-of-fold predictions)",
                 fontsize=12, color=TEXT_PRIMARY, pad=12)
    ax.text(0.98, 0.55,
            "Note: this pooled AP differs from the bar chart's\n"
            "per-fold-mean AP (0.0381) — averaging small per-fold\n"
            "scores vs. one score over all folds' predictions\n"
            "combined are both valid, non-identical estimates.",
            transform=ax.transAxes, fontsize=8, color=TEXT_MUTED, va="center", ha="right")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "04_pr_curve.png", dpi=160)
    plt.close(fig)


# =====================================================================
# 5) Calibration (reliability curve, deployed config, pooled out-of-fold)
# =====================================================================
def plot_calibration(y_true, y_prob):
    """Quantile-binned reliability curve. Uniform-width bins (the sklearn
    default) are useless here: predicted probabilities cluster below ~0.05
    under this class imbalance, so a uniform grid puts nearly every point in
    the first bin. Quantile bins instead guarantee each bin has a comparable
    number of predictions.
    """
    brier = brier_score_loss(y_true, y_prob)
    n_bins = 8
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    lims = [0, max(mean_pred.max(), frac_pos.max()) * 1.15]
    ax.plot(lims, lims, color=GRAY_DEEMPH, linewidth=1.2, linestyle=(0, (4, 3)), label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", color=BLUE, linewidth=2, markersize=5,
            label="Deployed model")

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Mean predicted probability (per quantile bin)")
    ax.set_ylabel("Observed injury frequency (per quantile bin)")
    ax.set_title(f"Calibration — deployed config\nBrier score = {brier:.5f} "
                 f"(pooled out-of-fold predictions, {n_bins} quantile bins)",
                 fontsize=12, color=TEXT_PRIMARY, pad=12)
    ax.text(0.02, 0.98,
            "Read with caution: with ~100-200 positives spread over 8 bins,\n"
            "each point averages a handful of events -- this curve has wide,\n"
            "unshown uncertainty, not a precise per-bin estimate.",
            transform=ax.transAxes, fontsize=8, color=TEXT_MUTED, va="top", ha="left")
    style_ax(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "06_calibration.png", dpi=160)
    plt.close(fig)


# =====================================================================
# 5) Per-athlete risk trajectories
# =====================================================================
def plot_risk_trajectory(player_id, daily_df_player, pipeline, feature_cols, flat_columns,
                          window_cfg, injury_events, threshold, index):
    windows = build_windows(daily_df_player, window_cfg)
    X = windows.X_flat.reindex(columns=flat_columns).to_numpy()
    probs = pipeline.predict_proba(X)[:, 1]
    dates = windows.window_end_date

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, probs, color=BLUE, linewidth=2, label="Predicted injury risk")
    ax.axhline(threshold, color=GRAY_DEEMPH, linewidth=1.2, label=f"Operating threshold ({threshold:.4f})")

    player_events = injury_events[injury_events["player_id"] == player_id]["date"]
    for i, d in enumerate(player_events):
        ax.axvline(d, color=CRITICAL, linewidth=1.2, alpha=0.85,
                   label="Reported injury" if i == 0 else None)

    ax.set_yscale("log")
    ax.set_ylabel("Predicted risk (probability, log scale)")
    team, short_id = player_id.split("-")[0], player_id.split("-")[1][:8]
    ax.set_title(f"Risk trajectory — {team} player {short_id} "
                 f"({len(player_events)} reported injury dates)", fontsize=12, color=TEXT_PRIMARY, pad=12)
    style_ax(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncols=1)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"05_risk_trajectory_{index}_{team}_{short_id}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    force_recompute = "--force" in sys.argv
    deployed_key = ("xgboost", "none")

    base_cfg = load_config("config/default.yaml")
    raw_frames = load_raw_frames(base_cfg.data)
    injury_path = Path(base_cfg.data.subjective_dir) / "injury" / "injury.csv"

    if CACHE_PATH.exists() and not force_recompute:
        print(f"Loading cached CV comparison from {CACHE_PATH} (pass --force to recompute)...")
        cache = np.load(CACHE_PATH, allow_pickle=True)
        comparison_rows = cache["comparison_rows"].tolist()
        deployed_confusion = cache["deployed_confusion"]
        deployed_pr_auc = float(cache["deployed_pr_auc"])
        deployed_probs = (cache["deployed_y_true"], cache["deployed_y_prob"])
    else:
        print("Recomputing model x imbalance-strategy comparison via GroupKFold CV...")
        comparison_rows = []
        deployed_confusion, deployed_probs, deployed_pr_auc = None, None, None
        for imbalance in ("none", "class_weight", "random_oversample", "smote"):
            for model_name in ("logreg", "random_forest", "xgboost"):
                fold_metrics, y_true, y_prob = run_deployed_config_cv(
                    base_cfg, raw_frames, injury_path, model_name, imbalance)
                if fold_metrics is None:
                    print(f"  skip {model_name}+{imbalance}: no usable folds")
                    continue
                agg = aggregate_fold_metrics(fold_metrics)
                comparison_rows.append({"model": model_name, "imbalance": imbalance, "pr_auc": agg["pr_auc_mean"]})
                print(f"  {model_name:14s} + {imbalance:18s} PR-AUC={agg['pr_auc_mean']:.4f}")
                if (model_name, imbalance) == deployed_key:
                    deployed_confusion = agg["confusion_matrix"]
                    deployed_pr_auc = agg["pr_auc_mean"]
                    deployed_probs = (y_true, y_prob)

        np.savez(CACHE_PATH, comparison_rows=np.array(comparison_rows, dtype=object),
                 deployed_confusion=np.array(deployed_confusion), deployed_pr_auc=deployed_pr_auc,
                 deployed_y_true=deployed_probs[0], deployed_y_prob=deployed_probs[1])
        print(f"Cached CV comparison to {CACHE_PATH}")

    plot_pr_auc_by_strategy(comparison_rows, deployed_key)
    plot_confusion_matrix(deployed_confusion, deployed_pr_auc)
    plot_pr_curve(*deployed_probs)
    plot_calibration(*deployed_probs)
    print("Saved 01_pr_auc_by_strategy.png, 03_confusion_matrix.png, 04_pr_curve.png, 06_calibration.png")

    pipeline = joblib.load(ARTIFACT_DIR / "pipeline.joblib")
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    plot_feature_importance(pipeline, manifest["flat_columns"])
    print("Saved 02_feature_importance.png")

    daily_df = build_daily_df(raw_frames, injury_path, base_cfg.data, base_cfg.target)
    injury_events = raw_loaders.load_injury_events(injury_path, base_cfg.data.date_format, min_severity="minor")
    window_cfg = base_cfg.window
    window_cfg.input_sessions, window_cfg.output_days = (
        manifest["window"]["input_sessions"], manifest["window"]["output_days"])

    example_players = [
        "TeamA-4051bba7-1170-4c43-b912-8c38815a7625",  # 47 real injury reports (persistent knee issue)
        "TeamA-3e5f6e2b-46b7-4890-84a9-3bbb2649af5a",  # 28 real injury reports
        "TeamA-f20565cc-df22-46a7-aa97-af8ed00601d7",  # 0 reports, consistently low risk (contrast)
    ]
    for i, player_id in enumerate(example_players, start=1):
        player_df = daily_df[daily_df["player_id"] == player_id]
        plot_risk_trajectory(player_id, player_df, pipeline, manifest["feature_cols"],
                              manifest["flat_columns"], window_cfg, injury_events,
                              manifest["threshold"], i)
        print(f"Saved 05_risk_trajectory_{i}_*.png for {player_id}")

    print(f"\nAll PNGs written to {REPORTS_DIR}")
