"""Static PNG report for model_artifact_ring/ (train_ring_model.py) -- the
reduced stress/sleep_duration/sleep_quality/fatigue + day_of_week model
consumed by colmi_r02_client/injury_predict.py, NOT the full model in
model_artifact/ (see reports/ for that one).

Produces, into reports_ring/:
  01_pr_curve.png             Precision-recall curve over concatenated
                               out-of-fold predictions (GroupKFold, 5 folds).
  02_confusion_matrix.png     Aggregate confusion matrix (summed across folds)
                               at each fold's tuned threshold.
  03_feature_importance.png   Top-20 XGBoost feature importances from the
                               exported (deployed) ring pipeline.
  04_calibration.png          Quantile-binned reliability curve, pooled
                               out-of-fold predictions.

Same palette as scripts/generate_report_plots.py (dataviz skill reference
palette), so the two reports read as one system in the README.

Usage:
    python scripts/generate_ring_report_plots.py
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

from src.config import load_config
from src.cv.splitters import iter_splits, train_val_split_for_threshold
from src.eval.metrics import compute_metrics, tune_threshold
from src.eval.report import aggregate_fold_metrics
from src.models.pipeline import build_flat_pipeline
from src.windowing.windows import build_windows
from scripts.train_ring_model import RING_WELLNESS_FEATURES, build_ring_daily_df
from src.config import ImbalanceConfig, ModelConfig, WindowConfig

REPO = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO / "reports_ring"
ARTIFACT_DIR = REPO / "model_artifact_ring"
REPORTS_DIR.mkdir(exist_ok=True)

# ---- palette (same as scripts/generate_report_plots.py) ----
BLUE = "#2a78d6"
BLUE_LIGHT_100 = "#cde2fb"
GRAY_DEEMPH = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

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


def _blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blue_seq", [BLUE_LIGHT_100, BLUE])


def run_ring_cv(daily_df, window_cfg, cv_cfg, model_cfg, imbalance_cfg):
    """Pooled out-of-fold (y_true, y_prob) + per-fold metrics + aggregate
    confusion matrix for the ring feature set, GroupKFold by player."""
    windows = build_windows(daily_df, window_cfg)
    X, y, groups = windows.X_flat.to_numpy(), windows.y, windows.groups

    fold_metrics, all_y_true, all_y_prob = [], [], []
    for train_idx, test_idx in iter_splits(X, y, groups, cv_cfg):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(set(y_train)) < 2:
            continue
        X_train, X_test = X[train_idx], X[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        threshold = 0.5
        try:
            inner_tr, inner_val = train_val_split_for_threshold(X_train, y_train, groups_train, cv_cfg)
            if len(set(y_train[inner_tr])) >= 2:
                pipe = build_flat_pipeline(model_cfg, imbalance_cfg, cv_cfg.random_seed, y_train[inner_tr])
                pipe.fit(X_train[inner_tr], y_train[inner_tr])
                prob_val = pipe.predict_proba(X_train[inner_val])[:, 1]
                threshold = tune_threshold(y_train[inner_val], prob_val, "f1")
        except ValueError:
            pass

        pipe = build_flat_pipeline(model_cfg, imbalance_cfg, cv_cfg.random_seed, y_train)
        pipe.fit(X_train, y_train)
        prob_test = pipe.predict_proba(X_test)[:, 1]

        fold_metrics.append(compute_metrics(y_test, prob_test, groups_test, threshold))
        all_y_true.append(y_test)
        all_y_prob.append(prob_test)

    return fold_metrics, np.concatenate(all_y_true), np.concatenate(all_y_prob)


def plot_pr_curve(y_true, y_prob, ap_pooled):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot(recall, precision, color=BLUE, linewidth=2)
    ax.axhline(baseline, color=GRAY_DEEMPH, linewidth=1.5, linestyle=(0, (1, 0)))
    ax.text(0.98, baseline, f"no-skill baseline ({baseline:.4f})", ha="right", va="bottom",
            fontsize=9, color=TEXT_MUTED, transform=ax.get_yaxis_transform())

    ax.set_xlim(0, 1); ax.set_ylim(0, max(precision.max(), baseline) * 1.15)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-recall curve — ring feature set\n"
                 f"Pooled average precision = {ap_pooled:.4f} (concatenated out-of-fold predictions)",
                 fontsize=12, color=TEXT_PRIMARY, pad=12)
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "01_pr_curve.png", dpi=160)
    plt.close(fig)


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

    ax.set_title(f"Confusion matrix — ring feature set\n(sum over 5 folds; per-fold mean PR-AUC {pr_auc:.4f})",
                 fontsize=11.5, color=TEXT_PRIMARY, pad=12)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "02_confusion_matrix.png", dpi=160)
    plt.close(fig)


def plot_feature_importance(pipeline, flat_columns, top_n=20):
    importances = pipeline.named_steps["classifier"].feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    names = [flat_columns[i] for i in order][::-1]
    values = [importances[i] for i in order][::-1]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(names)), values, color=BLUE, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("XGBoost feature importance (gain-based)")
    ax.set_title(f"Top {top_n} features — ring model (54 total columns)", fontsize=12, color=TEXT_PRIMARY, pad=12)
    style_ax(ax, grid_axis="x")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "03_feature_importance.png", dpi=160)
    plt.close(fig)


def plot_calibration(y_true, y_prob):
    brier = brier_score_loss(y_true, y_prob)
    n_bins = 8
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    lims = [0, max(mean_pred.max(), frac_pos.max()) * 1.15]
    ax.plot(lims, lims, color=GRAY_DEEMPH, linewidth=1.2, linestyle=(0, (4, 3)), label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", color=BLUE, linewidth=2, markersize=5, label="Ring model")

    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Mean predicted probability (per quantile bin)")
    ax.set_ylabel("Observed injury frequency (per quantile bin)")
    ax.set_title(f"Calibration — ring feature set\nBrier score = {brier:.5f} "
                 f"(pooled out-of-fold predictions, {n_bins} quantile bins)",
                 fontsize=12, color=TEXT_PRIMARY, pad=12)
    ax.text(0.02, 0.98,
            "Read with caution: ~100 positives spread over 8 bins --\n"
            "each point averages a handful of events, wide unshown uncertainty.",
            transform=ax.transAxes, fontsize=8, color=TEXT_MUTED, va="top", ha="left")
    style_ax(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "04_calibration.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    base_cfg = load_config("config/default.yaml")
    injury_path = Path(base_cfg.data.subjective_dir) / "injury" / "injury.csv"
    daily_df = build_ring_daily_df(base_cfg.data, base_cfg.target, injury_path)

    window_cfg = WindowConfig(input_sessions=7, output_days=3, max_span_days=14, stride=1)
    model_cfg = ModelConfig(name="xgboost")
    imbalance_cfg = ImbalanceConfig(strategy="none")
    base_cfg.cv.n_splits = 5

    print("Running GroupKFold CV on the ring feature set...")
    fold_metrics, y_true, y_prob = run_ring_cv(daily_df, window_cfg, base_cfg.cv, model_cfg, imbalance_cfg)
    agg = aggregate_fold_metrics(fold_metrics)
    ap_pooled = average_precision_score(y_true, y_prob)
    print(f"Per-fold mean PR-AUC={agg['pr_auc_mean']:.4f}  Pooled AP={ap_pooled:.4f}")

    plot_pr_curve(y_true, y_prob, ap_pooled)
    plot_confusion_matrix(agg["confusion_matrix"], agg["pr_auc_mean"])
    plot_calibration(y_true, y_prob)

    pipeline = joblib.load(ARTIFACT_DIR / "pipeline.joblib")
    manifest = json.loads((ARTIFACT_DIR / "manifest.json").read_text(encoding="utf-8"))
    plot_feature_importance(pipeline, manifest["flat_columns"])

    print(f"\nSaved 01-04 report PNGs to {REPORTS_DIR}")
