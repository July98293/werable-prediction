"""Retrains the injury-risk model on a REDUCED feature set that a Colmi R02
ring can actually approximate, instead of the full SoccerMon feature set
used by train_and_export_model.py (which needs coach-logged training-load
diaries and full wellness surveys the ring has no way to produce).

Kept features (all still SUBJECTIVE-SCALE wellness columns from the real
SoccerMon dataset -- the ring only supplies them a proxy value at inference
time, see colmi_r02_client/injury_predict.py):
    - stress          (ring proxy: HR-based stress classifier, see stress_predict.py)
    - sleep_duration  (ring proxy: sleep.py, Big Data BLE sleep sync)
    - sleep_quality   (ring proxy: sleep.py, derived from sleep-stage composition)
    - fatigue         (ring proxy: daily movement/training volume from steps+
                        calories+distance -- NOT the same construct as subjective
                        fatigue, see injury_predict.py's docstring caveat)
    - day_of_week     (free: derived from the calendar, no sensor needed)

Explicitly DROPPED vs. the full model: readiness, soreness, mood (no
physiological proxy exists for these), and every training-load feature
(acwr, atl, ctl28, ctl42, daily_load, monotony, strain, rpe_mean, rpe_max,
n_sessions, duration_sum, weekly_load, is_matchday) -- those were computed
in training from coach RPE x duration session logs, which a consumer ring
cannot produce or approximate honestly.

Because this drops most of the signal the full model had, expect materially
different (likely worse, or differently-biased) validated metrics than
model_artifact/manifest.json. That is reported here, not hidden.

Usage:
    python scripts/train_ring_model.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ImbalanceConfig, ModelConfig, WindowConfig, load_config, with_overrides
from src.data import raw_loaders
from src.data import target as target_mod
from src.data import merge
from src.eval.metrics import compute_metrics, tune_threshold
from src.eval.report import aggregate_fold_metrics
from src.models.pipeline import build_flat_pipeline
from src.windowing.windows import build_windows
from scripts.run_experiment import run_cv_for_windows

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "model_artifact_ring"

RING_WELLNESS_FEATURES = ["fatigue", "sleep_duration", "sleep_quality", "stress"]


def build_ring_daily_df(data_cfg, target_cfg, injury_path: Path) -> pd.DataFrame:
    """Same grid/merge/label machinery as run_experiment.build_daily_df, but
    fed only the 4 ring-approximable wellness columns (+ their _missing
    indicators) instead of the full wellness + training-load + GPS feature
    set. day_of_week is added directly (it needs no data source).
    """
    wellness = raw_loaders.load_wellness(data_cfg.subjective_dir, data_cfg.date_format)
    keep_cols = ["player_id", "date"] + RING_WELLNESS_FEATURES + [f"{f}_missing" for f in RING_WELLNESS_FEATURES]
    wellness = wellness[keep_cols]

    grid = merge.build_full_grid([wellness])
    daily_df = merge.merge_features(grid, [wellness])
    daily_df["day_of_week"] = daily_df["date"].dt.dayofweek

    events = raw_loaders.load_injury_events(injury_path, data_cfg.date_format, target_cfg.min_severity)
    daily_label = target_mod.build_daily_label(events, target_cfg)
    daily_df = merge.attach_injury_label(daily_df, daily_label)
    return daily_df


if __name__ == "__main__":
    base_cfg = load_config("config/default.yaml")
    injury_path = Path(base_cfg.data.subjective_dir) / "injury" / "injury.csv"
    daily_df = build_ring_daily_df(base_cfg.data, base_cfg.target, injury_path)

    # "Last week -> next 3 days", matching the colmi_r02_client dashboard's
    # framing (colmi_r02_client/injury_predict.py): 7 days of ring history
    # predicting a 3-day-ahead injury window.
    window_cfg = WindowConfig(input_sessions=7, output_days=3, max_span_days=14, stride=1)
    model_cfg = ModelConfig(name="xgboost")
    imbalance_cfg = ImbalanceConfig(strategy="none")

    windows = build_windows(daily_df, window_cfg)
    feature_cols = windows.feature_cols
    flat_columns = list(windows.X_flat.columns)
    X, y, groups = windows.X_flat.to_numpy(), windows.y, windows.groups
    print(f"Built {len(y)} windows ({int(y.sum())} positive) across {len(set(groups))} players, "
          f"{len(feature_cols)} feature_cols -> {len(flat_columns)} flat columns")

    rng = np.random.default_rng(42)
    unique_players = np.array(sorted(set(groups)))
    rng.shuffle(unique_players)
    n_holdout = max(1, int(0.2 * len(unique_players)))
    holdout_players = set(unique_players[:n_holdout])
    holdout_mask = np.array([g in holdout_players for g in groups])

    threshold_pipe = build_flat_pipeline(model_cfg, imbalance_cfg, base_cfg.cv.random_seed, y[~holdout_mask],
                                          base_cfg.feature_selection, X[~holdout_mask].shape[1])
    threshold_pipe.fit(X[~holdout_mask], y[~holdout_mask])
    prob_holdout = threshold_pipe.predict_proba(X[holdout_mask])[:, 1]
    threshold = tune_threshold(y[holdout_mask], prob_holdout, "f1")
    print(f"Operating threshold picked on a {n_holdout}-player holdout slice: {threshold:.4f}")

    holdout_metrics = compute_metrics(y[holdout_mask], prob_holdout, groups[holdout_mask], threshold)
    print(f"Holdout metrics ({n_holdout} never-trained-on players): "
          f"PR-AUC={holdout_metrics['pr_auc']:.4f} precision={holdout_metrics['precision']:.4f} "
          f"recall={holdout_metrics['recall']:.4f} brier={holdout_metrics['brier_score']:.5f}")

    cv_cfg = with_overrides(base_cfg, window__input_sessions=window_cfg.input_sessions,
                             window__output_days=window_cfg.output_days,
                             model__name=model_cfg.name, imbalance__strategy=imbalance_cfg.strategy,
                             cv__n_repeats=5, name="ring_config_validated")
    cv_fold_metrics, cv_folds_skipped = run_cv_for_windows(windows, cv_cfg)
    validated_metrics = aggregate_fold_metrics(cv_fold_metrics)
    validated_metrics["n_folds_skipped"] = cv_folds_skipped
    print(f"Validated CV metrics ({validated_metrics['n_folds']} folds over "
          f"{validated_metrics['n_repeats']} repeats): PR-AUC={validated_metrics['pr_auc_mean']:.4f} "
          f"[{validated_metrics['pr_auc_ci_low']:.4f}, {validated_metrics['pr_auc_ci_high']:.4f}] "
          f"precision={validated_metrics['precision_mean']:.4f} recall={validated_metrics['recall_mean']:.4f} "
          f"brier={validated_metrics['brier_score_mean']:.5f}")

    final_pipe = build_flat_pipeline(model_cfg, imbalance_cfg, base_cfg.cv.random_seed, y,
                                      base_cfg.feature_selection, X.shape[1])
    final_pipe.fit(pd.DataFrame(X, columns=flat_columns), y)

    # 3-tier high/medium/low risk bands (colmi_r02_client/injury_predict.py's
    # dashboard shows only this coarse label, never a raw percentage -- see
    # README caveats on precision at this positive rate). Percentile cutpoints
    # over the deployed pipeline's own training-window predictions, same
    # approach as model_artifact/'s risk_band_cutpoints but collapsed to 3
    # bands instead of 4.
    train_probs = final_pipe.predict_proba(pd.DataFrame(X, columns=flat_columns))[:, 1]
    risk_band_cutpoints = {
        "medium": float(np.percentile(train_probs, 75)),
        "high": float(np.percentile(train_probs, 95)),
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    import joblib
    joblib.dump(final_pipe, ARTIFACT_DIR / "pipeline.joblib")

    manifest = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": model_cfg.name,
        "imbalance_strategy": imbalance_cfg.strategy,
        "target_mode": base_cfg.target.mode,
        "target_gap_days": base_cfg.target.gap_days,
        "window": {
            "input_sessions": window_cfg.input_sessions,
            "output_days": window_cfg.output_days,
            "max_span_days": window_cfg.max_span_days,
        },
        "feature_cols": feature_cols,
        "flat_columns": flat_columns,
        "threshold": threshold,
        "risk_band_cutpoints": risk_band_cutpoints,
        "n_training_windows": int(len(y)),
        "n_training_positives": int(y.sum()),
        "n_players": int(len(unique_players)),
        "note": "RING-DERIVED FEATURE SUBSET, not the full SoccerMon model (see "
                "model_artifact/manifest.json for that one). Trained on only "
                "stress/sleep_duration/sleep_quality/fatigue + day_of_week -- dropped "
                "readiness/soreness/mood (no physiological proxy) and all coach-logged "
                "training-load features (acwr/atl/ctl.../rpe.../n_sessions/...). At "
                "inference time colmi_r02_client/injury_predict.py fills these 4 "
                "features from ring-derived proxies, not real subjective surveys -- "
                "treat predictions as a rough, unvalidated-on-real-ring-data estimate. "
                "validated_metrics/holdout_metrics below are computed on REAL SUBJECTIVE "
                "SoccerMon labels, not on ring proxy data, so they measure this reduced "
                "feature set's ceiling under ideal (real-survey) inputs, not the accuracy "
                "of the ring proxies themselves.",
        "validated_metrics": {
            "pr_auc_mean": validated_metrics["pr_auc_mean"],
            "pr_auc_ci_low": validated_metrics["pr_auc_ci_low"],
            "pr_auc_ci_high": validated_metrics["pr_auc_ci_high"],
            "precision_mean": validated_metrics["precision_mean"],
            "recall_mean": validated_metrics["recall_mean"],
            "f1_mean": validated_metrics["f1_mean"],
            "brier_score_mean": validated_metrics["brier_score_mean"],
            "n_folds": validated_metrics["n_folds"],
            "n_repeats": validated_metrics["n_repeats"],
            "n_folds_skipped": validated_metrics["n_folds_skipped"],
        },
        "holdout_metrics": {
            "pr_auc": holdout_metrics["pr_auc"],
            "precision": holdout_metrics["precision"],
            "recall": holdout_metrics["recall"],
            "f1": holdout_metrics["f1"],
            "brier_score": holdout_metrics["brier_score"],
            "n_holdout_players": n_holdout,
            "n_holdout_windows": int(holdout_mask.sum()),
            "n_holdout_positives": int(y[holdout_mask].sum()),
        },
    }
    with open(ARTIFACT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Exported ring-feature pipeline + manifest to {ARTIFACT_DIR}")
