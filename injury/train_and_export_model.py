"""Fits the best config found in the grid search (XGBoost, input_sessions=3,
output_days=7, new_onset target, no synthetic oversampling -- see the earlier
grid + imbalance-strategy comparison) on the FULL real SoccerMon dataset, and
exports everything the demo Flask app (app.py) needs to serve predictions:

- model_artifact/pipeline.joblib     the fitted impute->scale->classify pipeline
- model_artifact/manifest.json        feature/column order, window config, threshold
- model_artifact/players_snapshot.json  per-player recent history + current risk,
                                         precomputed so the frontend's player list
                                         and detail view don't need live inference
- model_artifact/sample_inputs.json    a few illustrative (synthetic, clearly
                                         labeled as such) what-if scenarios for the
                                         manual-input demo path

This is a DEMO artifact: it's fit on all available data (not held out) because
its job is to back an MVP showcase, not to reproduce the validated PR-AUC
numbers from scripts/run_experiment.py -- those come from proper GroupKFold CV
and remain the source of truth for actual model quality.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ImbalanceConfig, ModelConfig, load_config, with_overrides
from src.data import raw_loaders
from src.eval.metrics import compute_metrics, tune_threshold
from src.eval.report import aggregate_fold_metrics
from src.inference import flatten_window, get_latest_window
from src.models.pipeline import build_flat_pipeline
from src.windowing.windows import build_windows, get_feature_columns
from scripts.run_experiment import build_daily_df, load_raw_frames, run_cv_for_windows

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "model_artifact"

RECENT_HISTORY_COLUMNS = [
    "readiness", "soreness", "fatigue", "sleep_quality", "sleep_duration",
    "mood", "stress", "acwr", "daily_load", "rpe_mean", "n_sessions",
]
RECENT_HISTORY_DAYS = 21
RISK_PERCENTILE_CUTPOINTS = (60, 85, 95)  # -> low / moderate / serious / critical


def _clean(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def risk_band(probability: float, cutpoints: dict[str, float]) -> str:
    if probability >= cutpoints["critical"]:
        return "critical"
    if probability >= cutpoints["serious"]:
        return "serious"
    if probability >= cutpoints["moderate"]:
        return "moderate"
    return "low"


def build_players_snapshot(daily_df, pipeline, feature_cols, flat_columns, window_cfg,
                            injury_path: Path, date_format: str) -> list[dict]:
    events = raw_loaders.load_injury_events(injury_path, date_format, min_severity="minor")
    injury_counts = events.groupby("player_id").size().to_dict() if not events.empty else {}

    per_player = []
    for player_id, group in daily_df.groupby("player_id"):
        group = group.sort_values("date")
        latest = get_latest_window(group, window_cfg, feature_cols)

        entry = {
            "player_id": player_id,
            "team": player_id.split("-")[0],
            "n_days_tracked": int(len(group)),
            "injury_reports_ever": int(injury_counts.get(player_id, 0)),
            "recent_history": [],
            "risk_probability": None,
            "window_start": None,
            "window_end": None,
            "insufficient_data": latest is None,
        }

        recent = group.tail(RECENT_HISTORY_DAYS)
        for _, row in recent.iterrows():
            point = {"date": _clean(row["date"])}
            for col in RECENT_HISTORY_COLUMNS:
                point[col] = _clean(row.get(col))
            entry["recent_history"].append(point)

        if latest is not None:
            X_seq, window_rows = latest
            flat_row = flatten_window(X_seq, feature_cols, window_cfg.input_sessions, flat_columns)
            entry["risk_probability"] = round(float(pipeline.predict_proba(flat_row)[:, 1][0]), 6)
            entry["window_start"] = _clean(window_rows["date"].iloc[0])
            entry["window_end"] = _clean(window_rows["date"].iloc[-1])

        per_player.append(entry)

    valid_scores = np.array([p["risk_probability"] for p in per_player if p["risk_probability"] is not None])
    cutpoints = {
        "moderate": float(np.percentile(valid_scores, RISK_PERCENTILE_CUTPOINTS[0])),
        "serious": float(np.percentile(valid_scores, RISK_PERCENTILE_CUTPOINTS[1])),
        "critical": float(np.percentile(valid_scores, RISK_PERCENTILE_CUTPOINTS[2])),
    }
    for p in per_player:
        p["risk_band"] = risk_band(p["risk_probability"], cutpoints) if p["risk_probability"] is not None else None

    per_player.sort(key=lambda p: (p["risk_probability"] is None, -(p["risk_probability"] or 0)))
    return per_player, cutpoints


def build_sample_inputs() -> dict:
    """Illustrative what-if scenarios for the manual-input demo path.
    EXPLICITLY SYNTHETIC -- not derived from any real athlete -- so this can
    never be mistaken for real SoccerMon data (unlike players_snapshot.json,
    which is entirely real).
    """
    return {
        "_note": "These are illustrative synthetic scenarios for the what-if "
                 "demo form, NOT real athletes. Real per-player data lives in "
                 "players_snapshot.json.",
        "scenarios": {
            "well_recovered": {
                "label": "Well-recovered training day",
                "trend_direction": "flat",
                "inputs": {
                    "readiness": 8, "soreness": 2, "fatigue": 2, "sleep_quality": 4,
                    "sleep_duration": 8, "mood": 4, "stress": 2, "acwr": 0.9,
                    "daily_load": 400, "monotony": 1.2, "strain": 900, "atl": 220,
                    "ctl28": 230, "rpe_mean": 5, "n_sessions": 1,
                },
            },
            "accumulating_fatigue": {
                "label": "Accumulating fatigue, load creeping up",
                "trend_direction": "worsening",
                "inputs": {
                    "readiness": 4, "soreness": 4, "fatigue": 4, "sleep_quality": 2,
                    "sleep_duration": 6, "mood": 2, "stress": 4, "acwr": 1.6,
                    "daily_load": 850, "monotony": 2.4, "strain": 4200, "atl": 480,
                    "ctl28": 380, "rpe_mean": 8, "n_sessions": 2,
                },
            },
            "acute_load_spike": {
                "label": "Acute load spike after a light patch (classic ACWR-spike profile)",
                "trend_direction": "worsening",
                "inputs": {
                    "readiness": 5, "soreness": 3, "fatigue": 3, "sleep_quality": 3,
                    "sleep_duration": 7, "mood": 3, "stress": 3, "acwr": 2.6,
                    "daily_load": 1100, "monotony": 3.1, "strain": 6800, "atl": 610,
                    "ctl28": 260, "rpe_mean": 9, "n_sessions": 2,
                },
            },
        },
    }


if __name__ == "__main__":
    base_cfg = load_config("config/default.yaml")
    raw_frames = load_raw_frames(base_cfg.data)
    injury_path = Path(base_cfg.data.subjective_dir) / "injury" / "injury.csv"
    daily_df = build_daily_df(raw_frames, injury_path, base_cfg.data, base_cfg.target)

    window_cfg = base_cfg.window
    window_cfg.input_sessions, window_cfg.output_days = 3, 7
    model_cfg = ModelConfig(name="xgboost")
    imbalance_cfg = ImbalanceConfig(strategy="none")

    windows = build_windows(daily_df, window_cfg)
    feature_cols = windows.feature_cols
    flat_columns = list(windows.X_flat.columns)
    X, y, groups = windows.X_flat.to_numpy(), windows.y, windows.groups

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

    # Honest validated performance for the exact config being deployed: proper
    # repeated GroupKFold-by-player CV (never the all-data-fit pipeline below).
    # This is what model_artifact/manifest.json's "validated_metrics" and the
    # demo UI's metrics banner show -- see README's "Non-negotiables" section
    # for why this must never be conflated with the serving pipeline's fit.
    cv_cfg = with_overrides(base_cfg, window__input_sessions=window_cfg.input_sessions,
                             window__output_days=window_cfg.output_days,
                             model__name=model_cfg.name, imbalance__strategy=imbalance_cfg.strategy,
                             cv__n_repeats=5, name="deployed_config_validated")
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
        "n_training_windows": int(len(y)),
        "n_training_positives": int(y.sum()),
        "n_players": int(len(unique_players)),
        "note": "The SERVING pipeline (pipeline.joblib) is fit on the FULL dataset for demo "
                "coverage, not held out -- its own predictions are not evidence of model quality. "
                "validated_metrics (repeated player-level GroupKFold CV) and holdout_metrics "
                "(a 20% never-trained-on player slice) below are the actual quality evidence; "
                "see scripts/run_experiment.py for the full config grid.",
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

    players_snapshot, risk_cutpoints = build_players_snapshot(
        daily_df, final_pipe, feature_cols, flat_columns, window_cfg,
        injury_path, base_cfg.data.date_format)
    manifest["risk_band_cutpoints"] = risk_cutpoints
    with open(ARTIFACT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    with open(ARTIFACT_DIR / "players_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(players_snapshot, f, indent=2)

    with open(ARTIFACT_DIR / "sample_inputs.json", "w", encoding="utf-8") as f:
        json.dump(build_sample_inputs(), f, indent=2)

    print(f"Exported pipeline + manifest + {len(players_snapshot)} player snapshots to {ARTIFACT_DIR}")
    print(f"Risk band cutpoints (percentile-based): {risk_cutpoints}")
