"""Discrete-time hazard/survival evaluation: an alternative framing of the
same injury-prediction problem, alongside (not replacing) the direct
output_days=N window classification in scripts/run_experiment.py.

Each output_days=1 window is already a person-period hazard estimate --
p(onset in the next single day | window features) -- in the standard
discrete-time survival sense (Singer & Willett). This script fits that
1-day-ahead hazard model under the same leakage-safe GroupKFold-by-player CV
as run_experiment.py, then evaluates it two ways a hazard model is normally
judged, neither of which the direct classification framing answers:

1. Concordance index (Harrell's C): does the model rank players who go on to
   report a new injury as higher-risk than players who don't, over their
   observed follow-up? (src/eval/survival_metrics.concordance_index)
2. Cumulative incidence: compounding the model's own daily hazard estimates
   forward (1 - prod(1 - hazard) over a horizon) against actually observing
   an event in that horizon -- does chaining daily hazards track multi-day
   risk as well as directly classifying output_days=7 windows does?

Usage:
    python scripts/run_survival_experiment.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.config import ImbalanceConfig, ModelConfig, load_config, with_overrides
from src.cv.splitters import iter_repeated_splits
from src.eval.survival_metrics import (concordance_index, cumulative_incidence,
                                        player_episodes, rolling_horizon_label)
from src.models.pipeline import build_flat_pipeline
from src.windowing.windows import build_windows
from scripts.run_experiment import build_daily_df, load_raw_frames

HORIZON_DAYS = 7  # compared against the grid's best output_days=7 direct-classification PR-AUC (see README)


def run_one_config(daily_df, cfg) -> dict | None:
    windows = build_windows(daily_df, cfg.window)
    X, y = windows.X_flat.to_numpy(), windows.y
    groups, dates = windows.groups, windows.window_end_date

    c_indices, ci_pr_aucs, n_events_total = [], [], 0

    for repeat, train_idx, test_idx in iter_repeated_splits(X, y, groups, cfg.cv):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(set(y_train)) < 2:
            continue
        X_train, X_test = X[train_idx], X[test_idx]
        groups_test, dates_test = groups[test_idx], dates[test_idx]

        pipe = build_flat_pipeline(cfg.model, cfg.imbalance, cfg.cv.random_seed, y_train,
                                    cfg.feature_selection, X_train.shape[1])
        pipe.fit(X_train, y_train)
        prob_test = pipe.predict_proba(X_test)[:, 1]

        episodes = player_episodes(y_test, prob_test, groups_test, dates_test)
        n_events_total += int(episodes["event_observed"].sum())
        c = concordance_index(episodes["event_time"], episodes["risk_score"].to_numpy(),
                               episodes["event_observed"].to_numpy())
        if c is not None:
            c_indices.append(c)

        # cumulative incidence, computed per player from that player's own
        # chronologically-ordered daily hazard predictions in this fold's test set
        df = pd.DataFrame({"player_id": groups_test, "date": dates_test,
                            "y_true": y_test, "y_prob": prob_test}).sort_values(["player_id", "date"])
        ci_all, label_all = [], []
        for _, g in df.groupby("player_id"):
            ci_all.append(cumulative_incidence(g["y_prob"].to_numpy(), HORIZON_DAYS))
            label_all.append(rolling_horizon_label(g["y_true"].to_numpy(), HORIZON_DAYS))
        ci_all = np.concatenate(ci_all)
        label_all = np.concatenate(label_all)
        if label_all.sum() > 0:
            ci_pr_aucs.append(average_precision_score(label_all, ci_all))

    if not c_indices and not ci_pr_aucs:
        return None
    return {
        "name": cfg.name,
        "model": cfg.model.name,
        "input_sessions": cfg.window.input_sessions,
        "n_folds_with_events": len(c_indices),
        "c_index_mean": float(np.mean(c_indices)) if c_indices else float("nan"),
        "c_index_std": float(np.std(c_indices)) if c_indices else float("nan"),
        "cum_incidence_pr_auc_mean": float(np.mean(ci_pr_aucs)) if ci_pr_aucs else float("nan"),
        "n_events_total": n_events_total,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--repeats", type=int, default=3,
                         help="GroupKFold repeats to pool per config (default 3; see run_experiment.py --repeats).")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    base_cfg.cv.n_repeats = args.repeats
    base_cfg.window.output_days = 1  # every config here is a 1-day-ahead hazard model
    base_cfg.imbalance = ImbalanceConfig(strategy="none")  # matches the deployed config's best strategy

    raw_frames = load_raw_frames(base_cfg.data)
    injury_path = Path(base_cfg.data.subjective_dir) / "injury" / "injury.csv"
    daily_df = build_daily_df(raw_frames, injury_path, base_cfg.data, base_cfg.target)

    rows = []
    for input_sessions in (3, 5, 7):
        for model_name in ("logreg", "xgboost"):
            cfg = with_overrides(base_cfg, window__input_sessions=input_sessions,
                                  model__name=model_name, name=f"hazard_in{input_sessions}_{model_name}")
            try:
                row = run_one_config(daily_df, cfg)
            except ValueError as e:
                print(f"[skip] {cfg.name}: {e}")
                continue
            if row is None:
                print(f"[skip] {cfg.name}: no fold produced a comparable event pair.")
                continue
            rows.append(row)
            print(f"  {cfg.name:28s} C-index={row['c_index_mean']:.4f} (+/-{row['c_index_std']:.4f}, "
                  f"{row['n_folds_with_events']} folds)  "
                  f"cum-incidence-PR-AUC@{HORIZON_DAYS}d={row['cum_incidence_pr_auc_mean']:.4f}  "
                  f"n_events={row['n_events_total']}")

    if not rows:
        print("No configs produced results.")
        return

    df = pd.DataFrame(rows).sort_values("c_index_mean", ascending=False).reset_index(drop=True)
    print("\n=== Discrete-time hazard model results (C-index=0.5 is chance, 1.0 is perfect ranking) ===")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(df.to_string(index=False))
    print(f"\nCompare cum_incidence_pr_auc_mean above to the direct output_days={HORIZON_DAYS} PR-AUC numbers "
          f"in README.md / full_grid_output.txt -- same underlying question (multi-day risk), two different "
          f"model framings. rolling_horizon_label is an approximation of the direct output_days={HORIZON_DAYS} "
          f"label built from output_days=1 labels (see src/eval/survival_metrics.rolling_horizon_label "
          f"docstring), so treat this as directionally informative, not a byte-for-byte comparison.")


if __name__ == "__main__":
    main()
