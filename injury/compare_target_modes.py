"""A/B comparison: target.mode="new_onset" (the default -- label=1 only on
the first report after a >=gap_days gap) vs. "any_report" (label=1 on every
date with an active qualifying complaint).

Why this matters: new_onset collapses SoccerMon's repeated-complaint
surveillance records (one player has 47 rows over ~2 months for one evolving
injury) down to ~15-117 true onset events depending on window config -- a
positive class small enough that GroupKFold folds routinely end up
single-class (see README's "expect several folds to end up with a single-
class training set" note). any_report keeps every active-complaint date as
positive, trading "predicts a genuinely new problem" for a much larger,
less scarce positive class. This script runs the same grid under both
target modes so the scarcity-vs-signal tradeoff is visible side by side,
instead of assumed.

Usage:
    python scripts/compare_target_modes.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, with_overrides
from src.eval.report import aggregate_fold_metrics, build_results_table, config_to_row, print_results_table
from scripts.run_experiment import build_daily_df, load_raw_frames, run_cv_for_windows
from src.windowing.windows import build_windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--repeats", type=int, default=3,
                         help="GroupKFold repeats to pool per config (default 3).")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    base_cfg.cv.n_repeats = args.repeats
    base_cfg.window.input_sessions, base_cfg.window.output_days = 3, 7  # the deployed window config

    raw_frames = load_raw_frames(base_cfg.data)
    injury_path = Path(base_cfg.data.subjective_dir) / "injury" / "injury.csv"

    rows = []
    for target_mode in ("new_onset", "any_report"):
        for model_name in ("logreg", "random_forest", "xgboost"):
            for imbalance in ("none", "random_oversample"):
                cfg = with_overrides(
                    base_cfg, target__mode=target_mode, model__name=model_name,
                    imbalance__strategy=imbalance,
                    name=f"{target_mode}_{model_name}_{imbalance}")
                daily_df = build_daily_df(raw_frames, injury_path, cfg.data, cfg.target)
                try:
                    windows = build_windows(daily_df, cfg.window)
                except ValueError as e:
                    print(f"[skip] {cfg.name}: {e}")
                    continue
                fold_metrics, n_folds_skipped = run_cv_for_windows(windows, cfg)
                if not fold_metrics:
                    print(f"[skip] {cfg.name}: every fold was skipped.")
                    continue
                agg = aggregate_fold_metrics(fold_metrics)
                agg["n_folds_skipped"] = n_folds_skipped
                rows.append(config_to_row(cfg, agg))

    if not rows:
        print("No configs produced results.")
        return

    df = build_results_table(rows)
    print("\n=== new_onset vs any_report, input_sessions=3 output_days=7, "
          f"{args.repeats} repeats pooled ===")
    print_results_table(df)

    for mode in ("new_onset", "any_report"):
        sub = df[df["target_mode"] == mode]
        if not sub.empty:
            best = sub.iloc[0]
            print(f"\nBest {mode}: {best['name']}  PR-AUC={best['pr_auc']} {best['pr_auc_ci95']}  "
                  f"n_positives={best['n_positives']}  players_caught={best['players_caught']}")


if __name__ == "__main__":
    main()
