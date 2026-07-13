"""CLI entrypoint: builds a grid of ExperimentConfig (window sizes x models),
runs the full load -> merge -> window -> CV -> eval pipeline for each, and
prints one results table across all configs.

Usage:
    python scripts/run_experiment.py --config config/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ExperimentConfig, load_config, with_overrides
from src.cv.splitters import iter_repeated_splits, train_val_split_for_threshold
from src.data import merge, raw_loaders
from src.data import target as target_mod
from src.eval.metrics import compute_metrics, tune_threshold
from src.eval.report import (aggregate_fold_metrics, build_results_table,
                              config_to_row, print_results_table)
from src.features.time_confounders import add_time_confounders
from src.models.pipeline import build_flat_pipeline, fit_predict_lstm
from src.windowing.windows import build_windows


def load_raw_frames(data_cfg):
    wellness = raw_loaders.load_wellness(data_cfg.subjective_dir, data_cfg.date_format)
    training_load = raw_loaders.load_training_load_wide(data_cfg.subjective_dir, data_cfg.date_format)
    session_daily = raw_loaders.load_session_daily(data_cfg.subjective_dir, data_cfg.date_format)
    matchdays = raw_loaders.load_matchdays(
        Path(data_cfg.subjective_dir) / "game-performance" / "game-performance.csv", data_cfg.date_format)
    gps_daily = raw_loaders.load_gps_daily(data_cfg.objective_dir, data_cfg.gps_column_map, data_cfg.gps_agg_func)
    return wellness, training_load, session_daily, matchdays, gps_daily


def build_daily_df(raw_frames, injury_path, data_cfg, target_cfg):
    wellness, training_load, session_daily, matchdays, gps_daily = raw_frames
    feature_frames = [wellness, training_load, session_daily, matchdays]
    grid_sources = list(feature_frames)
    if not gps_daily.empty:
        feature_frames.append(gps_daily)
        grid_sources.append(gps_daily)

    grid = merge.build_full_grid(grid_sources)
    daily_df = merge.merge_features(grid, feature_frames)
    daily_df = add_time_confounders(daily_df)

    events = raw_loaders.load_injury_events(injury_path, data_cfg.date_format, target_cfg.min_severity)
    daily_label = target_mod.build_daily_label(events, target_cfg)
    daily_df = merge.attach_injury_label(daily_df, daily_label)
    return daily_df


DEFAULT_THRESHOLD_FALLBACK = 0.5


def run_cv_for_windows(windows, cfg: ExperimentConfig) -> tuple[list[dict], int]:
    """Returns (fold_metrics, n_folds_skipped), pooled across cfg.cv.n_repeats
    independent GroupKFold shufflings (n_repeats=1 reproduces the old
    single-pass behavior exactly). With <1% positives spread across a
    handful of players, a GroupKFold split can easily leave a fold's
    training set (or its inner validation slice) with a single class -- that
    is an expected consequence of the real class imbalance, not a bug, so we
    skip the affected step with a warning instead of crashing the whole
    config.
    """
    fold_metrics = []
    n_folds_skipped = 0
    is_lstm = cfg.model.name == "lstm"
    X = windows.X_seq if is_lstm else windows.X_flat.to_numpy()
    y = windows.y
    groups = windows.groups

    for repeat, train_idx, test_idx in iter_repeated_splits(X, y, groups, cfg.cv):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]

        if len(set(y_train)) < 2:
            print(f"[{cfg.name}] skipping fold: training fold has only one class "
                  f"(n_pos={int(y_train.sum())}/{len(y_train)}).")
            n_folds_skipped += 1
            continue

        threshold = DEFAULT_THRESHOLD_FALLBACK
        try:
            inner_train_idx, inner_val_idx = train_val_split_for_threshold(X_train, y_train, groups_train, cfg.cv)
            y_fit, y_val = y_train[inner_train_idx], y_train[inner_val_idx]
            if len(set(y_fit)) < 2:
                raise ValueError("inner training split has only one class")
            X_fit, X_val = X_train[inner_train_idx], X_train[inner_val_idx]
            if is_lstm:
                prob_val = fit_predict_lstm(X_fit, y_fit, X_val, cfg.model, cfg.imbalance, cfg.cv.random_seed)
            else:
                pipe = build_flat_pipeline(cfg.model, cfg.imbalance, cfg.cv.random_seed, y_fit,
                                            cfg.feature_selection, X_fit.shape[1])
                pipe.fit(X_fit, y_fit)
                prob_val = pipe.predict_proba(X_val)[:, 1]
            threshold = tune_threshold(y_val, prob_val, cfg.threshold_metric)
        except ValueError as e:
            print(f"[{cfg.name}] threshold tuning skipped this fold ({e}); "
                  f"falling back to threshold={DEFAULT_THRESHOLD_FALLBACK}.")

        try:
            if is_lstm:
                prob_test = fit_predict_lstm(X_train, y_train, X_test, cfg.model, cfg.imbalance, cfg.cv.random_seed)
            else:
                pipe = build_flat_pipeline(cfg.model, cfg.imbalance, cfg.cv.random_seed, y_train,
                                            cfg.feature_selection, X_train.shape[1])
                pipe.fit(X_train, y_train)
                prob_test = pipe.predict_proba(X_test)[:, 1]
        except ValueError as e:
            print(f"[{cfg.name}] skipping fold: could not fit on training fold ({e}).")
            n_folds_skipped += 1
            continue

        metrics = compute_metrics(y_test, prob_test, groups_test, threshold)
        metrics["repeat"] = repeat
        fold_metrics.append(metrics)
    return fold_metrics, n_folds_skipped


def build_grid(base_cfg: ExperimentConfig) -> list[ExperimentConfig]:
    grid = []
    for input_sessions in (3, 5, 7):
        for output_days in (1, 3, 7):
            for model_name in ("logreg", "random_forest", "xgboost", "lstm"):
                grid.append(with_overrides(
                    base_cfg,
                    window__input_sessions=input_sessions,
                    window__output_days=output_days,
                    model__name=model_name,
                    name=f"in{input_sessions}_out{output_days}_{model_name}",
                ))
    return grid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--quick", action="store_true",
                         help="Run a single small config instead of the full grid (for smoke-testing).")
    parser.add_argument("--repeats", type=int, default=None,
                         help="Override cv.n_repeats: number of independently-shuffled "
                              "GroupKFold passes to pool per config, for a mean +/- 95%% CI "
                              "instead of a single-shuffle point estimate.")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    if args.repeats is not None:
        base_cfg.cv.n_repeats = args.repeats
    raw_frames = load_raw_frames(base_cfg.data)
    injury_path = Path(base_cfg.data.subjective_dir) / "injury" / "injury.csv"

    grid = [base_cfg] if args.quick else build_grid(base_cfg)

    daily_df_cache: dict = {}
    windows_cache: dict = {}
    rows = []
    for cfg in grid:
        target_key = (cfg.target.mode, cfg.target.gap_days, cfg.target.min_severity)
        if target_key not in daily_df_cache:
            daily_df_cache[target_key] = build_daily_df(raw_frames, injury_path, cfg.data, cfg.target)
        daily_df = daily_df_cache[target_key]

        window_key = target_key + (cfg.window.input_sessions, cfg.window.output_days,
                                    cfg.window.max_span_days, cfg.window.stride)
        if window_key not in windows_cache:
            try:
                windows_cache[window_key] = build_windows(daily_df, cfg.window)
            except ValueError as e:
                print(f"[skip] {cfg.name}: {e}")
                windows_cache[window_key] = None
        windows = windows_cache[window_key]
        if windows is None:
            continue

        try:
            fold_metrics, n_folds_skipped = run_cv_for_windows(windows, cfg)
        except Exception as e:
            print(f"[skip] {cfg.name}: {e}")
            continue
        if not fold_metrics:
            print(f"[skip] {cfg.name}: every fold was skipped (see warnings above).")
            continue

        agg = aggregate_fold_metrics(fold_metrics)
        agg["n_folds_skipped"] = n_folds_skipped
        rows.append(config_to_row(cfg, agg))

    if not rows:
        print("No configs produced results.")
        return

    df = build_results_table(rows)
    print_results_table(df)


if __name__ == "__main__":
    main()
