"""Turns the last week of ring history into a high/medium/low injury-risk
band for the next few days, using the reduced-feature model trained by
../injury 2/scripts/train_ring_model.py (model_artifact_ring/), not the full
SoccerMon model in injury 2/model_artifact/.

That full model needs 15 inputs: subjective wellness surveys (readiness,
soreness, fatigue, mood, stress, sleep_quality, sleep_duration) and
coach-logged training-load metrics (acwr, atl, ctl28/42, daily_load,
monotony, strain, rpe_mean/max, n_sessions, ...). A ring cannot produce most
of those - readiness/soreness/mood have no physiological proxy, and the
training-load metrics come from RPE x duration session diaries, not sensor
data. So model_artifact_ring/ was retrained on only the 4 features (+
day_of_week, free from the calendar) that this module can approximate:

    - stress          <- stress_predict.py's HR-based stress/no-stress model,
                          rescaled from a probability onto the training data's
                          1-10 self-report scale. A binary HR classifier's
                          probability is not the same construct as a person's
                          subjective 1-10 stress rating; treat it as a rough
                          proxy, not a substitute.
    - sleep_duration,
      sleep_quality    <- sleep.py's Big Data sleep-stage sync. sleep.py is
                          EXPERIMENTAL/unverified against real hardware (see
                          its docstring) - if it fails or the ring doesn't
                          support it, these fall back to "missing" rather than
                          a guessed value, exactly like SoccerMon players who
                          skipped a wellness survey (the model was trained
                          with real missingness of this shape).
    - fatigue          <- NOT the subjective fatigue rating the model was
                          trained on. This is a movement-volume proxy (today's
                          calories from steps.SportDetail vs. a trailing
                          baseline), standing in for "how much load has this
                          person's body taken on lately" the same way GPS
                          player-load would in a real sports-science setup.
                          Perceived fatigue and training volume are related
                          but distinct constructs - a spike here means "moved
                          a lot", not necessarily "feels tired".

validated_metrics/holdout_metrics in model_artifact_ring/manifest.json were
computed on real SoccerMon subjective labels, not on ring-proxy data - they
show this reduced feature set's ceiling under ideal inputs, not the accuracy
of these proxies. Treat any prediction from this module as an illustrative
estimate, not a validated clinical or coaching signal.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd

from colmi_r02_client import date_utils, hr, sleep as ring_sleep, steps, stress_predict
from colmi_r02_client.client import Client

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# NOTE: relies on the sibling project folder literally being named "injury 2".
ARTIFACT_DIR = os.path.join(REPO_ROOT, "injury 2", "model_artifact_ring")
PIPELINE_PATH = os.path.join(ARTIFACT_DIR, "pipeline.joblib")
MANIFEST_PATH = os.path.join(ARTIFACT_DIR, "manifest.json")

FATIGUE_BASELINE_DAYS = 14
"""How many trailing days of step/calorie history to average as the "normal
load" a day's movement volume gets compared against, to build the fatigue proxy."""

STRESS_MIN_READINGS = 2


class ModelNotTrained(RuntimeError):
    """model_artifact_ring/{pipeline.joblib,manifest.json} haven't been generated yet.
    Run `python scripts/train_ring_model.py` from the `injury 2/` directory."""


class NotEnoughData(RuntimeError):
    """Not enough ring history to build a full prediction window."""


@dataclass
class DayObservation:
    day: date
    stress: float | None = None
    sleep_duration_hours: float | None = None
    sleep_quality: float | None = None
    daily_calories: float | None = None
    fatigue: float | None = None


RISK_BANDS = ("low", "medium", "high")


def risk_band_from_probability(probability: float, cutpoints: dict[str, float]) -> str:
    """Collapses the raw probability to a 3-tier low/medium/high band using
    manifest["risk_band_cutpoints"] (percentiles of the model's own training
    predictions - see train_ring_model.py). At this positive rate (<1%),
    precision on any single probability threshold is low (see README), so
    the dashboard only ever shows this coarse band, never the raw number.
    """
    if probability >= cutpoints["high"]:
        return "high"
    if probability >= cutpoints["medium"]:
        return "medium"
    return "low"


@dataclass
class InjuryRiskResult:
    risk_probability: float
    risk_band: str
    above_operating_threshold: bool
    threshold: float
    days_used: list[DayObservation] = field(default_factory=list)

    def measured_fraction(self) -> float:
        """Share of (day x feature) cells that came from real ring data rather
        than falling back to the model's missing-value imputation - a rough
        confidence signal, since this prediction leans on 4 approximated
        features to begin with."""
        cells = []
        for d in self.days_used:
            cells += [d.stress is not None, d.sleep_duration_hours is not None,
                      d.sleep_quality is not None, d.fatigue is not None]
        return sum(cells) / len(cells) if cells else 0.0


def _load_pipeline_and_manifest():
    if not os.path.exists(PIPELINE_PATH) or not os.path.exists(MANIFEST_PATH):
        raise ModelNotTrained(
            f"Missing {PIPELINE_PATH} or {MANIFEST_PATH}.\n"
            "From 'injury 2/', run `python scripts/train_ring_model.py` to generate them."
        )
    pipeline = joblib.load(PIPELINE_PATH)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return pipeline, manifest


def _stress_proxy(readings: list[tuple[datetime, int]]) -> float | None:
    """Maps stress_predict's [0, 1] stress probability onto the SoccerMon
    stress feature's 1-10 self-report scale. See module docstring caveat."""
    if len(readings) < STRESS_MIN_READINGS:
        return None
    try:
        _, proba = stress_predict.predict_stress(readings)
    except (stress_predict.ModelNotTrained, stress_predict.NotEnoughData):
        return None
    return 1.0 + proba * 9.0


def _fatigue_proxy(daily_calories: float | None, baseline_calories: float | None) -> float | None:
    """Today's movement volume relative to the trailing baseline, mapped onto
    a 1-10 scale: at the baseline -> ~5.5, at 2x the baseline or more -> 10,
    zero activity -> 1. See module docstring caveat - this is training/
    movement volume, not felt fatigue."""
    if daily_calories is None or not baseline_calories:
        return None
    ratio = daily_calories / (2.0 * baseline_calories)
    return max(1.0, min(10.0, 1.0 + 9.0 * ratio))


async def _fetch_day_hr_readings(client: Client, day: date) -> list[tuple[datetime, int]]:
    day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    log = await client.get_heart_rate_log(day_start)
    if not isinstance(log, hr.HeartRateLog):
        return []
    return [(ts, reading) for reading, ts in log.heart_rates_with_times() if reading != 0]


async def _fetch_day_calories(client: Client, day: date) -> float | None:
    day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    result = await client.get_steps(day_start)
    if isinstance(result, steps.NoData) or not result:
        return None
    return float(sum(s.calories for s in result))


async def collect_observations(client: Client, input_sessions: int) -> list[DayObservation]:
    """Gathers `input_sessions` days of ring history (oldest first, ending
    today) plus a longer trailing baseline for the fatigue proxy. Expects an
    already-connected client (call inside `async with client:`).
    """
    today = date_utils.now().date()
    baseline_days = [today - timedelta(days=i) for i in range(FATIGUE_BASELINE_DAYS)]
    window_days = list(reversed(baseline_days[:input_sessions]))  # oldest -> newest

    calories_by_day: dict[date, float | None] = {}
    for d in baseline_days:
        calories_by_day[d] = await _fetch_day_calories(client, d)
    known_calories = [c for c in calories_by_day.values() if c is not None]
    baseline_calories = float(np.mean(known_calories)) if known_calories else None

    try:
        sleep_nights = await client.get_sleep_data()
    except Exception as e:  # sleep.py is experimental; never let it break the whole prediction
        logger.warning(f"Sleep data fetch failed, treating as unavailable: {e}")
        sleep_nights = ring_sleep.NoData()
    nights_by_date: dict[date, ring_sleep.SleepNight] = {}
    if not isinstance(sleep_nights, ring_sleep.NoData):
        nights_by_date = {n.night_date: n for n in sleep_nights}

    observations = []
    for d in window_days:
        readings = await _fetch_day_hr_readings(client, d)
        stress = _stress_proxy(readings)

        night = nights_by_date.get(d)
        sleep_duration_hours = night.total_asleep_minutes / 60.0 if night else None
        sleep_quality = night.quality_score if night else None

        daily_calories = calories_by_day.get(d)
        fatigue = _fatigue_proxy(daily_calories, baseline_calories)

        observations.append(DayObservation(
            day=d, stress=stress, sleep_duration_hours=sleep_duration_hours,
            sleep_quality=sleep_quality, daily_calories=daily_calories, fatigue=fatigue,
        ))
    return observations


def _build_flat_row(observations: list[DayObservation], feature_cols: list[str],
                     flat_columns: list[str]) -> pd.DataFrame:
    """Same t-major/feature-minor raw layout + slope/delta/std trend columns
    as injury 2/src/windowing/windows.py::build_windows / compute_trend_features,
    reimplemented locally (mirrors how stress_predict.py stays self-contained
    rather than importing across the "injury 2" project boundary).
    """
    value_lookup = {
        "stress": lambda o: o.stress,
        "stress_missing": lambda o: 0.0 if o.stress is not None else 1.0,
        "sleep_duration": lambda o: o.sleep_duration_hours,
        "sleep_duration_missing": lambda o: 0.0 if o.sleep_duration_hours is not None else 1.0,
        "sleep_quality": lambda o: o.sleep_quality,
        "sleep_quality_missing": lambda o: 0.0 if o.sleep_quality is not None else 1.0,
        "fatigue": lambda o: o.fatigue,
        "fatigue_missing": lambda o: 0.0 if o.fatigue is not None else 1.0,
        "day_of_week": lambda o: float(o.day.weekday()),
    }
    missing = [c for c in feature_cols if c not in value_lookup]
    if missing:
        raise ModelNotTrained(
            f"model_artifact_ring/manifest.json expects feature_cols {missing} that "
            "colmi_r02_client/injury_predict.py doesn't know how to build. The ring "
            "model was likely retrained with a different feature set; update "
            "_build_flat_row's value_lookup to match."
        )

    def _value(feat: str, o: DayObservation) -> float:
        v = value_lookup[feat](o)
        return np.nan if v is None else v

    X_seq = np.array([[_value(feat, o) for feat in feature_cols] for o in observations])[None, :, :]

    input_sessions = X_seq.shape[1]
    flat_raw_columns = [f"{feat}_t{t}" for t in range(1, input_sessions + 1) for feat in feature_cols]
    flat = pd.DataFrame(X_seq.reshape(1, -1), columns=flat_raw_columns)

    # Plain (non-nan-aware) mean/std, matching injury 2/src/windowing/windows.py::
    # compute_trend_features exactly: a window with a missing day propagates NaN
    # into the whole trend feature rather than silently averaging over the rest,
    # so it hits the same downstream median imputer the model was calibrated on.
    t = np.arange(input_sessions, dtype=float)
    t_centered = t - t.mean()
    denom = np.sum(t_centered ** 2)
    x_mean = X_seq.mean(axis=1, keepdims=True)
    x_centered = X_seq - x_mean
    slope = np.tensordot(x_centered, t_centered, axes=([1], [0])) / denom
    delta = X_seq[:, -1, :] - X_seq[:, 0, :]
    std = X_seq.std(axis=1)

    trend_cols = {}
    for j, feat in enumerate(feature_cols):
        trend_cols[f"{feat}_slope"] = slope[:, j]
        trend_cols[f"{feat}_delta"] = delta[:, j]
        trend_cols[f"{feat}_std"] = std[:, j]
    trend = pd.DataFrame(trend_cols)

    combined = pd.concat([flat, trend], axis=1)
    return combined.reindex(columns=flat_columns)


async def predict_injury_risk(client: Client) -> InjuryRiskResult:
    """client must already be connected (call inside `async with client:`)."""
    pipeline, manifest = _load_pipeline_and_manifest()
    input_sessions = manifest["window"]["input_sessions"]

    observations = await collect_observations(client, input_sessions)
    flat_row = _build_flat_row(observations, manifest["feature_cols"], manifest["flat_columns"])

    probability = float(pipeline.predict_proba(flat_row)[:, 1][0])
    threshold = manifest["threshold"]
    return InjuryRiskResult(
        risk_probability=probability,
        risk_band=risk_band_from_probability(probability, manifest["risk_band_cutpoints"]),
        above_operating_threshold=probability >= threshold,
        threshold=threshold,
        days_used=observations,
    )
