"""
Turns heart-rate readings from the ring into a stress / no-stress prediction,
using the Random Forest trained by stress-ml/train_ring_model.py.

Caveat: the WESAD ECG_Rate signal the model was trained on is a continuous,
high-resolution (700Hz) heart-rate curve derived from a chest ECG. The ring
only gives sparse, irregularly-spaced HR readings from an optical sensor, so
the rolling mean/std computed here are a coarser approximation of the same
quantities the model saw during training, not an exact match. The original
model_11 (trained on EDA/respiration/skin temperature too, which the ring
can't measure) is not used here - see stress-ml/README.md.
"""

import os
from datetime import datetime

import joblib
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRESS_ML_DIR = os.path.join(REPO_ROOT, "stress-ml")
MODELS_DIR = os.path.join(STRESS_ML_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model_11_ring.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# ECG_Rate is WESAD's beats-per-minute signal; it's used as the ring's heart
# rate proxy. Must match stress_predictor.RING_FEATURES.
RING_FEATURES = [
    "ECG_Rate",
    "ECG_Rate_mean_60s",
    "ECG_Rate_std_60s",
    "ECG_Rate_mean_300s",
    "ECG_Rate_std_300s",
]

MIN_WINDOW_SECONDS = 300  # the model's longest rolling window


class ModelNotTrained(RuntimeError):
    """model_11_ring.pkl / scaler.pkl haven't been generated yet."""


class NotEnoughData(RuntimeError):
    """Not enough heart rate history to compute the model's rolling features."""


def _load_model_and_scaler():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise ModelNotTrained(
            f"Missing {MODEL_PATH} or {SCALER_PATH}.\n"
            "From stress-ml/, run `python stress_predictor.py` once (to cache "
            "WESAD features, Ctrl+C after 'Saved preprocessed data to ...' is "
            "fine) then `python train_ring_model.py` to generate them."
        )
    model = joblib.load(MODEL_PATH)
    scaler_bundle = joblib.load(SCALER_PATH)
    return model, scaler_bundle["scaler"], scaler_bundle["columns"]


def _build_features(readings: list[tuple[datetime, int]]) -> dict[str, float]:
    """
    readings: (timestamp, bpm) tuples, any order, from Client.stream_heart_rate().
    Resamples to ~1Hz and computes the same rolling mean/std features (60s,
    300s windows) the model was trained on, for the most recent timestamp.
    """
    if len(readings) < 2:
        raise NotEnoughData("Need at least a few heart rate readings to predict stress")

    series = pd.Series(
        [bpm for _, bpm in readings],
        index=pd.DatetimeIndex([ts for ts, _ in readings]),
    ).sort_index()
    series = series[~series.index.duplicated(keep="last")]

    span_seconds = (series.index[-1] - series.index[0]).total_seconds()
    if span_seconds < MIN_WINDOW_SECONDS:
        raise NotEnoughData(
            f"Only {span_seconds:.0f}s of heart rate history collected, need at "
            f"least {MIN_WINDOW_SECONDS}s (the model's longest rolling window)."
        )

    resampled = series.resample("1s").mean().interpolate(limit_direction="both")

    return {
        "ECG_Rate": float(resampled.iloc[-1]),
        "ECG_Rate_mean_60s": float(resampled.tail(60).mean()),
        "ECG_Rate_std_60s": float(resampled.tail(60).std(ddof=0)),
        "ECG_Rate_mean_300s": float(resampled.tail(300).mean()),
        "ECG_Rate_std_300s": float(resampled.tail(300).std(ddof=0)),
    }


def _scale_ring_features(raw_features: dict[str, float], scaler, scaler_columns: list[str]) -> list[float]:
    """Standardize using the mean/std the scaler learned per column at train time."""
    scaled = []
    for col in RING_FEATURES:
        idx = scaler_columns.index(col)
        scaled.append((raw_features[col] - scaler.mean_[idx]) / scaler.scale_[idx])
    return scaled


def predict_stress(readings: list[tuple[datetime, int]]) -> tuple[str, float]:
    """
    readings: (timestamp, bpm) tuples, e.g. from Client.stream_heart_rate().
    Returns (label, stress_probability) where label is "stress" or "no stress".
    Raises ModelNotTrained or NotEnoughData if prediction isn't possible yet.
    """
    model, scaler, scaler_columns = _load_model_and_scaler()
    raw_features = _build_features(readings)
    scaled_values = _scale_ring_features(raw_features, scaler, scaler_columns)

    X = pd.DataFrame([scaled_values], columns=RING_FEATURES)
    proba_stress = float(model.predict_proba(X)[0][1])
    label = "stress" if proba_stress >= 0.5 else "no stress"
    return label, proba_stress
