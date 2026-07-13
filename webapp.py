"""Local dashboard: a small Flask app that shows ring data (heart rate, SpO2,
HRV, sleep, steps/distance) plus the stress and injury-risk models, styled
after a rounded-card wellness-app aesthetic.

Two data sources, kept explicit so the UI can label them correctly:
  - /api/history   reads from the SQLite database populated by `colmi_r02_client
                    sync` (colmi_r02_client/db.py) -- historical trend charts.
  - /api/live       connects to the ring live over BLE for a fresh snapshot
                    (current HR/SpO2/HRV, today's steps, last night's sleep,
                    a stress prediction, and an injury-risk band). This does
                    ~20-30 sequential BLE round trips (injury risk alone needs
                    a 14-day movement baseline + a week of heart-rate logs),
                    so it can take up to a minute -- the frontend shows a
                    loading state, this is not a bug.

Notes/experimental features surfaced honestly in the API response itself
(not hidden): HRV real-time readings are flagged unreliable by
colmi_r02_client/real_time.py itself; sleep data depends on sleep.py's
unverified Big Data protocol implementation; the ring has no GPS/location
sensor, so "distance" here is step-derived, not a real route.

Run:
    python -m colmi_r02_client.webapp --address AA:BB:CC:DD:EE:FF
Then open http://127.0.0.1:5050
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import timedelta
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from colmi_r02_client import date_utils, db, hr, injury_predict, real_time, sleep as ring_sleep, steps, stress_predict
from colmi_r02_client.client import Client

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "webapp_static"

app = Flask(__name__, static_folder=None)
app.config["RING_ADDRESS"] = None
app.config["DB_PATH"] = None


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/history")
def api_history():
    days = request.args.get("days", default=7, type=int)
    db_path = app.config["DB_PATH"]
    if db_path is None or not Path(db_path).exists():
        return jsonify({"days": [], "note": "No synced database yet -- run `colmi_r02_client sync` first."})

    address = app.config["RING_ADDRESS"]
    since = date_utils.now() - timedelta(days=days)
    with db.get_db_session(Path(db_path)) as session:
        history = db.get_daily_history(session, address, since=since)
    return jsonify({"days": history})


def _hr_log_readings(hr_log) -> list[tuple]:
    if not isinstance(hr_log, hr.HeartRateLog):
        return []
    return [(ts, reading) for reading, ts in hr_log.heart_rates_with_times() if reading != 0]


async def _fetch_live_snapshot(address: str) -> dict:
    client = Client(address)
    result: dict = {"address": address}

    async with client:
        result["device_info"] = await client.get_device_info()

        battery = await client.get_battery()
        result["battery"] = {"level": battery.battery_level, "charging": battery.charging}

        hr_reading = await client.get_realtime_reading(real_time.RealTimeReading.HEART_RATE)
        result["heart_rate"] = {"value": hr_reading[-1] if hr_reading else None, "readings": hr_reading}

        spo2_reading = await client.get_realtime_reading(real_time.RealTimeReading.SPO2)
        result["spo2"] = {"value": spo2_reading[-1] if spo2_reading else None}

        hrv_value, hrv_error = None, None
        try:
            hrv_reading = await client.get_realtime_reading(real_time.RealTimeReading.HRV)
            hrv_value = hrv_reading[-1] if hrv_reading else None
        except Exception as e:
            hrv_error = str(e)
        result["hrv"] = {
            "value": hrv_value, "error": hrv_error,
            "caveat": "HRV over this protocol is flagged unreliable by colmi_r02_client/real_time.py itself.",
        }

        today = date_utils.now()
        steps_result = await client.get_steps(today)
        if isinstance(steps_result, steps.NoData) or not steps_result:
            result["today_activity"] = {"steps": 0, "calories": 0, "distance_m": 0}
        else:
            result["today_activity"] = {
                "steps": sum(s.steps for s in steps_result),
                "calories": sum(s.calories for s in steps_result),
                "distance_m": sum(s.distance for s in steps_result),
            }

        sleep_error = None
        last_night = None
        try:
            nights = await client.get_sleep_data()
            if isinstance(nights, ring_sleep.NoData) or not nights:
                sleep_error = "No sleep data returned (unsupported ring, or nothing tracked yet)."
            else:
                latest = max(nights, key=lambda n: n.night_date)
                last_night = {
                    "date": latest.night_date.isoformat(),
                    "total_asleep_minutes": latest.total_asleep_minutes,
                    "deep_minutes": latest.deep_minutes,
                    "awake_minutes": latest.awake_minutes,
                    "quality_score": latest.quality_score,
                }
        except Exception as e:
            sleep_error = str(e)
        result["sleep"] = {
            "last_night": last_night, "error": sleep_error,
            "caveat": "Experimental: sleep.py's Big Data protocol parsing isn't verified against real hardware yet.",
        }

        stress_result, stress_error = None, None
        try:
            hr_log = await client.get_heart_rate_log(today)
            readings = _hr_log_readings(hr_log)
            label, proba = stress_predict.predict_stress(readings)
            stress_result = {"label": label, "probability": proba}
        except Exception as e:
            stress_error = str(e)
        result["stress"] = {"result": stress_result, "error": stress_error}

        injury_result, injury_error = None, None
        try:
            risk = await injury_predict.predict_injury_risk(client)
            injury_result = {
                "risk_band": risk.risk_band,
                "risk_probability": risk.risk_probability,
                "measured_fraction": risk.measured_fraction(),
            }
        except Exception as e:
            injury_error = str(e)
        result["injury_risk"] = {
            "result": injury_result, "error": injury_error,
            "caveat": "Illustrative only -- ring-derived proxies for a model with weak validated "
                      "precision even on real survey data. See README.",
        }

    return result


@app.get("/api/live")
def api_live():
    address = request.args.get("address") or app.config["RING_ADDRESS"]
    if not address:
        return jsonify({"error": "No ring address configured. Pass ?address=AA:BB:... or start with --address."}), 400
    try:
        snapshot = asyncio.run(_fetch_live_snapshot(address))
    except Exception as e:
        logger.exception("Live ring fetch failed")
        return jsonify({"error": f"Could not read from the ring: {e}"}), 502
    return jsonify(snapshot)


def main():
    parser = argparse.ArgumentParser(description="colmi_r02_client dashboard")
    parser.add_argument("--address", required=False, help="Default ring BLE address for /api/live")
    parser.add_argument("--db", default="ring_data.sqlite", help="Path to the SQLite database from `colmi_r02_client sync`")
    parser.add_argument("--port", type=int, default=5050)
    args = parser.parse_args()

    app.config["RING_ADDRESS"] = args.address
    app.config["DB_PATH"] = args.db
    app.run(debug=False, port=args.port)


if __name__ == "__main__":
    main()
