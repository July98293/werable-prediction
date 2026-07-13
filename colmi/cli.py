"""
A python client for connecting to the Colmi R02 Smart ring
"""

import csv
import dataclasses
from collections import deque
from datetime import datetime, timezone, timedelta
from io import StringIO
from pathlib import Path
import logging
import time

import asyncclick as click
from bleak import BleakScanner

from colmi_r02_client.client import Client
from colmi_r02_client import steps, pretty_print, db, date_utils, hr, real_time, stress_predict, injury_predict

logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")


async def _collect_heart_rate_for(client: Client, duration_seconds: float) -> list[tuple[datetime, int]]:
    """
    Collect heart rate readings until they span at least `duration_seconds`
    of *reading timestamps* (not wall-clock loop time - the ring can take
    tens of seconds to return its first reading, so stopping on loop time
    alone reliably undershoots the model's required window). Gives up after
    3x the requested duration in case the ring stops responding.
    """
    readings: list[tuple[datetime, int]] = []
    start = time.monotonic()
    async for reading in client.stream_heart_rate():
        readings.append(reading)
        span = (readings[-1][0] - readings[0][0]).total_seconds()
        if span >= duration_seconds:
            break
        if time.monotonic() - start >= duration_seconds * 3:
            break
    return readings

logger = logging.getLogger(__name__)


@click.group()
@click.option("--debug/--no-debug", default=False)
@click.option(
    "--record/--no-record",
    default=False,
    help="Write all received packets to a file",
)
@click.option("--address", required=False, help="Bluetooth address")
@click.option("--name", required=False, help="Bluetooth name of the device, slower but will work on macOS")
@click.pass_context
async def cli_client(context: click.Context, debug: bool, record: bool, address: str | None, name: str | None) -> None:
    if (address is None and name is None) or (address is not None and name is not None):
        context.fail("You must pass either the address option(preferred) or the name option, but not both")

    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("bleak").setLevel(logging.INFO)

    record_to = None
    if record:
        now = int(time.time())
        captures = Path("captures")
        captures.mkdir(exist_ok=True)
        record_to = captures / Path(f"colmi_response_capture_{now}.bin")
        logger.info(f"Recording responses to {record_to}")

    if name is not None:
        devices = await BleakScanner.discover()
        found = next((x for x in devices if x.name == name), None)
        if found is None:
            context.fail("No device found with given name")
        address = found.address

    assert address

    client = Client(address, record_to=record_to)

    context.obj = client


@cli_client.command()
@click.pass_obj
async def info(client: Client) -> None:
    """Get device info and battery level"""

    async with client:
        print("device info", await client.get_device_info())
        print("battery:", await client.get_battery())


@cli_client.command()
@click.option(
    "--when",
    type=click.DateTime(),
    required=False,
    help="The date you want heart rate log / steps for (default: today)",
)
@click.pass_obj
async def status(client: Client, when: datetime | None) -> None:
    """Show battery, heart rate, SpO2, steps, and a stress prediction all at once"""

    if when is None:
        when = datetime.now(timezone.utc)

    try:
        stress_predict._load_model_and_scaler()  # fail fast, before we bother connecting
    except stress_predict.ModelNotTrained as e:
        raise click.ClickException(str(e))

    async with client:
        click.echo("=== Device ===")
        click.echo(f"Info: {await client.get_device_info()}")
        click.echo(f"Battery: {await client.get_battery()}")

        click.echo("\n=== Real-time ===")
        hr_reading = await client.get_realtime_reading(real_time.RealTimeReading.HEART_RATE)
        click.echo(f"Heart rate: {hr_reading if hr_reading else 'no reading, is the ring being worn?'}")
        spo2_reading = await client.get_realtime_reading(real_time.RealTimeReading.SPO2)
        click.echo(f"SpO2: {spo2_reading if spo2_reading else 'no reading, is the ring being worn?'}")

        click.echo("\n=== Heart rate log ===")
        hr_log = await client.get_heart_rate_log(when)
        if isinstance(hr_log, hr.HeartRateLog):
            recent = [(r, ts) for r, ts in hr_log.heart_rates_with_times() if r != 0][-5:]
            if recent:
                for reading, ts in recent:
                    click.echo(f"  {ts.strftime('%H:%M')}: {reading}")
            else:
                click.echo("  No data")
        else:
            click.echo("  No data")

        click.echo("\n=== Steps ===")
        steps_result = await client.get_steps(when)
        if isinstance(steps_result, steps.NoData):
            click.echo("  No data")
        else:
            click.echo(pretty_print.print_dataclasses(steps_result))

        click.echo(
            f"\n=== Stress prediction (collecting {stress_predict.MIN_WINDOW_SECONDS}s "
            "of heart rate, keep the ring worn) ==="
        )
        readings = await _collect_heart_rate_for(client, stress_predict.MIN_WINDOW_SECONDS)

    if not readings:
        click.echo("  Could not collect enough heart rate data for a prediction.")
        return

    try:
        label, proba = stress_predict.predict_stress(readings)
        click.echo(f"  {label} (stress probability: {proba:.2%})")
    except stress_predict.NotEnoughData as e:
        click.echo(f"  {e}")


@cli_client.command()
@click.option(
    "--target",
    type=click.DateTime(),
    required=True,
    help="The date you want logs for",
)
@click.pass_obj
async def get_heart_rate_log(client: Client, target: datetime) -> None:
    """Get heart rate for given date"""

    async with client:
        log = await client.get_heart_rate_log(target)
        print("Data:", log)
        if isinstance(log, hr.HeartRateLog):
            for reading, ts in log.heart_rates_with_times():
                if reading != 0:
                    print(f"{ts.strftime('%H:%M')}, {reading}")


@cli_client.command()
@click.option(
    "--when",
    type=click.DateTime(),
    required=False,
    help="The date and time you want to set the ring to",
)
@click.pass_obj
async def set_time(client: Client, when: datetime | None) -> None:
    """
    Set the time on the ring, required if you want to be able to interpret any of the logged data
    """

    if when is None:
        when = date_utils.now()
    async with client:
        await client.set_time(when)
    click.echo("Time set successfully")
    click.echo("Please ignore the unexpected packet. It's expectedly unexpected")


@cli_client.command()
@click.pass_obj
async def get_heart_rate_log_settings(client: Client) -> None:
    """Get heart rate log settings"""

    async with client:
        click.echo("heart rate log settings:")
        click.echo(await client.get_heart_rate_log_settings())


@cli_client.command()
@click.option("--enable/--disable", default=True, show_default=True, help="Logging status")
@click.option(
    "--interval",
    type=click.IntRange(0, 255),
    help="Interval in minutes to measure heart rate",
    default=60,
    show_default=True,
)
@click.pass_obj
async def set_heart_rate_log_settings(client: Client, enable: bool, interval: int) -> None:
    """Set heart rate log settings"""

    async with client:
        click.echo("Changing heart rate log settings")
        await client.set_heart_rate_log_settings(enable, interval)
        click.echo(await client.get_heart_rate_log_settings())
        click.echo("Done")


@cli_client.command()
@click.pass_obj
@click.argument("reading", nargs=1, type=click.Choice(list(real_time.REAL_TIME_MAPPING.keys())))
async def get_real_time(client: Client, reading: str) -> None:
    """Get any real time measurement (like heart rate or SPO2)"""
    async with client:
        click.echo("Starting reading, please wait.")
        reading_type = real_time.REAL_TIME_MAPPING[reading]
        result = await client.get_realtime_reading(reading_type)
        if result:
            click.echo(result)
        else:
            click.echo(f"Error, no {reading.replace('-', ' ')} detected. Is the ring being worn?")


STRESS_BUFFER_SECONDS = 2 * stress_predict.MIN_WINDOW_SECONDS  # keep enough history for the model's longest window, drop the rest


@cli_client.command()
@click.option(
    "--duration",
    type=click.IntRange(min=stress_predict.MIN_WINDOW_SECONDS),
    default=stress_predict.MIN_WINDOW_SECONDS,
    show_default=True,
    help="Seconds of heart rate history to collect before predicting "
    f"(minimum {stress_predict.MIN_WINDOW_SECONDS}, the model's longest rolling window). Ignored with --live.",
)
@click.option(
    "--live",
    is_flag=True,
    default=False,
    help="Keep monitoring and print a new prediction every --interval seconds "
    "instead of stopping after one",
)
@click.option(
    "--interval",
    type=click.IntRange(min=1),
    default=30,
    show_default=True,
    help="Seconds between predictions in --live mode",
)
@click.pass_obj
async def predict_stress(client: Client, duration: int, live: bool, interval: int) -> None:
    """Collect live heart rate from the ring and predict stress / no stress"""

    try:
        stress_predict._load_model_and_scaler()  # fail fast, before we bother connecting
    except stress_predict.ModelNotTrained as e:
        raise click.ClickException(str(e))

    if not live:
        click.echo(f"Collecting {duration}s of heart rate data from the ring, keep it worn...")
        async with client:
            readings = await _collect_heart_rate_for(client, duration)

        if not readings:
            raise click.ClickException("No heart rate readings received. Is the ring being worn?")

        click.echo(f"Collected {len(readings)} readings, running prediction...")
        try:
            label, proba = stress_predict.predict_stress(readings)
        except stress_predict.NotEnoughData as e:
            raise click.ClickException(str(e))

        click.echo(f"Prediction: {label} (stress probability: {proba:.2%})")
        return

    click.echo(
        f"Live monitoring: warming up (need {stress_predict.MIN_WINDOW_SECONDS}s of heart "
        f"rate), then a prediction every {interval}s. Keep the ring worn. Ctrl+C to stop."
    )
    buffer: deque[tuple[datetime, int]] = deque()
    async with client:
        last_prediction = time.monotonic()
        try:
            async for reading in client.stream_heart_rate():
                buffer.append(reading)
                cutoff = reading[0] - timedelta(seconds=STRESS_BUFFER_SECONDS)
                while buffer and buffer[0][0] < cutoff:
                    buffer.popleft()

                if time.monotonic() - last_prediction < interval:
                    continue
                last_prediction = time.monotonic()

                try:
                    label, proba = stress_predict.predict_stress(list(buffer))
                except stress_predict.NotEnoughData:
                    continue

                now_str = datetime.now().strftime("%H:%M:%S")
                click.echo(f"[{now_str}] {label} (stress probability: {proba:.2%})")
        except KeyboardInterrupt:
            click.echo("\nStopped monitoring.")


@cli_client.command()
@click.pass_obj
async def predict_injury(client: Client) -> None:
    """Estimate injury risk over the next 3 days from the last week of ring
    history (EXPERIMENTAL - see colmi_r02_client/injury_predict.py's module
    docstring for what this does and doesn't measure)"""

    try:
        pipeline, manifest = injury_predict._load_pipeline_and_manifest()
    except injury_predict.ModelNotTrained as e:
        raise click.ClickException(str(e))

    click.echo(
        f"Collecting {manifest['window']['input_sessions']} days of ring history "
        f"(heart rate, steps, sleep if available)..."
    )
    async with client:
        result = await injury_predict.predict_injury_risk(client)

    click.echo(f"\nInjury risk (next {manifest['window']['output_days']} days): {result.risk_band.upper()}")
    click.echo(f"Raw probability: {result.risk_probability:.2%} "
               f"(operating threshold: {result.threshold:.2%})")
    click.echo(f"Above threshold: {'yes' if result.above_operating_threshold else 'no'}")
    click.echo(f"Share of feature values from real ring data (rest imputed as missing): "
               f"{result.measured_fraction():.0%}")
    click.echo("\nPer-day values used (None = missing, imputed by the model):")
    for day_obs in result.days_used:
        click.echo(
            f"  {day_obs.day}: stress={day_obs.stress} sleep_hours={day_obs.sleep_duration_hours} "
            f"sleep_quality={day_obs.sleep_quality} fatigue_proxy={day_obs.fatigue}"
        )
    click.echo(
        "\nReminder: this is a reduced-feature, ring-only approximation of the injury "
        "model in 'injury 2/' -- treat it as illustrative, not a validated risk score."
    )


@cli_client.command()
@click.pass_obj
@click.option(
    "--when",
    type=click.DateTime(),
    required=False,
    help="The date you want steps for",
)
@click.option("--as-csv", is_flag=True, help="Print as CSV", default=False)
async def get_steps(client: Client, when: datetime | None = None, as_csv: bool = False) -> None:
    """Get step data"""

    if when is None:
        when = datetime.now(tz=timezone.utc)
    async with client:
        result = await client.get_steps(when)
        if isinstance(result, steps.NoData):
            click.echo("No results for day")
            return

        if not as_csv:
            click.echo(pretty_print.print_dataclasses(result))
        else:
            out = StringIO()
            writer = csv.DictWriter(out, fieldnames=[f.name for f in dataclasses.fields(steps.SportDetail)])
            writer.writeheader()
            for r in result:
                writer.writerow(dataclasses.asdict(r))
            click.echo(out.getvalue())


@cli_client.command()
@click.pass_obj
async def reboot(client: Client) -> None:
    """Reboot the ring"""

    async with client:
        await client.reboot()
        click.echo("Ring rebooted")


@cli_client.command()
@click.pass_obj
@click.option(
    "--command",
    type=click.IntRange(min=0, max=255),
    help="Raw command",
)
@click.option(
    "--subdata",
    type=str,
    help="Hex encoded subdata array, will be parsed into a bytearray",
)
@click.option("--replies", type=click.IntRange(min=0), default=0, help="How many reply packets to wait for")
async def raw(client: Client, command: int, subdata: str | None, replies: int) -> None:
    """Send the ring a raw command"""

    p_subdata = bytearray.fromhex(subdata) if subdata is not None else bytearray()

    async with client:
        results = await client.raw(command, p_subdata, replies)
        click.echo(results)


@cli_client.command()
@click.pass_obj
@click.option(
    "--db",
    "db_path",
    type=click.Path(writable=True, path_type=Path),
    help="Path to a directory or file to use as the database. If dir, then filename will be ring_data.sqlite",
)
@click.option(
    "--start",
    type=click.DateTime(),
    required=False,
    help="The date you want to start grabbing data from",
)
@click.option(
    "--end",
    type=click.DateTime(),
    required=False,
    help="The date you want to start grabbing data to",
)
async def sync(client: Client, db_path: Path | None, start: datetime | None, end: datetime | None) -> None:
    """
    Sync all data from the ring to a sqlite database

    Currently grabs:
        - heart rates
    """

    if db_path is None:
        db_path = Path.cwd()
    if db_path.is_dir():
        db_path /= Path("ring_data.sqlite")

    click.echo(f"Writing to {db_path}")
    with db.get_db_session(db_path) as session:
        if start is None:
            start = db.get_last_sync(session, client.address)
        else:
            start = date_utils.naive_to_aware(start)

        if start is None:
            start = date_utils.now() - timedelta(days=7)

        if end is None:
            end = date_utils.now()
        else:
            end = date_utils.naive_to_aware(end)

        click.echo(f"Syncing from {start} to {end}")

        async with client:
            fd = await client.get_full_data(start, end)
            db.full_sync(session, fd)
            when = datetime.now(tz=timezone.utc)
            click.echo("Ignore unexpect packet")
            await client.set_time(when)

    click.echo("Done")


DEVICE_NAME_PREFIXES = [
    "R01",
    "R02",
    "R03",
    "R04",
    "R05",
    "R06",
    "R07",
    "R09",
    "R10",
    "COLMI",
    "VK-5098",
    "MERLIN",
    "Hello Ring",
    "RING1",
    "boAtring",
    "TR-R02",
    "SE",
    "EVOLVEO",
    "GL-SR2",
    "Blaupunkt",
    "KSIX RING",
]


@click.group()
async def util():
    """Generic utilities for the R02 that don't need an address."""


@util.command()
@click.option("--all", is_flag=True, help="Print all devices, no name filtering", default=False)
async def scan(all: bool) -> None:
    """Scan for possible devices based on known prefixes and print the bluetooth address."""

    # TODO maybe bluetooth specific stuff like this should be in another package?
    devices = await BleakScanner.discover()

    if len(devices) > 0:
        click.echo("Found device(s)")
        click.echo(f"{'Name':>20}  | Address")
        click.echo("-" * 44)
        for d in devices:
            name = d.name
            if name and (all or any(name for p in DEVICE_NAME_PREFIXES if name.startswith(p))):
                click.echo(f"{name:>20}  |  {d.address}")
    else:
        click.echo("No devices found. Try moving the ring closer to computer")


@click.command()
@click.option("--address", required=False, help="Ring BLE address used for live snapshots (optional -- history works without it)")
@click.option(
    "--db",
    "db_path",
    type=click.Path(path_type=Path),
    default="ring_data.sqlite",
    help="Path to the SQLite database written by `colmi_r02_client sync`",
)
@click.option("--port", type=int, default=5050)
def dashboard(address: str | None, db_path: Path, port: int) -> None:
    """Launch the local ring dashboard (sleep/steps/HR/HRV/movement + stress + injury risk)."""
    from colmi_r02_client import webapp

    webapp.app.config["RING_ADDRESS"] = address
    webapp.app.config["DB_PATH"] = str(db_path)
    click.echo(f"Dashboard running at http://127.0.0.1:{port} (Ctrl+C to stop)")
    if address is None:
        click.echo("No --address given: history view works from the synced database, "
                    "but you'll need to enter an address in the page to pull a live snapshot.")
    webapp.app.run(debug=False, port=port)
