import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
import dataclasses
from dataclasses import dataclass
import logging
from pathlib import Path
from types import TracebackType
from typing import Any

from bleak import BleakClient, BleakScanner
from bleak.backends.characteristic import BleakGATTCharacteristic

from colmi_r02_client import battery, date_utils, steps, set_time, blink_twice, hr, hr_settings, packet, reboot, real_time, sleep

UART_SERVICE_UUID = "6E40FFF0-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

DEVICE_INFO_UUID = "0000180A-0000-1000-8000-00805F9B34FB"
DEVICE_HW_UUID = "00002A27-0000-1000-8000-00805F9B34FB"
DEVICE_FW_UUID = "00002A26-0000-1000-8000-00805F9B34FB"

logger = logging.getLogger(__name__)


def empty_parse(_packet: bytearray) -> None:
    """Used for commands that we expect a response, but there's nothing in the response"""
    return None


def log_packet(packet: bytearray) -> None:
    print("received: ", packet)


# TODO move this maybe?
@dataclass
class FullData:
    address: str
    heart_rates: list[hr.HeartRateLog | hr.NoData]
    sport_details: list[list[steps.SportDetail] | steps.NoData]
    sleep_nights: list[sleep.SleepNight] = dataclasses.field(default_factory=list)
    """Empty if the ring doesn't support the Big Data sleep sync or the fetch failed
    (see sleep.py's module docstring - this is experimental)."""


COMMAND_HANDLERS: dict[int, Callable[[bytearray], Any]] = {
    battery.CMD_BATTERY: battery.parse_battery,
    real_time.CMD_START_REAL_TIME: real_time.parse_real_time_reading,
    real_time.CMD_STOP_REAL_TIME: empty_parse,
    steps.CMD_GET_STEP_SOMEDAY: steps.SportDetailParser().parse,
    hr.CMD_READ_HEART_RATE: hr.HeartRateLogParser().parse,
    set_time.CMD_SET_TIME: empty_parse,
    hr_settings.CMD_HEART_RATE_LOG_SETTINGS: hr_settings.parse_heart_rate_log_settings,
}
"""
TODO put these somewhere nice

These are commands that we expect to have a response returned for
they must accept a packet as bytearray and then return a value to be put
in the queue for that command type
NOTE: if the value returned is None, it is not added to the queue, this is to support
multi packet messages where the parser has state
"""


class Client:
    def __init__(self, address: str, record_to: Path | None = None):
        self.address = address
        self.bleak_client: BleakClient | None = None
        self.queues: dict[int, asyncio.Queue] = {cmd: asyncio.Queue() for cmd in COMMAND_HANDLERS}
        self.record_to = record_to
        self.big_data_write_char: BleakGATTCharacteristic | None = None
        """None if this ring doesn't expose the Big Data GATT service (see sleep.py)."""
        self._big_data_reassembler = sleep.BigDataSleepReassembler()
        self._big_data_queue: asyncio.Queue = asyncio.Queue()

    async def __aenter__(self) -> "Client":
        logger.info(f"Connecting to {self.address}")
        await self.connect()
        logger.info("Connected!")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        logger.info("Disconnecting")
        if exc_val is not None:
            logger.error("had an error")
        await self.disconnect()

    async def connect(self, max_attempts: int = 6, retry_delay: float = 3.0):
        # On Windows, BleakClient(address).connect() often raises
        # BleakDeviceNotFoundError because the WinRT backend needs a fresh
        # scan result, not just a bare address string. Scanning for the
        # device right before connecting (and connecting to that BLEDevice)
        # is the documented workaround and also works fine on other OSes.
        # The ring also has a low advertising duty cycle and the Windows BLE
        # stack can flake out mid-connect ("Unreachable"), so each attempt
        # re-scans for a fresh device reference rather than reusing a
        # possibly-stale one across retries.
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            device = await BleakScanner.find_device_by_address(self.address, timeout=10.0)
            if device is None:
                last_error = RuntimeError(f"Could not find a BLE device at {self.address}")
                logger.warning(
                    f"Ring not found (attempt {attempt}/{max_attempts}), "
                    "keep it nearby and awake, retrying..."
                )
                await asyncio.sleep(retry_delay)
                continue

            self.bleak_client = BleakClient(device)
            try:
                # The default bleak connect timeout (10s) is too short for
                # this ring's GATT service discovery on Windows.
                await self.bleak_client.connect(timeout=20.0)
                last_error = None
                break
            except Exception as e:  # bleak raises several exception types here (TimeoutError, BleakError, ...)
                last_error = e
                logger.warning(f"Connect attempt {attempt}/{max_attempts} failed ({e!r}), retrying...")
                await asyncio.sleep(retry_delay)

        if last_error is not None:
            raise RuntimeError(
                f"Could not connect to the ring at {self.address} after {max_attempts} attempts "
                f"(last error: {last_error!r}). Make sure it's nearby, awake, and not connected "
                "to another app (e.g. close the phone app / turn off its Bluetooth)."
            ) from last_error

        nrf_uart_service = self.bleak_client.services.get_service(UART_SERVICE_UUID)
        assert nrf_uart_service
        rx_char = nrf_uart_service.get_characteristic(UART_RX_CHAR_UUID)
        assert rx_char
        self.rx_char = rx_char

        await self.bleak_client.start_notify(UART_TX_CHAR_UUID, self._handle_tx)

        big_data_service = self.bleak_client.services.get_service(sleep.BIG_DATA_SERVICE_UUID)
        if big_data_service is not None:
            write_char = big_data_service.get_characteristic(sleep.BIG_DATA_WRITE_CHAR_UUID)
            notify_char = big_data_service.get_characteristic(sleep.BIG_DATA_NOTIFY_CHAR_UUID)
            if write_char is not None and notify_char is not None:
                self.big_data_write_char = write_char
                await self.bleak_client.start_notify(notify_char, self._handle_big_data)
            else:
                logger.info("Big Data service present but missing expected characteristics; sleep data unavailable")
        else:
            logger.info("Ring doesn't expose the Big Data BLE service; sleep data unavailable")

    async def disconnect(self):
        await self.bleak_client.disconnect()

    def _handle_tx(self, _: BleakGATTCharacteristic, packet: bytearray) -> None:
        """Bleak callback that handles new packets from the ring."""

        logger.info(f"Received packet {packet}")

        assert len(packet) == 16, f"Packet is the wrong length {packet}"
        packet_type = packet[0]
        assert packet_type < 127, f"Packet has error bit set {packet}"

        if packet_type in COMMAND_HANDLERS:
            result = COMMAND_HANDLERS[packet_type](packet)
            if result is not None:
                self.queues[packet_type].put_nowait(result)
            else:
                logger.debug(f"No result returned from parser for {packet_type}")
        else:
            logger.warning(f"Did not expect this packet: {packet}")

        if self.record_to is not None:
            with self.record_to.open("ab") as f:
                f.write(packet)
                f.write(b"\n")

    def _handle_big_data(self, _: BleakGATTCharacteristic, packet: bytearray) -> None:
        """Bleak callback for the Big Data notify characteristic (see sleep.py).
        Unlike _handle_tx, chunks here are NOT fixed-length, so they're fed to
        the reassembler until a full payload is available.
        """
        logger.info(f"Received Big Data packet ({len(packet)} bytes): {packet}")
        try:
            payload = self._big_data_reassembler.feed(bytearray(packet))
        except sleep.SleepParseError as e:
            logger.warning(f"Big Data reassembly failed, dropping in-flight transfer: {e}")
            self._big_data_reassembler.reset()
            self._big_data_queue.put_nowait(e)
            return

        if payload is not None:
            self._big_data_queue.put_nowait(payload)

        if self.record_to is not None:
            with self.record_to.open("ab") as f:
                f.write(packet)
                f.write(b"\n")

    async def send_packet(self, packet: bytearray) -> None:
        logger.debug(f"Sending packet: {packet}")
        await self.bleak_client.write_gatt_char(self.rx_char, packet, response=False)

    async def send_big_data_packet(self, packet: bytearray) -> None:
        if self.big_data_write_char is None:
            raise RuntimeError("This ring doesn't expose the Big Data BLE service")
        logger.debug(f"Sending Big Data packet: {packet}")
        await self.bleak_client.write_gatt_char(self.big_data_write_char, packet, response=False)

    async def get_battery(self) -> battery.BatteryInfo:
        await self.send_packet(battery.BATTERY_PACKET)
        result = await self.queues[battery.CMD_BATTERY].get()
        assert isinstance(result, battery.BatteryInfo)
        return result

    async def _poll_real_time_reading(self, reading_type: real_time.RealTimeReading) -> list[int] | None:
        start_packet = real_time.get_start_packet(reading_type)
        stop_packet = real_time.get_stop_packet(reading_type)

        await self.send_packet(start_packet)

        valid_readings: list[int] = []
        error = False
        tries = 0
        while len(valid_readings) < 6 and tries < 20:
            try:
                data: real_time.Reading | real_time.ReadingError = await asyncio.wait_for(
                    self.queues[real_time.CMD_START_REAL_TIME].get(),
                    timeout=2,
                )
                if isinstance(data, real_time.ReadingError):
                    error = True
                    break
                if data.value != 0:
                    valid_readings.append(data.value)
            except TimeoutError:
                tries += 1

        await self.send_packet(stop_packet)
        if error:
            return None
        return valid_readings

    async def get_realtime_reading(self, reading_type: real_time.RealTimeReading) -> list[int] | None:
        return await self._poll_real_time_reading(reading_type)

    async def stream_heart_rate(self, poll_interval: float = 2.0):
        """
        Continuously poll the ring for real-time heart rate, yielding
        (timestamp, bpm) readings as they arrive, forever - the caller
        decides when to stop consuming. Each poll only returns a short burst
        of readings (see _poll_real_time_reading), so this just re-polls
        back-to-back. Used to build up history for the stress predictor's
        rolling-window features, either as a single batch or continuously.
        """
        while True:
            values = await self.get_realtime_reading(real_time.RealTimeReading.HEART_RATE)
            if values:
                now = datetime.now(timezone.utc)
                for v in values:
                    yield now, v
            await asyncio.sleep(poll_interval)

    async def set_time(self, ts: datetime) -> None:
        await self.send_packet(set_time.set_time_packet(ts))

    async def blink_twice(self) -> None:
        await self.send_packet(blink_twice.BLINK_TWICE_PACKET)

    async def get_device_info(self) -> dict[str, str]:
        client = self.bleak_client
        data = {}
        device_info_service = client.services.get_service(DEVICE_INFO_UUID)
        assert device_info_service

        hw_info_char = device_info_service.get_characteristic(DEVICE_HW_UUID)
        assert hw_info_char
        hw_version = await client.read_gatt_char(hw_info_char)
        data["hw_version"] = hw_version.decode("utf-8")

        fw_info_char = device_info_service.get_characteristic(DEVICE_FW_UUID)
        assert fw_info_char
        fw_version = await client.read_gatt_char(fw_info_char)
        data["fw_version"] = fw_version.decode("utf-8")

        return data

    async def get_heart_rate_log(self, target: datetime | None = None) -> hr.HeartRateLog | hr.NoData:
        if target is None:
            target = date_utils.start_of_day(date_utils.now())
        await self.send_packet(hr.read_heart_rate_packet(target))
        return await asyncio.wait_for(
            self.queues[hr.CMD_READ_HEART_RATE].get(),
            timeout=2,
        )

    async def get_heart_rate_log_settings(self) -> hr_settings.HeartRateLogSettings:
        await self.send_packet(hr_settings.READ_HEART_RATE_LOG_SETTINGS_PACKET)
        return await asyncio.wait_for(
            self.queues[hr_settings.CMD_HEART_RATE_LOG_SETTINGS].get(),
            timeout=2,
        )

    async def set_heart_rate_log_settings(self, enabled: bool, interval: int) -> None:
        await self.send_packet(hr_settings.hr_log_settings_packet(hr_settings.HeartRateLogSettings(enabled, interval)))

        # clear response from queue as it's unused and wrong
        await asyncio.wait_for(
            self.queues[hr_settings.CMD_HEART_RATE_LOG_SETTINGS].get(),
            timeout=2,
        )

    async def get_steps(self, target: datetime, today: datetime | None = None) -> list[steps.SportDetail] | steps.NoData:
        if today is None:
            today = datetime.now(timezone.utc)

        if target.tzinfo != timezone.utc:
            logger.info("Converting target time to utc")
            target = target.astimezone(tz=timezone.utc)

        days = (today.date() - target.date()).days
        logger.debug(f"Looking back {days} days")

        await self.send_packet(steps.read_steps_packet(days))
        return await asyncio.wait_for(
            self.queues[steps.CMD_GET_STEP_SOMEDAY].get(),
            timeout=2,
        )

    async def get_sleep_data(self, timeout: float = 5.0) -> list[sleep.SleepNight] | sleep.NoData:
        """Fetch recent nights of sleep-stage data via the Big Data protocol.

        EXPERIMENTAL, see sleep.py's module docstring: the byte layout isn't
        verified against a real device. Any failure (unsupported ring,
        timeout, malformed response) returns NoData rather than raising, so
        callers can treat "no sleep data" uniformly regardless of the reason.
        """
        if self.big_data_write_char is None:
            return sleep.NoData()

        await self.send_big_data_packet(sleep.sleep_request_packet())
        try:
            result = await asyncio.wait_for(self._big_data_queue.get(), timeout=timeout)
        except TimeoutError:
            logger.info("Timed out waiting for sleep data")
            return sleep.NoData()

        if isinstance(result, sleep.SleepParseError):
            return sleep.NoData()
        return sleep.parse_sleep_payload(result, date_utils.now().date())

    async def reboot(self) -> None:
        await self.send_packet(reboot.REBOOT_PACKET)

    async def raw(self, command: int, subdata: bytearray, replies: int = 0) -> list[bytearray]:
        p = packet.make_packet(command, subdata)
        await self.send_packet(p)

        results = []
        while replies > 0:
            data: bytearray = await asyncio.wait_for(
                self.queues[command].get(),
                timeout=2,
            )
            results.append(data)
            replies -= 1

        return results

    async def get_full_data(self, start: datetime, end: datetime) -> FullData:
        """
        Fetches all data from the ring between start and end. Useful for syncing.
        """
        heart_rate_logs = []
        sport_detail_logs = []
        for d in date_utils.dates_between(start, end):
            heart_rate_logs.append(await self.get_heart_rate_log(d))
            sport_detail_logs.append(await self.get_steps(d))

        try:
            sleep_nights = await self.get_sleep_data()
            if isinstance(sleep_nights, sleep.NoData):
                sleep_nights = []
        except Exception as e:  # sleep.py is experimental; never let it break a sync
            logger.warning(f"Sleep data fetch failed during sync, continuing without it: {e}")
            sleep_nights = []

        return FullData(self.address, heart_rates=heart_rate_logs, sport_details=sport_detail_logs,
                         sleep_nights=sleep_nights)
