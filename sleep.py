"""Sleep data over the ring's "Big Data" BLE protocol.

EXPERIMENTAL / UNVERIFIED AGAINST REAL HARDWARE. Unlike every other module in
this client, this one was not built by capturing and inspecting real packets
from a ring (see MYSTERIES.md / --record for how the rest of this client was
reverse engineered). It's implemented from third-party protocol notes for the
"Big Data" characteristic (colmi.puxtril.com/bigdata/, cross-checked against
Gadgetbridge's Colmi support, which independently confirms the service/
characteristic UUIDs and the general magic-byte/length/CRC16 framing but also
shows this exact multi-packet reassembly is a real source of bugs even in
that mature, hardware-tested project). Specifically uncertain / unverified
here:

    - The CRC16 algorithm is undocumented anywhere found; this module does
      NOT validate it on receive, and sends the documented placeholder
      (0xFFFF) on the outgoing empty-payload request rather than computing one.
    - The exact per-day payload layout below (days_ago, byte_count, start/end
      minute-of-day, then repeating (stage, duration) pairs) is inferred from
      partial docs, not a confirmed byte-for-byte spec.

Because of this, every parse path here is defensive: anything that doesn't
fit the expected shape raises SleepParseError rather than returning a
plausible-looking but possibly-wrong result, so callers (see
colmi_r02_client/injury_predict.py) can fall back to "no data" instead of
silently feeding bad numbers into a prediction. Treat this module as a
starting point to validate/fix against a real ring capture, not a finished,
trustworthy implementation.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from enum import IntEnum

BIG_DATA_SERVICE_UUID = "de5bf728-d711-4e47-af26-65e3012a5dc7"
BIG_DATA_WRITE_CHAR_UUID = "de5bf72a-d711-4e47-af26-65e3012a5dc7"
BIG_DATA_NOTIFY_CHAR_UUID = "de5bf729-d711-4e47-af26-65e3012a5dc7"

BIG_DATA_MAGIC = 188  # 0xBC
DATA_ID_SLEEP = 39

HEADER_LEN = 6  # magic, data_id, length(2, LE), crc16(2, LE)
DAY_HEADER_LEN = 6  # days_ago(1), byte_count(1), start_min(2, LE), end_min(2, LE)
PERIOD_LEN = 2  # stage(1), duration_minutes(1)


class SleepParseError(ValueError):
    """The Big Data sleep payload didn't match the (unverified) expected layout."""


class SleepStage(IntEnum):
    NO_DATA = 0
    ERROR = 1
    LIGHT = 2
    DEEP = 3
    AWAKE = 5


@dataclass
class SleepPeriod:
    stage: SleepStage
    duration_minutes: int


@dataclass
class SleepNight:
    night_date: date
    start_minute: int
    """Minutes from midnight the tracked sleep started."""
    end_minute: int
    periods: list[SleepPeriod]

    @property
    def total_asleep_minutes(self) -> int:
        return sum(p.duration_minutes for p in self.periods if p.stage in (SleepStage.LIGHT, SleepStage.DEEP))

    @property
    def deep_minutes(self) -> int:
        return sum(p.duration_minutes for p in self.periods if p.stage == SleepStage.DEEP)

    @property
    def awake_minutes(self) -> int:
        return sum(p.duration_minutes for p in self.periods if p.stage == SleepStage.AWAKE)

    @property
    def quality_score(self) -> float | None:
        """Rough 1-10 proxy for the SoccerMon `sleep_quality` self-report scale,
        built from sleep architecture instead of a subjective rating: reward a
        higher deep-sleep share and fewer/shorter awakenings. Not validated
        against any real subjective-vs-ring-quality comparison - see the
        module docstring and injury_predict.py's caveats.
        """
        total = self.total_asleep_minutes
        if total == 0:
            return None
        deep_ratio = self.deep_minutes / total
        n_awakenings = sum(1 for p in self.periods if p.stage == SleepStage.AWAKE)
        awake_penalty = min(n_awakenings * 0.5, 4.0)
        score = 4.0 + deep_ratio * 10.0 - awake_penalty
        return max(1.0, min(10.0, score))


class NoData:
    """Returned when the ring has no sleep data for the requested range, or
    when the Big Data service isn't available/didn't respond in time."""


def sleep_request_packet(data_id: int = DATA_ID_SLEEP) -> bytearray:
    """6-byte Big Data request: magic, data_id, length=0x0000, crc16=0xFFFF
    (per colmi.puxtril.com/bigdata/, the CRC field on empty-payload requests
    is documented as this fixed placeholder, not a computed checksum).
    """
    return bytearray([BIG_DATA_MAGIC, data_id, 0x00, 0x00, 0xFF, 0xFF])


def parse_big_data_header(packet: bytearray) -> tuple[int, int]:
    """Returns (data_id, declared_payload_length) from the first notification
    of a Big Data response. Raises SleepParseError if it doesn't look like a
    Big Data header at all (wrong magic byte).
    """
    if len(packet) < HEADER_LEN or packet[0] != BIG_DATA_MAGIC:
        raise SleepParseError(f"Not a Big Data packet (magic byte): {packet!r}")
    data_id = packet[1]
    length = packet[2] | (packet[3] << 8)
    return data_id, length


def parse_sleep_payload(payload: bytearray, today: date) -> list[SleepNight]:
    """payload: the reassembled Big Data response body (everything after the
    6-byte header, concatenated across all notification chunks). Layout
    (unverified, see module docstring):

        n_days (1 byte)
        per day:
            days_ago (1 byte)
            byte_count (1 byte)   -- length in bytes of this day's period list
            start_minute (uint16 LE) -- minutes from midnight
            end_minute (uint16 LE)
            periods: byte_count // 2 x (stage: uint8, duration_minutes: uint8)
    """
    if len(payload) < 1:
        raise SleepParseError("Empty sleep payload")

    offset = 0
    n_days = payload[offset]
    offset += 1

    nights: list[SleepNight] = []
    for _ in range(n_days):
        if offset + DAY_HEADER_LEN > len(payload):
            raise SleepParseError(f"Truncated day header at offset {offset} (payload len {len(payload)})")

        days_ago = payload[offset]
        byte_count = payload[offset + 1]
        start_minute = payload[offset + 2] | (payload[offset + 3] << 8)
        end_minute = payload[offset + 4] | (payload[offset + 5] << 8)
        offset += DAY_HEADER_LEN

        if offset + byte_count > len(payload):
            raise SleepParseError(f"Truncated period list at offset {offset} (need {byte_count} bytes)")
        if byte_count % PERIOD_LEN != 0:
            raise SleepParseError(f"Period byte_count {byte_count} is not a multiple of {PERIOD_LEN}")

        periods = []
        for i in range(offset, offset + byte_count, PERIOD_LEN):
            try:
                stage = SleepStage(payload[i])
            except ValueError as e:
                raise SleepParseError(f"Unknown sleep stage byte {payload[i]} at offset {i}") from e
            duration_minutes = payload[i + 1]
            periods.append(SleepPeriod(stage=stage, duration_minutes=duration_minutes))
        offset += byte_count

        nights.append(SleepNight(
            night_date=today - timedelta(days=days_ago),
            start_minute=start_minute,
            end_minute=end_minute,
            periods=periods,
        ))

    return nights


class BigDataSleepReassembler:
    """Accumulates notification chunks from the Big Data notify characteristic
    for a single in-flight sleep request until the header-declared payload
    length is reached, then hands the full payload to parse_sleep_payload.

    The first chunk of a transfer carries the 6-byte header (magic/data_id/
    length/crc); every chunk after that is raw continuation payload with no
    header, appended as-is (this matches the framing implied by Gadgetbridge's
    Colmi logs, e.g. "got 20 bytes while expecting 147+6" -- 20 bytes was
    already header + partial payload).
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._declared_length: int | None = None
        self._payload = bytearray()

    def feed(self, chunk: bytearray) -> bytearray | None:
        """Returns the complete reassembled payload once enough bytes have
        arrived, else None. Raises SleepParseError on a malformed first chunk.
        """
        if self._declared_length is None:
            data_id, length = parse_big_data_header(chunk)
            if data_id != DATA_ID_SLEEP:
                raise SleepParseError(f"Expected sleep data_id {DATA_ID_SLEEP}, got {data_id}")
            self._declared_length = length
            self._payload.extend(chunk[HEADER_LEN:])
        else:
            self._payload.extend(chunk)

        if len(self._payload) >= self._declared_length:
            result = self._payload[: self._declared_length]
            self.reset()
            return result
        return None
