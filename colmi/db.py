from datetime import datetime, timezone
from pathlib import Path
import logging
from typing import Any

from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session, relationship
from sqlalchemy import select, UniqueConstraint, ForeignKey, create_engine, event, func, types
from sqlalchemy.engine import Engine, Dialect

from colmi_r02_client import hr, sleep, steps
from colmi_r02_client.client import FullData
from colmi_r02_client.date_utils import start_of_day, end_of_day

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class DateTimeInUTC(types.TypeDecorator):
    """
    TypeDecorator for sqlalchemy that will:

        1. make sure that you cannot save datetimes with no tzinfo/timezone
        2. convert any datetime to utc before saving it
        3. add the utc timezone to all datetimes retrieved from the database
    """

    impl = types.DateTime
    cache_ok = True

    def process_bind_param(self, value: Any | None, _dialect: Dialect) -> datetime | None:
        if value is None:
            return None

        if not isinstance(value, datetime):
            raise ValueError(f"Trying to store {value} that's not a datetime")

        if value.tzinfo is None:
            raise ValueError(f"Trying to store {value} with no timezone")

        return value.astimezone(timezone.utc)

    def process_result_value(self, value: Any | None, _dialect: Dialect) -> datetime | None:
        if value is None:
            return None

        if not isinstance(value, datetime):
            raise ValueError(f"Trying to add timezone to {value} that's not a datetime")

        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)


class Ring(Base):
    __tablename__ = "rings"
    __table_args__ = (UniqueConstraint("address"),)
    ring_id: Mapped[int] = mapped_column(primary_key=True)
    address: Mapped[str]
    heart_rates: Mapped[list["HeartRate"]] = relationship(back_populates="ring")
    sport_details: Mapped[list["SportDetail"]] = relationship(back_populates="ring")
    sleep_nights: Mapped[list["SleepNightRow"]] = relationship(back_populates="ring")
    syncs: Mapped[list["Sync"]] = relationship(back_populates="ring")


class Sync(Base):
    __tablename__ = "syncs"
    sync_id: Mapped[int] = mapped_column(primary_key=True)
    ring_id = mapped_column(ForeignKey("rings.ring_id"), nullable=False)
    timestamp = mapped_column(DateTimeInUTC(timezone=True), nullable=False)
    comment: Mapped[str | None]
    ring: Mapped["Ring"] = relationship(back_populates="syncs")
    heart_rates: Mapped[list["HeartRate"]] = relationship(back_populates="sync")
    sport_details: Mapped[list["SportDetail"]] = relationship(back_populates="sync")
    sleep_nights: Mapped[list["SleepNightRow"]] = relationship(back_populates="sync")


class HeartRate(Base):
    __tablename__ = "heart_rates"
    __table_args__ = (UniqueConstraint("ring_id", "timestamp"),)
    heart_rate_id: Mapped[int] = mapped_column(primary_key=True)
    reading: Mapped[int]
    timestamp = mapped_column(DateTimeInUTC(timezone=True), nullable=False)
    ring_id = mapped_column(ForeignKey("rings.ring_id"), nullable=False)
    ring: Mapped["Ring"] = relationship(back_populates="heart_rates")
    sync_id = mapped_column(ForeignKey("syncs.sync_id"), nullable=False)
    sync: Mapped["Sync"] = relationship(back_populates="heart_rates")


class SportDetail(Base):
    __tablename__ = "sport_details"
    __table_args__ = (UniqueConstraint("ring_id", "timestamp"),)
    sport_detail_id: Mapped[int] = mapped_column(primary_key=True)
    calories: Mapped[int]
    steps: Mapped[int]
    distance: Mapped[int]
    timestamp = mapped_column(DateTimeInUTC(timezone=True), nullable=False)
    ring_id = mapped_column(ForeignKey("rings.ring_id"), nullable=False)
    ring: Mapped["Ring"] = relationship(back_populates="sport_details")
    sync_id = mapped_column(ForeignKey("syncs.sync_id"), nullable=False)
    sync: Mapped["Sync"] = relationship(back_populates="sport_details")


class SleepNightRow(Base):
    """Per-night aggregate from sleep.py's Big Data sleep sync (EXPERIMENTAL,
    see that module's docstring). Stores the derived summary, not individual
    stage periods, since that's all colmi_r02_client/injury_predict.py and
    the dashboard need."""

    __tablename__ = "sleep_nights"
    __table_args__ = (UniqueConstraint("ring_id", "night_date"),)
    sleep_night_id: Mapped[int] = mapped_column(primary_key=True)
    night_date = mapped_column(DateTimeInUTC(timezone=True), nullable=False)
    start_minute: Mapped[int]
    end_minute: Mapped[int]
    total_asleep_minutes: Mapped[int]
    deep_minutes: Mapped[int]
    awake_minutes: Mapped[int]
    quality_score: Mapped[float | None]
    ring_id = mapped_column(ForeignKey("rings.ring_id"), nullable=False)
    ring: Mapped["Ring"] = relationship(back_populates="sleep_nights")
    sync_id = mapped_column(ForeignKey("syncs.sync_id"), nullable=False)
    sync: Mapped["Sync"] = relationship(back_populates="sleep_nights")


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
    """Enable actual foreign key checks in sqlite on every connection to the database"""

    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def get_db_session(path: Path | None = None) -> Session:
    """
    Return a live db session with all tables created.

    TODO: probably not default to in memory... that's just useful for testing
    """

    url = "sqlite:///"
    if path is not None:
        url = url + str(path)
    else:
        logger.info("Using in memory sqlite database. Data will be lost after program exits")
        url = url + ":memory:"
    engine = create_engine(url, echo=False)
    Base.metadata.create_all(engine)
    return Session(engine)


def create_or_find_ring(session: Session, address: str) -> Ring:
    ring = session.scalars(select(Ring).where(Ring.address == address)).one_or_none()
    if ring is not None:
        return ring

    ring = Ring(address=address)
    session.add(ring)
    session.commit()  # not sure this should be here tbh
    return ring


def full_sync(session: Session, data: FullData) -> None:
    """
    TODO:
        - grab battery
    """

    ring = create_or_find_ring(session, data.address)
    sync = Sync(ring=ring, timestamp=datetime.now(tz=timezone.utc))
    session.add(sync)

    _add_heart_rate(sync, ring, data, session)
    _add_sport_details(sync, ring, data, session)
    _add_sleep_nights(sync, ring, data, session)
    session.commit()


def _add_heart_rate(sync: Sync, ring: Ring, data: FullData, session: Session) -> None:
    logger.info(f"Adding {len(data.heart_rates)} days of heart rates")
    for log in data.heart_rates:
        if isinstance(log, hr.NoData):
            logger.info("No heart rate data for date")
            continue

        existing = {}
        for heart_rate in session.scalars(
            select(HeartRate)
            .where(HeartRate.timestamp >= start_of_day(log.timestamp))
            .where(HeartRate.timestamp <= end_of_day(log.timestamp))
        ):
            existing[heart_rate.timestamp] = heart_rate.reading

        for reading, timestamp in log.heart_rates_with_times():
            if reading == 0:
                continue

            if x := existing.get(timestamp):
                if x != reading:
                    logger.warning(f"Inconsistent data detected! {timestamp} is {x} in db but got {reading} from ring")
            else:
                h = HeartRate(reading=reading, timestamp=timestamp, ring=ring, sync=sync)
                session.add(h)


def _add_sport_details(sync: Sync, ring: Ring, data: FullData, session: Session) -> None:
    logger.info(f"Adding {len(data.sport_details)} days of sport details")
    sport_detail_logs: list[steps.SportDetail] = []
    for slog in data.sport_details:
        if isinstance(slog, steps.NoData):
            logger.info("No step data for date")
        else:
            sport_detail_logs.extend(slog)
    if len(sport_detail_logs) == 0:
        return
    start = min([sport_detail.timestamp for sport_detail in sport_detail_logs])
    end = max([sport_detail.timestamp for sport_detail in sport_detail_logs])

    existing_sport_logs = {}
    for sport_detail_record in session.scalars(
        select(SportDetail)
        .where(SportDetail.timestamp >= start_of_day(start))
        .where(SportDetail.timestamp <= end_of_day(end))
    ):
        existing_sport_logs[sport_detail_record.timestamp] = sport_detail_record

    for sport_detail in sport_detail_logs:
        if x := existing_sport_logs.get(sport_detail.timestamp):
            x.calories = sport_detail.calories
            x.steps = sport_detail.steps
            x.distance = sport_detail.distance
            session.add(x)
        else:
            s = SportDetail(
                calories=sport_detail.calories,
                steps=sport_detail.steps,
                distance=sport_detail.distance,
                timestamp=sport_detail.timestamp,
                ring=ring,
                sync=sync,
            )
            session.add(s)


def _add_sleep_nights(sync: Sync, ring: Ring, data: FullData, session: Session) -> None:
    logger.info(f"Adding {len(data.sleep_nights)} nights of sleep data")
    if not data.sleep_nights:
        return

    existing = {}
    for row in session.scalars(select(SleepNightRow).where(SleepNightRow.ring_id == ring.ring_id)):
        existing[row.night_date] = row

    for night in data.sleep_nights:
        night_date = datetime(night.night_date.year, night.night_date.month, night.night_date.day, tzinfo=timezone.utc)
        if x := existing.get(night_date):
            x.start_minute = night.start_minute
            x.end_minute = night.end_minute
            x.total_asleep_minutes = night.total_asleep_minutes
            x.deep_minutes = night.deep_minutes
            x.awake_minutes = night.awake_minutes
            x.quality_score = night.quality_score
            session.add(x)
        else:
            session.add(SleepNightRow(
                night_date=night_date,
                start_minute=night.start_minute,
                end_minute=night.end_minute,
                total_asleep_minutes=night.total_asleep_minutes,
                deep_minutes=night.deep_minutes,
                awake_minutes=night.awake_minutes,
                quality_score=night.quality_score,
                ring=ring,
                sync=sync,
            ))


def get_last_sync(session: Session, ring_address: str) -> datetime | None:
    return session.scalars(select(func.max(Sync.timestamp)).join(Ring).where(Ring.address == ring_address)).one_or_none()


def get_daily_history(session: Session, ring_address: str, since: datetime) -> list[dict]:
    """One row per calendar day since `since`: average/min/max heart rate,
    total steps/calories/distance, and that night's sleep summary if synced.
    Used by webapp.py's history endpoint to drive the dashboard's week view.
    """
    ring = session.scalars(select(Ring).where(Ring.address == ring_address)).one_or_none()
    if ring is None:
        return []

    hr_rows = session.scalars(
        select(HeartRate).where(HeartRate.ring_id == ring.ring_id).where(HeartRate.timestamp >= since)
    ).all()
    sport_rows = session.scalars(
        select(SportDetail).where(SportDetail.ring_id == ring.ring_id).where(SportDetail.timestamp >= since)
    ).all()
    sleep_rows = session.scalars(
        select(SleepNightRow).where(SleepNightRow.ring_id == ring.ring_id).where(SleepNightRow.night_date >= since)
    ).all()

    by_day: dict[str, dict] = {}

    def _day(day_key: str) -> dict:
        return by_day.setdefault(day_key, {
            "date": day_key, "hr_readings": [], "steps": 0, "calories": 0, "distance": 0,
            "sleep_hours": None, "sleep_quality": None,
        })

    for row in hr_rows:
        _day(row.timestamp.date().isoformat())["hr_readings"].append(row.reading)
    for row in sport_rows:
        d = _day(row.timestamp.date().isoformat())
        d["steps"] += row.steps
        d["calories"] += row.calories
        d["distance"] += row.distance
    for row in sleep_rows:
        d = _day(row.night_date.date().isoformat())
        d["sleep_hours"] = round(row.total_asleep_minutes / 60.0, 2)
        d["sleep_quality"] = row.quality_score

    days = []
    for day_key in sorted(by_day):
        d = by_day[day_key]
        readings = d.pop("hr_readings")
        d["hr_avg"] = round(sum(readings) / len(readings), 1) if readings else None
        d["hr_min"] = min(readings) if readings else None
        d["hr_max"] = max(readings) if readings else None
        days.append(d)
    return days
