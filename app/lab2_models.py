"""Строгие контракты и нормализация времени без доверия к модели."""

from datetime import UTC, date, datetime
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentError(ValueError):
    """Публичный безопасный код ошибки, без пользовательских данных."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class NativeFunction(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    arguments: str = Field(max_length=10000)


class NativeCall(StrictModel):
    id: str = Field(min_length=1, max_length=200)
    type: Literal["function"]
    function: NativeFunction
    index: int | None = Field(default=None, ge=0)


ShortText = Annotated[str, Field(min_length=1, max_length=500)]


class WeatherArgs(StrictModel):
    city: Annotated[str, Field(min_length=1, max_length=100)]


class ScheduleArgs(StrictModel):
    date: Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]

    @field_validator("date")
    @classmethod
    def valid_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value


class ReminderArgs(StrictModel):
    text: ShortText
    remind_at: str

    @field_validator("remind_at")
    @classmethod
    def aware_date(cls, value: str) -> str:
        value = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(value)
        if "T" not in value or parsed.tzinfo is None:
            raise ValueError("timezone_required")
        return parsed.isoformat()


class Lesson(StrictModel):
    start: Annotated[str, Field(pattern=r"^\d{2}:\d{2}$")]
    end: Annotated[str, Field(pattern=r"^\d{2}:\d{2}$")]
    title: ShortText
    location: ShortText

    @field_validator("start", "end")
    @classmethod
    def valid_time(cls, value):
        from datetime import time

        time.fromisoformat(value)
        return value


class ScheduleResult(StrictModel):
    source: str = "Учебное вымышленное расписание"
    date: str
    lessons: list[Lesson]


class Candidate(StrictModel):
    name: str
    country: str
    admin1: str


class CurrentWeather(StrictModel):
    city: str
    country: str
    admin1: str
    temperature_c: float
    weather_code: int
    wind_speed: float
    wind_unit: Literal["km/h"] = "km/h"
    observed_at: str
    timezone: str


class WeatherResult(StrictModel):
    source: Literal["Open-Meteo"] = "Open-Meteo"
    status: Literal["ok", "not_found", "ambiguous"]
    candidates: list[Candidate] = Field(default_factory=list)
    current: CurrentWeather | None = None


class ReminderResult(StrictModel):
    id: str
    remind_at: str
    timezone: str
    created: bool


class WeekResult(StrictModel):
    timezone: str
    monday: str
    sunday: str
    days: list[ScheduleResult]


INPUT_MODELS = {
    "get_weather": WeatherArgs,
    "get_schedule": ScheduleArgs,
    "add_reminder": ReminderArgs,
}
OUTPUT_MODELS = {
    "get_weather": WeatherResult,
    "get_schedule": ScheduleResult,
    "add_reminder": ReminderResult,
}


def timezone_name(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        raise AgentError("unknown_timezone") from None


def normalize_reminder(args: ReminderArgs, zone: str | None, now: datetime) -> ReminderArgs:
    if zone is None:
        raise AgentError("timezone_required")
    tz = timezone_name(zone)
    supplied = datetime.fromisoformat(args.remind_at)
    # Проверяем именно предложенное локальное время и offset, а не молча сдвигаем его.
    local = supplied.replace(tzinfo=None)
    candidates = {
        local.replace(tzinfo=tz, fold=fold).astimezone(UTC)
        for fold in (0, 1)
        if local.replace(tzinfo=tz, fold=fold).astimezone(UTC).astimezone(tz).replace(tzinfo=None)
        == local
    }
    if len(candidates) != 1:
        raise AgentError("ambiguous_local_time")
    instant = next(iter(candidates))
    if supplied.astimezone(UTC) != instant:
        raise AgentError("timezone_offset_mismatch")
    if instant <= now.astimezone(UTC):
        raise AgentError("past_time")
    return args.model_copy(update={"remind_at": instant.astimezone(tz).isoformat()})
