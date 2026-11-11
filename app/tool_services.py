"""Фиксированные источники данных и ограниченные HTTP-запросы."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import aiohttp
from pydantic import ValidationError

from app.lab2_models import (
    AgentError,
    Candidate,
    CurrentWeather,
    ScheduleArgs,
    ScheduleResult,
    WeatherArgs,
    WeatherResult,
    WeekResult,
)


class ScheduleService:
    def __init__(self, path: str, zone: str, clock=None):
        self.path = Path(path)
        self.zone = zone
        self.clock = clock or (lambda: datetime.now(UTC))

    def day(self, value: str) -> ScheduleResult:
        args = ScheduleArgs(date=value)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return ScheduleResult(date=args.date, lessons=data["days"].get(args.date, []))
        except (OSError, ValueError, KeyError, TypeError, ValidationError):
            raise AgentError("schedule_unavailable") from None

    def week(self) -> WeekResult:
        today = self.clock().astimezone(ZoneInfo(self.zone)).date()
        monday = today - timedelta(days=today.weekday())
        return WeekResult(
            timezone=self.zone,
            monday=monday.isoformat(),
            sunday=(monday + timedelta(days=6)).isoformat(),
            days=[self.day((monday + timedelta(days=n)).isoformat()) for n in range(7)],
        )


class WeatherService:
    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    async def fetch(self, url: str, params: dict) -> dict:
        # Только два фиксированных HTTPS endpoint; не более одного повтора чтения.
        for attempt in range(2):
            try:
                async with (
                    aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as session,
                    session.get(url, params=params, allow_redirects=False) as response,
                ):
                    if response.status != 200:
                        raise AgentError("weather_http_error")
                    raw = bytearray()
                    async for chunk in response.content.iter_chunked(16384):
                        raw.extend(chunk)
                        if len(raw) > 200_000:
                            raise AgentError("weather_bad_response")
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        raise AgentError("weather_bad_response")
                    return data
            except (TimeoutError, aiohttp.ClientError):
                if attempt:
                    raise AgentError("weather_unavailable") from None
                await asyncio.sleep(0.1)
            except (ValueError, TypeError):
                raise AgentError("weather_bad_response") from None
        raise AgentError("weather_unavailable")

    async def weather(self, city: str) -> WeatherResult:
        city = WeatherArgs(city=city).city
        try:
            data = await self.fetch(
                "https://geocoding-api.open-meteo.com/v1/search",
                {"name": city, "count": 10, "language": "ru"},
            )
            places = data.get("results", [])
            if not places:
                return WeatherResult(status="not_found")
            if not isinstance(places, list):
                raise AgentError("weather_bad_response")
            qualifiers = [part.strip() for part in city.split(",")]
            if len(qualifiers) >= 3:
                # Выбор страны и области из показанных кандидатов, без скрытого выбора первого.
                places = [
                    p for p in places if p.get("admin1", "").casefold() == qualifiers[2].casefold()
                ]
                if not places:
                    return WeatherResult(status="not_found")
            candidates = [
                Candidate(name=p["name"], country=p.get("country", ""), admin1=p.get("admin1", ""))
                for p in places
            ]
            if len(places) > 1:
                return WeatherResult(status="ambiguous", candidates=candidates)
            place = places[0]
            data = await self.fetch(
                "https://api.open-meteo.com/v1/forecast",
                {
                    "latitude": float(place["latitude"]),
                    "longitude": float(place["longitude"]),
                    "current": "temperature_2m,weather_code,wind_speed_10m",
                    "timezone": "auto",
                    "wind_speed_unit": "kmh",
                },
            )
            current = data["current"]
            zone = data["timezone"]
            observed = datetime.fromisoformat(current["time"]).replace(tzinfo=ZoneInfo(zone))
            return WeatherResult(
                status="ok",
                current=CurrentWeather(
                    city=place["name"],
                    country=place.get("country", ""),
                    admin1=place.get("admin1", ""),
                    temperature_c=current["temperature_2m"],
                    weather_code=current["weather_code"],
                    wind_speed=current["wind_speed_10m"],
                    observed_at=observed.isoformat(),
                    timezone=zone,
                ),
            )
        except AgentError:
            raise
        except (ValueError, KeyError, TypeError, IndexError):
            raise AgentError("weather_bad_response") from None
