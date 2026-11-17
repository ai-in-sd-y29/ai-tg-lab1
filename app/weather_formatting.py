"""Представление проверенных погодных данных для ответа модели."""

from datetime import datetime
from zoneinfo import ZoneInfo

from app.lab2_models import WeatherResult

# WMO codes: https://open-meteo.com/en/docs#weathervariables
CONDITIONS = {
    0: ("☀️", "Ясно"),
    1: ("🌤️", "Преимущественно ясно"),
    2: ("⛅", "Переменная облачность"),
    3: ("☁️", "Пасмурно"),
    45: ("🌫️", "Туман"),
    48: ("🌫️", "Туман с изморозью"),
    51: ("🌦️", "Слабая морось"),
    53: ("🌦️", "Умеренная морось"),
    55: ("🌦️", "Сильная морось"),
    56: ("🌧️", "Слабая переохлаждённая морось"),
    57: ("🌧️", "Сильная переохлаждённая морось"),
    61: ("🌧️", "Небольшой дождь"),
    63: ("🌧️", "Умеренный дождь"),
    65: ("🌧️", "Сильный дождь"),
    66: ("🌧️", "Слабый переохлаждённый дождь"),
    67: ("🌧️", "Сильный переохлаждённый дождь"),
    71: ("🌨️", "Небольшой снег"),
    73: ("🌨️", "Умеренный снег"),
    75: ("🌨️", "Сильный снег"),
    77: ("🌨️", "Снежные зёрна"),
    80: ("🌧️", "Слабый ливень"),
    81: ("🌧️", "Умеренный ливень"),
    82: ("🌧️", "Сильный ливень"),
    85: ("🌨️", "Небольшой снежный ливень"),
    86: ("🌨️", "Сильный снежный ливень"),
    95: ("⛈️", "Гроза"),
    96: ("⛈️", "Гроза с небольшим градом"),
    97: ("⛈️", "Сильная гроза"),
    99: ("⛈️", "Гроза с сильным градом"),
}


def format_weather(result: WeatherResult) -> str:
    current = result.current
    if result.status != "ok" or current is None:
        raise ValueError("current_weather_required")
    icon, condition = CONDITIONS.get(current.weather_code, ("🌡️", "Состояние погоды не определено"))
    observed = datetime.fromisoformat(current.observed_at).astimezone(ZoneInfo(current.timezone))
    temperature = (
        f"{current.temperature_c:+.1f}"
        if current.temperature_c > 0
        else f"{current.temperature_c:.1f}"
    )
    temperature = temperature.rstrip("0").rstrip(".").replace(".", ",")
    wind_ms = f"{current.wind_speed / 3.6:.1f}".replace(".", ",")
    wind_kmh = f"{current.wind_speed:g}".replace(".", ",")
    location = ", ".join(
        dict.fromkeys(
            value for value in (current.admin1, current.country) if value and value != current.city
        )
    )
    return (
        f"📍 {current.city} — погода сейчас\n"
        + (f"{location}\n" if location else "")
        + f"\n{icon} {condition}\n"
        + f"🌡️ Температура  {temperature} °C\n"
        + f"💨 Ветер  {wind_ms} м/с ({wind_kmh} км/ч)\n"
        + f"\n🕒 Данные на {observed:%d.%m.%Y, %H:%M}\n"
        + f"Часовой пояс  {current.timezone}\n"
        + f"Источник  {result.source}"
    )
