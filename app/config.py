"""Единственное место чтения и проверки настроек приложения."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from aiogram.utils.token import TokenValidationError, validate_token
from dotenv import dotenv_values


class ConfigError(ValueError):
    """Ошибка настройки без секретных значений в сообщении."""


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    postgres_password: str = field(repr=False)
    telegram_proxy_url: str = field(default="", repr=False)
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "bot"
    postgres_user: str = "bot"
    log_level: str = "INFO"
    health_port: int = 8080
    llm_api_key: str = field(default="", repr=False)
    llm_base_url: str = ""
    llm_model: str = ""
    llm_timeout: int = 45
    llm_max_tokens: int = 1000
    history_max_messages: int = 20
    context_max_bytes: int = 24000

    @classmethod
    def load(
        cls,
        env_file: Path | str = ".env",
        *,
        environ: Mapping[str, str] | None = None,
        require_llm: bool = False,
    ) -> "Settings":
        values = {
            **dotenv_values(env_file, interpolate=False),
            **(os.environ if environ is None else environ),
        }

        def value(key: str, default: str = "") -> str:
            return values.get(key) or default

        def port(key: str, default: str) -> int:
            try:
                result = int(value(key, default))
                if not 1 <= result <= 65535:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужен номер порта от 1 до 65535.") from None

        token = value("BOT_TOKEN")
        try:
            validate_token(token)
        except TokenValidationError:
            raise ConfigError("BOT_TOKEN: укажите токен, полученный у BotFather.") from None
        password = value("POSTGRES_PASSWORD")
        if not password:
            raise ConfigError("POSTGRES_PASSWORD: пароль базы данных не задан.")
        proxy = value("TELEGRAM_PROXY_URL")
        if proxy:
            try:
                parsed = urlsplit(proxy)
                if (
                    parsed.scheme not in {"http", "socks5"}
                    or not parsed.hostname
                    or not parsed.port
                    or parsed.path not in {"", "/"}
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError
            except ValueError:
                raise ConfigError(
                    "TELEGRAM_PROXY_URL: нужен http://host:port или socks5://host:port; "
                    "при необходимости добавьте user:password@."
                ) from None
        level = value("LOG_LEVEL", "INFO").upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError("LOG_LEVEL: используйте DEBUG, INFO, WARNING, ERROR или CRITICAL.")

        def positive(key: str, default: int) -> int:
            try:
                result = int(value(key, str(default)))
                if result <= 0:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужно положительное целое число.") from None

        llm_key = value("LLM_API_KEY").strip()
        llm_url = value("LLM_BASE_URL").strip().rstrip("/")
        llm_model = value("LLM_MODEL").strip()
        if require_llm or any((llm_key, llm_url, llm_model)):
            for name, setting in (
                ("LLM_API_KEY", llm_key),
                ("LLM_BASE_URL", llm_url),
                ("LLM_MODEL", llm_model),
            ):
                if not setting:
                    raise ConfigError(f"{name}: обязательная настройка не задана.")
            try:
                endpoint = urlsplit(llm_url)
                if (
                    endpoint.scheme != "https"
                    or not endpoint.hostname
                    or endpoint.username
                    or endpoint.password
                    or endpoint.query
                    or endpoint.fragment
                ):
                    raise ValueError
                _ = endpoint.port
            except ValueError:
                raise ConfigError("LLM_BASE_URL: нужен HTTPS-адрес без учётных данных.") from None
        return cls(
            bot_token=token,
            postgres_password=password,
            telegram_proxy_url=proxy,
            postgres_host=value("POSTGRES_HOST", "127.0.0.1"),
            postgres_port=port("POSTGRES_PORT", "5432"),
            postgres_db=value("POSTGRES_DB", "bot"),
            postgres_user=value("POSTGRES_USER", "bot"),
            log_level=level,
            health_port=port("HEALTH_PORT", "8080"),
            llm_api_key=llm_key,
            llm_base_url=llm_url,
            llm_model=llm_model,
            llm_timeout=positive("LLM_TIMEOUT", 45),
            llm_max_tokens=positive("LLM_MAX_TOKENS", 1000),
            history_max_messages=positive("HISTORY_MAX_MESSAGES", 20),
            context_max_bytes=positive("CONTEXT_MAX_BYTES", 24000),
        )
