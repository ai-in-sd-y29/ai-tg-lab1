"""Клиент OpenAI-совместимого API (Chat Completions).

Обработчики Telegram не знают о HTTP: они получают объект с методом complete().
В тестах вместо него подставляется контролируемая реализация.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Literal, Protocol

import aiohttp
from aiohttp_socks import ProxyConnector

from app.config import Settings

logger = logging.getLogger("app.llm")

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    @property
    def truncated(self) -> bool:
        # "length" означает, что генерация остановлена лимитом токенов.
        return self.finish_reason == "length"


class LLMError(Exception):
    """Базовая ошибка обращения к модели. Текст не содержит ключей и промптов."""


class LLMTimeoutError(LLMError):
    pass


class LLMUnavailableError(LLMError):
    """Сеть недоступна, сервис вернул 5xx или 429."""


class LLMRequestError(LLMError):
    """Сервис отклонил запрос: неверный ключ, модель, параметры (4xx)."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


class LLMEmptyResponseError(LLMError):
    """Ответ получен, но в нём нет текста."""


class LLMClient(Protocol):
    model: str

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        request_id: str,
    ) -> LLMResult: ...


def _is_transient(error: LLMError) -> bool:
    return isinstance(error, LLMUnavailableError)


class OpenAICompatibleClient:
    # Пауза перед повтором растёт с номером попытки; в тестах обнуляется.
    retry_delay = 2.0

    def __init__(self, settings: Settings, session: aiohttp.ClientSession | None = None):
        self.model = settings.llm_model
        self._url = settings.llm_base_url + "/chat/completions"
        self._api_key = settings.llm_api_key
        self._timeout = settings.llm_timeout_seconds
        self._max_tokens = settings.llm_max_output_tokens
        self._max_retries = settings.llm_max_retries
        self._proxy_url = settings.llm_proxy_url
        self._session = session
        self._own_session = session is None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = ProxyConnector.from_url(self._proxy_url) if self._proxy_url else None
            # trust_env=False: системный HTTP_PROXY не должен незаметно менять маршрут.
            self._session = aiohttp.ClientSession(connector=connector, trust_env=False)
            self._own_session = True
        return self._session

    async def close(self) -> None:
        if self._own_session and self._session is not None and not self._session.closed:
            await self._session.close()

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float,
        request_id: str,
    ) -> LLMResult:
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": self._max_tokens,
        }
        attempt = 0
        while True:
            attempt += 1
            started = time.monotonic()
            logger.info(
                "LLM запрос начат: request_id=%s model=%s attempt=%d messages=%d temperature=%s",
                request_id,
                self.model,
                attempt,
                len(messages),
                temperature,
            )
            try:
                result = await self._send(payload)
            except LLMError as exc:
                elapsed = time.monotonic() - started
                logger.warning(
                    "LLM запрос завершился ошибкой: request_id=%s model=%s attempt=%d "
                    "duration=%.2fs error=%s (%s)",
                    request_id,
                    self.model,
                    attempt,
                    elapsed,
                    type(exc).__name__,
                    exc,
                )
                if _is_transient(exc) and attempt <= self._max_retries:
                    await asyncio.sleep(min(self.retry_delay * attempt, 5.0))
                    continue
                raise
            elapsed = time.monotonic() - started
            logger.info(
                "LLM запрос выполнен: request_id=%s model=%s duration=%.2fs finish_reason=%s "
                "prompt_tokens=%s completion_tokens=%s",
                request_id,
                result.model,
                elapsed,
                result.finish_reason,
                result.prompt_tokens,
                result.completion_tokens,
            )
            return result

    async def _send(self, payload: dict) -> LLMResult:
        headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            async with asyncio.timeout(self._timeout):
                async with self._get_session().post(
                    self._url, json=payload, headers=headers
                ) as response:
                    status = response.status
                    try:
                        body = await response.json(content_type=None)
                    except ValueError:
                        body = None
        except TimeoutError:
            raise LLMTimeoutError(f"нет ответа за {self._timeout:g} с") from None
        except aiohttp.ClientError as exc:
            raise LLMUnavailableError(f"сетевая ошибка {type(exc).__name__}") from None

        if status == 429 or status >= 500:
            raise LLMUnavailableError(f"HTTP {status}")
        if status >= 400:
            raise LLMRequestError(f"HTTP {status}: {_error_message(body)}", status)
        return _parse_result(body, str(payload["model"]))


# Провайдеры иногда цитируют в ошибке часть ключа (например, sk-or-v1***abcd).
_KEY_FRAGMENT = re.compile(r"\b(?:sk|pk|rk|key)[-_][\w*.-]+", re.IGNORECASE)


def _error_message(body: object) -> str:
    # Текст ошибки провайдера помогает понять причину (неверная модель, параметр).
    # Ограничиваем длину и вырезаем фрагменты ключей перед записью в журнал.
    message = "без описания"
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            message = error["message"]
        elif isinstance(error, str):
            message = error
    return _KEY_FRAGMENT.sub("[скрыто]", message)[:200]


def _parse_result(body: object, requested_model: str) -> LLMResult:
    if not isinstance(body, dict):
        raise LLMEmptyResponseError("ответ не является JSON-объектом")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMEmptyResponseError("в ответе нет choices")
    choice = choices[0]
    message = choice.get("message")
    text = message.get("content") if isinstance(message, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise LLMEmptyResponseError("в ответе нет текста")
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}

    def token_count(key: str) -> int | None:
        count = usage.get(key)
        return count if isinstance(count, int) else None

    return LLMResult(
        text=text.strip(),
        model=str(body.get("model") or requested_model),
        finish_reason=choice.get("finish_reason"),
        prompt_tokens=token_count("prompt_tokens"),
        completion_tokens=token_count("completion_tokens"),
        total_tokens=token_count("total_tokens"),
    )
