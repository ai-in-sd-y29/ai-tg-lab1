"""Подменяемый клиент OpenAI-совместимого Chat Completions API."""

import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

import aiohttp

from app.config import Settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Содержит только безопасный код, без HTTP-тела и секретов."""


@dataclass(frozen=True)
class Completion:
    text: str
    usage: dict
    model: str


class LanguageModel(Protocol):
    async def complete(self, messages: list[dict[str, str]], temperature: float) -> Completion: ...


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session: aiohttp.ClientSession | None = None

    async def close(self) -> None:
        if self.session:
            await self.session.close()

    async def complete(self, messages: list[dict[str, str]], temperature: float) -> Completion:
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.settings.llm_timeout),
            )
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        # Yandex URI содержит folder_id; в логах оставляем только имя модели.
        model_label = self.settings.llm_model.rsplit("/", 2)[-2:]
        logger.info("LLM start request=%s model=%s", request_id, "/".join(model_label))
        headers = {"Authorization": f"Bearer {self.settings.llm_api_key}"}
        if self.settings.llm_model.startswith("gpt://"):
            headers["OpenAI-Project"] = self.settings.llm_model.split("/")[2]
        try:
            async with self.session.post(
                self.settings.llm_base_url + "/chat/completions",
                headers=headers,
                json={
                    "model": self.settings.llm_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": self.settings.llm_max_tokens,
                    "stream": False,
                },
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise LLMError(f"http_{response.status}")
                # Ограничение памяти при ошибочном/аномально большом ответе провайдера.
                raw = bytearray()
                async for chunk in response.content.iter_chunked(16384):
                    raw.extend(chunk)
                    if len(raw) > 1_000_000:
                        raise LLMError("response_too_large")
                data = json.loads(raw)
                choice = data["choices"][0]
                text = choice["message"]["content"]
                if not isinstance(text, str) or not text.strip():
                    raise LLMError("empty_response")
                if choice.get("finish_reason") != "stop":
                    raise LLMError("incomplete_response")
                result = Completion(text, data.get("usage") or {}, data.get("model", ""))
        except LLMError as exc:
            logger.warning(
                "LLM failure request=%s code=%s elapsed=%.2f",
                request_id,
                exc,
                time.monotonic() - started,
            )
            raise
        except (TimeoutError, aiohttp.ClientError, ValueError, KeyError, IndexError, TypeError):
            logger.warning(
                "LLM failure request=%s code=transport_or_format elapsed=%.2f",
                request_id,
                time.monotonic() - started,
            )
            raise LLMError("transport_or_format") from None
        logger.info("LLM success request=%s elapsed=%.2f", request_id, time.monotonic() - started)
        return result
