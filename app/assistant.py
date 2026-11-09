"""Логика ассистента, не зависящая от Telegram.

Обработчики вызывают методы этого класса; HTTP-запросы к модели выполняет
LLMClient, SQL — Repository.
"""

import hashlib
import logging
from dataclasses import dataclass

from app.config import ALLOWED_TEMPERATURES, Settings
from app.llm import (
    ChatMessage,
    LLMClient,
    LLMEmptyResponseError,
    LLMError,
    LLMRequestError,
    LLMResult,
    LLMTimeoutError,
)
from app.prompts import MODES, Mode
from app.storage import Repository, UserSettings

logger = logging.getLogger("app.assistant")

ERROR_TIMEOUT = "Модель не ответила вовремя. Попробуйте ещё раз через минуту."
ERROR_UNAVAILABLE = "Сервис языковой модели сейчас недоступен. Попробуйте позже."
ERROR_REJECTED = "Не удалось получить ответ модели. Попробуйте позже или сообщите преподавателю."
ERROR_EMPTY = "Модель вернула пустой ответ. Попробуйте переформулировать запрос."
TRUNCATED_NOTE = "\n\n(Ответ обрезан из-за ограничения длины. Попросите продолжить.)"


class InvalidTemperatureError(ValueError):
    pass


@dataclass(frozen=True)
class Reply:
    text: str
    result: LLMResult | None = None


def parse_temperature(raw: str) -> float:
    """Принимает только значения из ALLOWED_TEMPERATURES; запятая допустима."""
    try:
        value = float(raw.strip().replace(",", "."))
    except ValueError:
        raise InvalidTemperatureError(raw) from None
    for allowed in ALLOWED_TEMPERATURES:
        if abs(value - allowed) < 1e-9:
            return allowed
    raise InvalidTemperatureError(raw)


def trim_history(
    history: list[ChatMessage],
    *,
    max_messages: int,
    budget_chars: int,
) -> list[ChatMessage]:
    """Оставляет самые новые сообщения, укладывающиеся в оба лимита.

    Сообщения отбрасываются с начала, поэтому порядок не нарушается. История
    всегда начинается с сообщения пользователя, чтобы не передавать модели
    ответ ассистента без вопроса, на который он отвечал.
    """
    kept: list[ChatMessage] = []
    used = 0
    for message in reversed(history[-max_messages:] if max_messages > 0 else []):
        if used + len(message.content) > budget_chars:
            break
        kept.append(message)
        used += len(message.content)
    kept.reverse()
    while kept and kept[0].role != "user":
        kept.pop(0)
    return kept


def user_error_text(error: LLMError) -> str:
    if isinstance(error, LLMTimeoutError):
        return ERROR_TIMEOUT
    if isinstance(error, LLMEmptyResponseError):
        return ERROR_EMPTY
    if isinstance(error, LLMRequestError):
        return ERROR_REJECTED
    return ERROR_UNAVAILABLE


class Assistant:
    def __init__(self, repository: Repository, llm: LLMClient, settings: Settings):
        self.repository = repository
        self.llm = llm
        self.settings = settings

    @property
    def model(self) -> str:
        return self.llm.model

    async def get_settings(self, chat_id: int) -> UserSettings:
        settings = await self.repository.get_settings(chat_id)
        if settings.mode not in MODES:
            # Режим мог быть удалён из кода после сохранения в БД.
            return UserSettings(temperature=settings.temperature)
        return settings

    async def build_messages(self, chat_id: int, mode: Mode, text: str) -> list[ChatMessage]:
        history = await self.repository.get_history(chat_id, self.settings.history_max_messages)
        budget = self.settings.history_max_chars - mode.prompt_chars() - len(text)
        trimmed = trim_history(
            history, max_messages=self.settings.history_max_messages, budget_chars=budget
        )
        if len(trimmed) < len(history):
            logger.info(
                "История сокращена: chat_hash=%s было=%d стало=%d",
                chat_hash(chat_id),
                len(history),
                len(trimmed),
            )
        return [*mode.prompt_messages(), *trimmed, ChatMessage("user", text)]

    async def ask(self, chat_id: int, text: str, request_id: str) -> Reply:
        user_settings = await self.get_settings(chat_id)
        mode = MODES[user_settings.mode]
        messages = await self.build_messages(chat_id, mode, text)
        try:
            result = await self.llm.complete(
                messages, temperature=user_settings.temperature, request_id=request_id
            )
        except LLMError as error:
            # Ответ ассистента не сохраняется: история остаётся без изменений.
            return Reply(user_error_text(error))
        await self.repository.append_exchange(chat_id, text, result.text)
        answer = result.text + (TRUNCATED_NOTE if result.truncated else "")
        return Reply(answer, result)

    async def switch_mode(self, chat_id: int, mode_name: str) -> str:
        mode = MODES[mode_name]
        await self.repository.set_mode(chat_id, mode_name)
        return f"Режим «{mode.title}» (/{mode.command}) включён. История очищена.\n{mode.greeting}"

    async def reset(self, chat_id: int) -> str:
        await self.repository.clear_history(chat_id)
        return "История диалога очищена. Режим и temperature сохранены."

    async def settings_text(self, chat_id: int) -> str:
        user_settings = await self.get_settings(chat_id)
        mode = MODES[user_settings.mode]
        allowed = ", ".join(f"{t:.1f}" for t in ALLOWED_TEMPERATURES)
        return (
            "Текущие настройки:\n"
            f"Режим: {mode.title} (/{mode.command})\n"
            f"Модель: {self.model}\n"
            f"Temperature: {user_settings.temperature:.1f}\n\n"
            f"Выберите temperature кнопкой ниже или командой /settings 0.3 "
            f"(допустимо: {allowed})."
        )

    async def set_temperature(self, chat_id: int, raw: str) -> str:
        allowed = ", ".join(f"{t:.1f}" for t in ALLOWED_TEMPERATURES)
        try:
            value = parse_temperature(raw)
        except InvalidTemperatureError:
            return f"Недопустимое значение temperature. Выберите одно из: {allowed}."
        await self.repository.set_temperature(chat_id, value)
        return f"Temperature = {value:.1f} сохранена и применится со следующего запроса."


def chat_hash(chat_id: int) -> str:
    """Короткий необратимый идентификатор чата для журналов."""
    return hashlib.sha256(str(chat_id).encode()).hexdigest()[:10]
