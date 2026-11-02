"""Сценарии диалога, изоляция пользователя и ограничение контекста."""

import asyncio
from contextlib import asynccontextmanager

from app.config import Settings
from app.llm import Completion, LanguageModel, LLMError
from app.prompts import PROMPTS
from app.repository import Repository

TEMPERATURES = {"0.0": 0.0, "0.3": 0.3, "0.7": 0.7, "1.0": 1.0}


class ContextTooLarge(ValueError):
    pass


def context_bytes(messages: list[dict[str, str]]) -> int:
    # UTF-8 bytes + запас на служебное представление каждой роли.
    return sum(len(m["content"].encode("utf-8")) + 32 for m in messages)


def build_context(
    prompt: str, history: list[dict[str, str]], text: str, max_messages: int, max_bytes: int
) -> list[dict[str, str]]:
    system = {"role": "system", "content": prompt}
    current = {"role": "user", "content": text}
    if context_bytes([system, current]) > max_bytes:
        raise ContextTooLarge
    # Лимит числа сообщений включает текущий запрос, но не системную инструкцию.
    pairs = max(0, (max_messages - 1) // 2)
    selected = history[-pairs * 2 :] if pairs else []
    while selected and context_bytes([system, *selected, current]) > max_bytes:
        selected = selected[2:]
    return [system, *selected, current]


class Dialogue:
    def __init__(self, repository: Repository, llm: LanguageModel, settings: Settings):
        self.repository = repository
        self.llm = llm
        self.settings = settings
        self._locks: dict[int, tuple[asyncio.Lock, int]] = {}

    @asynccontextmanager
    async def user(self, chat_id: int):
        # Включает команды: /reset не может обогнать незавершённый ответ.
        lock, count = self._locks.get(chat_id, (asyncio.Lock(), 0))
        self._locks[chat_id] = (lock, count + 1)
        try:
            async with lock:
                yield
        finally:
            _, count = self._locks[chat_id]
            if count == 1:
                del self._locks[chat_id]
            else:
                self._locks[chat_id] = (lock, count - 1)

    async def answer(self, chat_id: int, text: str) -> Completion:
        profile = await self.repository.profile(chat_id)
        history = await self.repository.history(chat_id, self.settings.history_max_messages // 2)
        messages = build_context(
            PROMPTS[profile.mode],
            history,
            text,
            self.settings.history_max_messages,
            self.settings.context_max_bytes,
        )
        try:
            completion = await self.llm.complete(messages, profile.temperature)
        except LLMError:
            # Сам вопрос сохраняется, но неполная пара не попадает в контекст модели.
            await self.repository.save_turn(
                chat_id, text, None, max(1, self.settings.history_max_messages // 2)
            )
            raise
        await self.repository.save_turn(
            chat_id, text, completion.text, max(1, self.settings.history_max_messages // 2)
        )
        return completion
