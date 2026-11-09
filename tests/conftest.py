"""Контролируемые замены LLM API и хранилища для тестов без сети и БД."""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import AnswerCallbackQuery, SendChatAction, SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from app.assistant import Assistant
from app.config import Settings
from app.handlers.assistant import create_router
from app.llm import ChatMessage, LLMResult
from app.storage import UserSettings

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"


class FakeLLM:
    """Возвращает заранее заданные ответы и запоминает каждый запрос."""

    model = "test-model"

    def __init__(self, *responses: str | Exception | LLMResult):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.delay = 0.0

    async def complete(self, messages, *, temperature, request_id):
        self.calls.append(
            {"messages": list(messages), "temperature": temperature, "request_id": request_id}
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        response = self.responses.pop(0) if self.responses else "Ответ модели"
        if isinstance(response, Exception):
            raise response
        if isinstance(response, LLMResult):
            return response
        return LLMResult(text=response, model=self.model, finish_reason="stop")


class InMemoryRepository:
    """Повторяет контракт PostgresRepository в памяти процесса."""

    def __init__(self):
        self.settings: dict[int, UserSettings] = {}
        self.history: dict[int, list[ChatMessage]] = {}

    async def get_settings(self, chat_id):
        return self.settings.get(chat_id, UserSettings())

    async def set_mode(self, chat_id, mode):
        current = self.settings.get(chat_id, UserSettings())
        self.settings[chat_id] = replace(current, mode=mode)
        self.history.pop(chat_id, None)

    async def set_temperature(self, chat_id, temperature):
        current = self.settings.get(chat_id, UserSettings())
        self.settings[chat_id] = replace(current, temperature=temperature)

    async def clear_history(self, chat_id):
        self.history.pop(chat_id, None)

    async def get_history(self, chat_id, limit):
        return list(self.history.get(chat_id, [])[-limit:]) if limit > 0 else []

    async def append_exchange(self, chat_id, user_text, assistant_text):
        self.history.setdefault(chat_id, []).extend(
            [ChatMessage("user", user_text), ChatMessage("assistant", assistant_text)]
        )


def make_settings(**overrides) -> Settings:
    values = {
        "bot_token": TOKEN,
        "postgres_password": "unused",
        "llm_base_url": "https://llm.example/v1",
        "llm_api_key": "sk-test-secret",
        "llm_model": "test-model",
        **overrides,
    }
    return Settings(**values)


class BotHarness:
    """Прогоняет апдейты через настоящий Dispatcher с подменённой сессией Telegram."""

    def __init__(self, llm: FakeLLM, repository: InMemoryRepository, settings: Settings):
        self.llm = llm
        self.repository = repository
        self.assistant = Assistant(repository, llm, settings)
        self.bot = Bot(TOKEN)
        self.bot.session = AsyncMock()
        self.dispatcher = Dispatcher()
        self.dispatcher.include_router(create_router())
        self._update_id = 0

    def _next_id(self) -> int:
        self._update_id += 1
        return self._update_id

    def _message(self, chat_id: int, text: str | None, chat_type: str = "private") -> Message:
        return Message(
            message_id=self._next_id(),
            date=datetime.now(UTC),
            chat=Chat(id=chat_id, type=chat_type),
            from_user=User(id=chat_id, is_bot=False, first_name="Студент"),
            text=text,
        )

    async def send(self, chat_id: int, text: str | None, chat_type: str = "private") -> None:
        update = Update(update_id=self._next_id(), message=self._message(chat_id, text, chat_type))
        await self.dispatcher.feed_update(self.bot, update, assistant=self.assistant)

    async def press(self, chat_id: int, data: str) -> None:
        callback = CallbackQuery(
            id=str(self._next_id()),
            from_user=User(id=chat_id, is_bot=False, first_name="Студент"),
            chat_instance="test",
            data=data,
            message=self._message(chat_id, "Текущие настройки"),
        )
        update = Update(update_id=self._next_id(), callback_query=callback)
        await self.dispatcher.feed_update(self.bot, update, assistant=self.assistant)

    def sent(self) -> list[SendMessage]:
        methods = [call.args[1] for call in self.bot.session.call_args_list]
        return [m for m in methods if isinstance(m, SendMessage)]

    def sent_texts(self, chat_id: int | None = None) -> list[str]:
        return [m.text for m in self.sent() if chat_id is None or m.chat_id == chat_id]

    def chat_actions(self) -> list[SendChatAction]:
        methods = [call.args[1] for call in self.bot.session.call_args_list]
        return [m for m in methods if isinstance(m, SendChatAction)]

    def callback_answers(self) -> list[AnswerCallbackQuery]:
        methods = [call.args[1] for call in self.bot.session.call_args_list]
        return [m for m in methods if isinstance(m, AnswerCallbackQuery)]

    def clear(self) -> None:
        self.bot.session.reset_mock()


@pytest.fixture
def harness_factory():
    def factory(*responses, **settings_overrides) -> BotHarness:
        return BotHarness(
            FakeLLM(*responses), InMemoryRepository(), make_settings(**settings_overrides)
        )

    return factory
