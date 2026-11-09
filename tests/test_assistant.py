import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, PhotoSize, Update, User

from app.config import ConfigError, Settings
from app.dialogue import ContextTooLarge, Dialogue, build_context, context_bytes
from app.handlers.assistant import (
    BAD_TEMPERATURE,
    HELP,
    LLM_FAILURE,
    create_router,
    split_text,
)
from app.llm import Completion, LLMError
from app.prompts import PROMPTS
from app.repository import Profile

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"


class MemoryRepository:
    """Подмена только хранилища; SQL отдельно проверяется реальной PostgreSQL."""

    def __init__(self):
        self.profiles = {}
        self.messages = {}
        self.failed = {}

    async def profile(self, chat_id):
        return self.profiles.setdefault(chat_id, Profile())

    async def history(self, chat_id, max_turns):
        return list(self.messages.get(chat_id, []))[-max_turns * 2 :] if max_turns else []

    async def save_turn(self, chat_id, question, answer, max_turns):
        if answer is None:
            self.failed.setdefault(chat_id, []).append(question)
            return
        items = self.messages.setdefault(chat_id, [])
        items.extend(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        )
        self.messages[chat_id] = items[-max_turns * 2 :]

    async def reset(self, chat_id):
        self.messages.pop(chat_id, None)

    async def change_mode(self, chat_id, mode):
        self.profiles[chat_id] = replace(await self.profile(chat_id), mode=mode)
        await self.reset(chat_id)

    async def set_temperature(self, chat_id, temperature):
        self.profiles[chat_id] = replace(await self.profile(chat_id), temperature=temperature)


@pytest.fixture
async def harness():
    repo = MemoryRepository()
    llm = AsyncMock()
    llm.complete.return_value = Completion("Ответ", {}, "test-model")
    settings = Settings(bot_token=TOKEN, postgres_password="test", llm_model="test-model")
    dialogue = Dialogue(repo, llm, settings)
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
    counter = 0

    async def send(text, chat_id=42, chat_type="private", photo=False):
        nonlocal counter
        counter += 1
        bot.session.reset_mock()
        message = Message(
            message_id=counter,
            date=datetime.now(UTC),
            chat=Chat(id=chat_id, type=chat_type),
            from_user=User(id=chat_id, is_bot=False, first_name="Test"),
            text=None if photo else text,
            photo=[PhotoSize(file_id="a", file_unique_id="b", width=1, height=1)]
            if photo
            else None,
        )
        await dispatcher.feed_update(
            bot, Update(update_id=counter, message=message), dialogue=dialogue
        )
        return [
            call.args[1].text
            for call in bot.session.call_args_list
            if isinstance(call.args[1], SendMessage)
        ]

    yield repo, llm, dialogue, send
    await bot.session.close()


async def test_default_mode_and_roles_order_and_parameters(harness):
    repo, llm, _, send = harness
    assert await send("Первый вопрос") == ["Ответ"]
    llm.complete.assert_awaited_with(
        [
            {"role": "system", "content": PROMPTS["study"]},
            {"role": "user", "content": "Первый вопрос"},
        ],
        0.3,
    )
    await send("Второй вопрос")
    assert llm.complete.call_args.args[0] == [
        {"role": "system", "content": PROMPTS["study"]},
        {"role": "user", "content": "Первый вопрос"},
        {"role": "assistant", "content": "Ответ"},
        {"role": "user", "content": "Второй вопрос"},
    ]
    assert len(repo.messages[42]) == 4


async def test_two_users_have_independent_history_and_temperature(harness):
    _, llm, _, send = harness
    await send("Секрет первого", 1)
    await send("/settings 1.0", 1)
    await send("Второй", 2)
    assert llm.complete.call_args.args == (
        [
            {"role": "system", "content": PROMPTS["study"]},
            {"role": "user", "content": "Второй"},
        ],
        0.3,
    )
    await send("Продолжим", 1)
    assert llm.complete.call_args.args[1] == 1.0
    assert llm.complete.call_args.args[0][1]["content"] == "Секрет первого"


@pytest.mark.parametrize("mode", ["study", "translate", "quiz"])
async def test_mode_change_clears_history_and_is_not_sent_to_llm(harness, mode):
    repo, llm, _, send = harness
    await send("Вопрос")
    await send("/settings 0.7")
    llm.complete.reset_mock()
    await send("/" + mode)
    assert (await repo.profile(42)) == Profile(mode, 0.7)
    assert await repo.history(42, 10) == []
    llm.complete.assert_not_awaited()
    await send("Новый вопрос")
    assert llm.complete.call_args.args[0][0]["content"] == PROMPTS[mode]


async def test_reset_only_current_user_preserves_settings(harness):
    repo, llm, _, send = harness
    await send("Вопрос", 1)
    await send("Другой вопрос", 2)
    await send("/settings 0.0", 1)
    llm.complete.reset_mock()
    assert await send("/reset", 1) == ["История очищена. Режим и температура сохранены."]
    assert await repo.history(1, 10) == []
    assert len(await repo.history(2, 10)) == 2
    assert (await repo.profile(1)).temperature == 0.0
    llm.complete.assert_not_awaited()


@pytest.mark.parametrize("value", ["0.0", "0.3", "0.7", "1.0"])
async def test_allowed_settings(harness, value):
    repo, llm, _, send = harness
    assert await send("/settings " + value) == [
        f"Режим: Учёба\nМодель: test-model\nТемпература: {value}",
    ]
    assert (await repo.profile(42)).temperature == float(value)
    llm.complete.assert_not_awaited()


@pytest.mark.parametrize("value", ["nan", "0.5", "-1", "2", "0.7 extra", "garbage"])
async def test_bad_settings_never_reach_api(harness, value):
    repo, llm, _, send = harness
    assert await send("/settings " + value) == [BAD_TEMPERATURE]
    assert (await repo.profile(42)).temperature == 0.3
    llm.complete.assert_not_awaited()


async def test_llm_error_safe_and_not_persisted(harness):
    repo, llm, _, send = harness
    llm.complete.side_effect = LLMError("private-technical-detail")
    assert await send("Вопрос") == [LLM_FAILURE]
    assert await repo.history(42, 10) == []
    assert repo.failed[42] == ["Вопрос"]


async def test_long_answer_sent_in_order_without_losing_characters(harness):
    _, llm, _, send = harness
    text = "<b>Ответ</b>\n" + "🙂" * 3000 + "a" * 4097
    llm.complete.return_value = Completion(text, {}, "test")
    parts = await send("Вопрос")
    assert "".join(parts) == text
    assert all(0 < len(p.encode("utf-16-le")) // 2 <= 4096 for p in parts)


async def test_commands_private_chat_and_text_filter(harness):
    _, llm, _, send = harness
    assert await send("/start") == [HELP]
    assert await send("/unknown") == ["Неизвестная команда. Список команд: /start."]
    assert await send("Вопрос", chat_type="group") == []
    assert await send("", photo=True) == []
    llm.complete.assert_not_awaited()


def test_context_trims_whole_old_pairs_and_preserves_current_and_system():
    history = [
        {"role": role, "content": f"{n}{role}"} for n in range(4) for role in ("user", "assistant")
    ]
    context = build_context("system", history, "current", 5, 10000)
    assert context == [
        {"role": "system", "content": "system"},
        *history[-4:],
        {"role": "user", "content": "current"},
    ]
    budget = context_bytes([context[0], *history[-2:], context[-1]])
    assert build_context("system", history, "current", 20, budget) == [
        context[0],
        *history[-2:],
        context[-1],
    ]
    with pytest.raises(ContextTooLarge):
        build_context("system", history, "current" * 100, 20, 100)
    assert build_context("system", history, "current", 1, 10000) == [context[0], context[-1]]


async def test_context_limit_applied_to_actual_api_request(harness):
    _, llm, dialogue, send = harness
    dialogue.settings = replace(dialogue.settings, history_max_messages=3)
    await send("Первый")
    await send("Второй")
    await send("Третий")
    assert [m["content"] for m in llm.complete.call_args.args[0][1:]] == [
        "Второй",
        "Ответ",
        "Третий",
    ]
    dialogue.settings = replace(dialogue.settings, context_max_bytes=10)
    llm.complete.reset_mock()
    assert await send("Большой") == [
        "Сообщение слишком большое для контекста модели. Сократите его.",
    ]
    llm.complete.assert_not_awaited()


async def test_user_lock_serializes_and_releases_after_cancel(harness):
    _, _, dialogue, _ = harness
    entered = asyncio.Event()

    async def waiter():
        async with dialogue.user(1):
            entered.set()

    async with dialogue.user(1):
        task = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        assert not entered.is_set()
        async with dialogue.user(2):
            pass
    await task
    assert entered.is_set()
    assert dialogue._locks == {}
    async with dialogue.user(1):
        task = asyncio.create_task(waiter())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert dialogue._locks == {}


def test_required_llm_configuration_and_secret_repr(tmp_path):
    env = {"BOT_TOKEN": TOKEN, "POSTGRES_PASSWORD": "test"}
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        Settings.load(tmp_path / "missing", environ=env, require_llm=True)
    env.update(LLM_API_KEY="private-key", LLM_BASE_URL="https://example.org/v1", LLM_MODEL="m")
    settings = Settings.load(tmp_path / "missing", environ=env, require_llm=True)
    assert "private-key" not in repr(settings)
    env["LLM_BASE_URL"] = "https://user:private-password@example.org/v1"
    with pytest.raises(ConfigError) as error:
        Settings.load(tmp_path / "missing", environ=env, require_llm=True)
    assert "private-password" not in str(error.value)


def test_split_empty_does_not_create_empty_message():
    assert split_text("") == []
