"""Реальные LLM + stdio MCP + PostgreSQL через Telegram handlers, транспорт Telegram заменён.

Никому не отправляет сообщений. Создаёт и удаляет отдельный Docker-контейнер с БД.
"""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User

from app.agent import Agent
from app.config import Settings
from app.db import create_pool
from app.dialogue import Dialogue
from app.handlers.assistant import create_router
from app.lab2_repository import AgentRepository
from app.llm import LLMClient
from app.mcp_client import connect_gateway
from app.repository import Repository
from scripts.lab2_environment import temporary_database


async def smoke(settings):
    pool = await create_pool(settings)
    llm = LLMClient(settings)
    bot = Bot("123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk")
    bot.session = AsyncMock()
    results = []
    owner = 1001
    counter = 0
    try:
        repo = Repository(pool)
        await repo.initialize()
        actions = AgentRepository(pool)
        await actions.initialize()
        dialogue = Dialogue(repo, llm, settings)
        dispatcher = Dispatcher()
        dispatcher.include_router(create_router())
        async with connect_gateway(settings) as gateway:
            if not gateway.available:
                raise RuntimeError("MCP unavailable")
            agent = Agent(dialogue, actions, gateway)

            async def send(text, chat_id=owner):
                nonlocal counter
                counter += 1
                bot.session.reset_mock()
                message = Message(
                    message_id=counter,
                    date=datetime.now(UTC),
                    chat=Chat(id=chat_id, type="private"),
                    from_user=User(id=chat_id, is_bot=False, first_name="Synthetic"),
                    text=text,
                )
                await dispatcher.feed_update(
                    bot, Update(update_id=counter, message=message), dialogue=dialogue, agent=agent
                )
                responses = [
                    c.args[1]
                    for c in bot.session.call_args_list
                    if isinstance(c.args[1], SendMessage)
                ]
                results.append({"query": text, "responses": [r.text for r in responses]})
                print(text, "=>", len(responses), "responses", flush=True)
                return responses

            for text in [
                "/tools",
                "/timezone Europe/Moscow",
                "Покажи расписание на 12 ноября 2026 года.",
                "/why",
                "Через два часа напомни сделать перерыв.",
            ]:
                responses = await send(text)
            assert responses[-1].reply_markup, "Missing confirmation buttons"
            assert await pool.fetchval("SELECT count(*) FROM lab2_reminders") == 0
            callback_data = responses[-1].reply_markup.inline_keyboard[0][0].callback_data
            for repetition in range(2):
                counter += 1
                bot.session.reset_mock()
                message = Message(
                    message_id=50,
                    date=datetime.now(UTC),
                    chat=Chat(id=owner, type="private"),
                    from_user=User(id=bot.id, is_bot=True, first_name="Bot"),
                    text="confirmation",
                )
                query = CallbackQuery(
                    id=str(counter),
                    from_user=User(id=owner, is_bot=False, first_name="Synthetic"),
                    chat_instance="synthetic",
                    message=message,
                    data=callback_data,
                )
                await dispatcher.feed_update(
                    bot,
                    Update(update_id=counter, callback_query=query),
                    dialogue=dialogue,
                    agent=agent,
                )
                replies = [
                    c.args[1].text
                    for c in bot.session.call_args_list
                    if isinstance(c.args[1], SendMessage)
                ]
                results.append(
                    {
                        "query": "confirm" if repetition == 0 else "repeat_confirm",
                        "responses": replies,
                    }
                )
                assert await pool.fetchval("SELECT count(*) FROM lab2_reminders") == 1
            await send("/why", chat_id=1002)
            assert await actions.timezone(1002) is None
        return {
            "date": datetime.now(UTC).isoformat(),
            "telegram_transport": "synthetic; no external Telegram requests",
            "backend": "real YandexGPT, real stdio MCP, isolated real PostgreSQL",
            "results": results,
            "reminders_after_two_confirmations": 1,
        }
    finally:
        await bot.session.close()
        await llm.close()
        await pool.close()


def main():
    with temporary_database(Settings.load(require_llm=True)) as settings:
        result = asyncio.run(smoke(settings))
    Path("docs/lab2-smoke.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Smoke passed: confirmed exactly one reminder.")


if __name__ == "__main__":
    main()
