"""12 запусков через обработчик бота, реальную PostgreSQL и LLM.

Telegram-транспорт подменён: сообщения ни одному человеку не отправляются.
Запуск: python -m scripts.lab1_experiment --output docs/lab1-experiment.json
Требуются dev-зависимости. Результаты содержат только заданные здесь учебные тексты.
"""

import argparse
import asyncio
import json
import logging
import secrets
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

from app.config import ConfigError, Settings
from app.db import create_pool
from app.dialogue import Dialogue
from app.handlers.assistant import create_router
from app.llm import LLMClient
from app.prompts import PROMPTS
from app.repository import Repository

QUERY = (
    "Предложи три разных способа объяснить новичку разницу между списком и кортежем Python. "
    "Ответ — ровно три нумерованных пункта (1., 2., 3.) без вступления. "
    "В каждом пункте используй отдельную аналогию и назови хотя бы один технический факт. "
    "После списка добавь одну строку, начинающуюся с 'Самопроверка:'."
)
CRITERIA = [
    "Ровно три нумерованных пункта и одна строка Самопроверка: без вступления",
    "В каждом пункте своя понятная аналогия и технический факт",
    "Нет явных противоречий: список изменяем, кортеж неизменяем; "
    "кортеж может содержать изменяемые объекты",
]
COMPARISON_QUERY = "Объясни новичку, чем отличается параметр функции от аргумента в Python."
WEAK_PROMPT = "Ты помощник. Отвечай на вопросы."


class RecordingClient:
    def __init__(self, client: LLMClient):
        self.client = client
        self.last = None

    async def complete(self, messages, temperature):
        self.last = await self.client.complete(messages, temperature)
        return self.last


def format_ok(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(lines) == 4 and all(
        line.startswith(prefix)
        for line, prefix in zip(lines, ("1.", "2.", "3.", "Самопроверка:"), strict=True)
    )


async def experiment(settings: Settings, output: Path) -> None:
    # Отдельный случайный пользователь исключает изменение реальных диалогов.
    chat_id = -(1_000_000_000 + secrets.randbelow(1_000_000_000))
    pool = await create_pool(settings)
    repository = Repository(pool)
    client = LLMClient(settings)
    recorder = RecordingClient(client)
    # Фиксируем остальные параметры для всех запусков.
    dialogue = Dialogue(repository, recorder, replace(settings, llm_max_tokens=1000))
    bot = Bot("123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk")
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
    run_id = 0
    result = {
        "date_utc": datetime.now(UTC).isoformat(),
        "provider": "Yandex AI Studio",
        "api": "OpenAI-compatible Chat Completions v1",
        "model": settings.llm_model.replace(settings.llm_model.split("/")[2], "<folder_id>")
        if settings.llm_model.startswith("gpt://")
        else settings.llm_model,
        "transport": "synthetic Telegram updates; real handler, PostgreSQL and LLM API",
        "query": QUERY,
        "system_prompt": PROMPTS["study"],
        "max_tokens": 1000,
        "stream": False,
        "other_sampling_parameters": "omitted; provider defaults, unchanged",
        "criteria_defined_before_runs": CRITERIA,
        "runs": [],
        "prompt_comparison": [],
    }

    def checkpoint():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    async def send(text):
        nonlocal run_id
        run_id += 1
        bot.session.reset_mock()
        message = Message(
            message_id=run_id,
            date=datetime.now(UTC),
            chat=Chat(id=chat_id, type="private"),
            from_user=User(id=abs(chat_id), is_bot=False, first_name="Experiment"),
            text=text,
        )
        await dispatcher.feed_update(
            bot, Update(update_id=run_id, message=message), dialogue=dialogue
        )
        return "".join(
            call.args[1].text
            for call in bot.session.call_args_list
            if isinstance(call.args[1], SendMessage)
        )

    try:
        await repository.initialize()
        await send("/study")
        for temperature in (0.0, 0.3, 0.7, 1.0):
            await send(f"/settings {temperature:.1f}")
            for repetition in range(1, 4):
                await send("/reset")
                assert await repository.history(chat_id, 10) == []
                recorder.last = None
                text = await send(QUERY)
                completion = recorder.last
                record = {
                    "run": len(result["runs"]) + 1,
                    "temperature": temperature,
                    "repetition": repetition,
                    "text": text,
                    "success": completion is not None,
                    "length": len(text),
                    "format_ok": format_ok(text),
                    "usage": completion.usage if completion else {},
                }
                result["runs"].append(record)
                checkpoint()
                print(f"Run {record['run']}/12: success={record['success']}", flush=True)
        await send("/settings 0.3")
        original = PROMPTS["study"]
        try:
            for label, prompt in (("weak", WEAK_PROMPT), ("improved", original)):
                PROMPTS["study"] = prompt
                await send("/reset")
                recorder.last = None
                text = await send(COMPARISON_QUERY)
                result["prompt_comparison"].append(
                    {
                        "label": label,
                        "query": COMPARISON_QUERY,
                        "system_prompt": prompt,
                        "temperature": 0.3,
                        "text": text,
                        "success": recorder.last is not None,
                        "usage": recorder.last.usage if recorder.last else {},
                    }
                )
                checkpoint()
        finally:
            PROMPTS["study"] = original
    finally:
        await client.close()
        await bot.session.close()
        # Удаляется только созданный экспериментом пользователь, не реальная история.
        try:
            await pool.execute("DELETE FROM lab1_users WHERE chat_id=$1", chat_id)
        finally:
            await pool.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output", type=Path, default=Path("docs/lab1-experiment.json"))
    args = parser.parse_args()
    if args.output.exists():
        print("Output already exists. Choose a new path to preserve previous results.")
        return 1
    # Никаких HTTP-тел, настроек или исключений внешних сервисов в консоли.
    logging.disable(logging.CRITICAL)
    try:
        settings = Settings.load(args.env_file, require_llm=True)
        asyncio.run(experiment(settings, args.output))
    except ConfigError as exc:
        print(str(exc))
        return 1
    except Exception as exc:
        print(f"Experiment failed ({type(exc).__name__}); details omitted.")
        return 1
    print("Experiment saved. Inspect every result; failed runs are retained.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
