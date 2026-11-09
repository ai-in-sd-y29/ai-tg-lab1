"""Тонкий Telegram-слой: команды, индикация и доставка текста."""

import logging

import asyncpg
from aiogram import F, Router
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from app.dialogue import TEMPERATURES, ContextTooLarge, Dialogue
from app.llm import LLMError
from app.prompts import MODE_NAMES

logger = logging.getLogger(__name__)
HELP = (
    "Я — AI-ассистент студента. Напишите вопрос или выберите режим:\n"
    "/study — объяснение программирования (по умолчанию)\n"
    "/translate — перевод на указанный вами язык\n"
    "/quiz — учебный опрос по программированию\n"
    "/settings — режим, модель и температура\n"
    "/settings 0.7 — установить температуру (0.0, 0.3, 0.7, 1.0)\n"
    "/reset — очистить историю, сохранив настройки"
)
LLM_FAILURE = "Не удалось получить ответ модели. Попробуйте ещё раз немного позже."
DB_FAILURE = "Не удалось сохранить или прочитать диалог. Попробуйте ещё раз немного позже."
BAD_TEMPERATURE = "Укажите /settings 0.0, /settings 0.3, /settings 0.7 или /settings 1.0."


def split_text(text: str, limit: int = 4096) -> list[str]:
    # Telegram использует UTF-16 для смещений: считаем emoji за две единицы.
    if limit < 2:
        raise ValueError("limit must be at least 2")
    chunks, buffer, size = [], [], 0
    for char in text:
        width = 2 if ord(char) > 0xFFFF else 1
        if size + width > limit:
            chunks.append("".join(buffer))
            buffer, size = [], 0
        buffer.append(char)
        size += width
    if buffer:
        chunks.append("".join(buffer))
    return chunks


async def handle_text(message: Message, dialogue: Dialogue) -> None:
    text = message.text or ""
    chat_id = message.chat.id
    async with dialogue.user(chat_id):
        try:
            await dialogue.repository.profile(chat_id)
            if text.startswith("/"):
                command, _, argument = text.partition(" ")
                command = command.split("@")[0].lower()
                argument = argument.strip()
                if command == "/start":
                    response = HELP
                elif command[1:] in MODE_NAMES:
                    mode = command[1:]
                    await dialogue.repository.change_mode(chat_id, mode)
                    response = f"Режим: {MODE_NAMES[mode]}. История очищена."
                elif command == "/reset":
                    await dialogue.repository.reset(chat_id)
                    response = "История очищена. Режим и температура сохранены."
                elif command == "/settings":
                    if argument:
                        if argument not in TEMPERATURES:
                            await message.answer(BAD_TEMPERATURE, parse_mode=None)
                            return
                        await dialogue.repository.set_temperature(chat_id, TEMPERATURES[argument])
                    profile = await dialogue.repository.profile(chat_id)
                    response = (
                        f"Режим: {MODE_NAMES[profile.mode]}\n"
                        f"Модель: {dialogue.settings.llm_model}\n"
                        f"Температура: {profile.temperature:.1f}"
                    )
                else:
                    response = "Неизвестная команда. Список команд: /start."
            else:
                async with ChatActionSender.typing(chat_id=chat_id, bot=message.bot):
                    completion = await dialogue.answer(chat_id, text)
                response = completion.text
        except ContextTooLarge:
            response = "Сообщение слишком большое для контекста модели. Сократите его."
        except LLMError:
            response = LLM_FAILURE
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError):
            logger.warning("Dialogue storage failure; details omitted")
            response = DB_FAILURE
        for part in split_text(response):
            await message.answer(part, parse_mode=None)


def create_router() -> Router:
    router = Router(name="assistant")
    router.message.register(handle_text, F.chat.type == "private", F.text)
    return router
