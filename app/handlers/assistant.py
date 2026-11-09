"""Обработчики Telegram: принимают команды и текст, отвечают пользователю.

Здесь нет SQL, HTTP-запросов к модели и текстов промптов — всё это в Assistant.
"""

import logging
import uuid

from aiogram import Bot, F, Router
from aiogram.enums import ChatType
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from aiogram.utils.chat_action import ChatActionSender

from app.assistant import Assistant, chat_hash
from app.config import ALLOWED_TEMPERATURES
from app.prompts import MODES
from app.text_split import split_message

logger = logging.getLogger("app.handlers")

TEMPERATURE_CALLBACK = "temperature:"
UNKNOWN_COMMAND = "Неизвестная команда. Список команд — /start."
TEXT_ONLY = "Я понимаю только текстовые сообщения."
INTERNAL_ERROR = "Что-то пошло не так. Попробуйте ещё раз."


def start_text() -> str:
    modes = "\n".join(f"/{m.command} — {m.description}" for m in MODES.values())
    return (
        "Привет! Я AI-ассистент студента на основе языковой модели.\n\n"
        "Режимы (при переключении история очищается):\n"
        f"{modes}\n\n"
        "Другие команды:\n"
        "/settings — режим, модель и temperature\n"
        "/reset — очистить историю диалога\n\n"
        "По умолчанию включён режим /study. Просто напишите сообщение."
    )


def temperature_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text=f"{t:.1f}", callback_data=f"{TEMPERATURE_CALLBACK}{t}")
                for t in ALLOWED_TEMPERATURES
            ]
        ]
    )


async def send_long(message: Message, text: str) -> None:
    # parse_mode=None: ответ модели отправляется как обычный текст без разметки.
    for part in split_message(text):
        await message.answer(part, parse_mode=None)


async def cmd_start(message: Message) -> None:
    await message.answer(start_text(), parse_mode=None)


async def cmd_mode(message: Message, command: CommandObject, assistant: Assistant) -> None:
    text = await assistant.switch_mode(message.chat.id, command.command.lower())
    await message.answer(text, parse_mode=None)


async def cmd_reset(message: Message, assistant: Assistant) -> None:
    await message.answer(await assistant.reset(message.chat.id), parse_mode=None)


async def cmd_settings(message: Message, command: CommandObject, assistant: Assistant) -> None:
    if command.args:
        text = await assistant.set_temperature(message.chat.id, command.args)
        await message.answer(text, parse_mode=None)
        return
    await message.answer(
        await assistant.settings_text(message.chat.id),
        parse_mode=None,
        reply_markup=temperature_keyboard(),
    )


async def on_temperature(callback: CallbackQuery, assistant: Assistant) -> None:
    raw = callback.data.removeprefix(TEMPERATURE_CALLBACK)
    chat_id = callback.message.chat.id
    text = await assistant.set_temperature(chat_id, raw)
    await callback.answer()
    await callback.message.answer(text, parse_mode=None)


async def unknown_command(message: Message) -> None:
    await message.answer(UNKNOWN_COMMAND, parse_mode=None)


async def on_text(message: Message, bot: Bot, assistant: Assistant) -> None:
    request_id = uuid.uuid4().hex[:12]
    logger.info(
        "Сообщение получено: request_id=%s chat_hash=%s length=%d",
        request_id,
        chat_hash(message.chat.id),
        len(message.text),
    )
    try:
        # Пока модель думает, Telegram показывает «печатает…».
        async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
            reply = await assistant.ask(message.chat.id, message.text, request_id)
    except Exception:
        logger.exception("Ошибка обработки: request_id=%s", request_id)
        await message.answer(INTERNAL_ERROR, parse_mode=None)
        return
    await send_long(message, reply.text)


async def on_other(message: Message) -> None:
    await message.answer(TEXT_ONLY, parse_mode=None)


def create_router() -> Router:
    """Новый роутер на каждый Dispatcher: aiogram не разрешает делить один роутер."""
    router = Router(name="assistant")
    # Бот работает только в личных чатах: группы и каналы игнорируются.
    router.message.filter(F.chat.type == ChatType.PRIVATE)
    router.callback_query.filter(F.message.chat.type == ChatType.PRIVATE)
    # Порядок важен: команды раньше общего обработчика текста.
    router.message.register(cmd_start, CommandStart())
    router.message.register(cmd_mode, Command(*MODES.keys()))
    router.message.register(cmd_reset, Command("reset"))
    router.message.register(cmd_settings, Command("settings"))
    router.callback_query.register(on_temperature, F.data.startswith(TEMPERATURE_CALLBACK))
    router.message.register(unknown_command, F.text.startswith("/"))
    router.message.register(on_text, F.text)
    router.message.register(on_other)
    return router
