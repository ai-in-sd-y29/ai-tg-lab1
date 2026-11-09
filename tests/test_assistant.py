"""Обязательные сценарии лабораторной № 1 через настоящий Dispatcher aiogram.

Telegram и LLM API подменены: ответы модели задаются в тесте, а отправленные
боту методы Telegram перехватываются и проверяются по точному тексту.
"""

import pytest

from app.assistant import (
    ERROR_EMPTY,
    ERROR_REJECTED,
    ERROR_TIMEOUT,
    ERROR_UNAVAILABLE,
    TRUNCATED_NOTE,
    trim_history,
)
from app.config import DEFAULT_TEMPERATURE
from app.handlers.assistant import INTERNAL_ERROR, TEXT_ONLY, UNKNOWN_COMMAND, start_text
from app.llm import (
    ChatMessage,
    LLMEmptyResponseError,
    LLMRequestError,
    LLMResult,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.prompts import MODES
from app.text_split import TELEGRAM_LIMIT, utf16_len

ALICE, BOB = 1001, 2002


# 1. Обычное сообщение передаётся модели с инструкцией активного режима.
async def test_message_goes_to_llm_with_active_mode_prompt(harness_factory):
    # Arrange
    harness = harness_factory("Список — это изменяемая последовательность.")
    harness.llm.delay = 0.05  # «заметное» время ответа, чтобы появился статус
    # Act
    await harness.send(ALICE, "Что такое список в Python?")
    # Assert
    call = harness.llm.calls[0]
    assert call["messages"] == [
        ChatMessage("system", MODES["study"].system_prompt),
        ChatMessage("user", "Что такое список в Python?"),
    ]
    assert call["temperature"] == DEFAULT_TEMPERATURE
    assert harness.sent_texts(ALICE) == ["Список — это изменяемая последовательность."]
    assert harness.sent()[0].parse_mode is None
    assert harness.chat_actions()[0].action == "typing"
    assert harness.repository.history[ALICE] == [
        ChatMessage("user", "Что такое список в Python?"),
        ChatMessage("assistant", "Список — это изменяемая последовательность."),
    ]


async def test_translate_mode_sends_few_shot_before_history(harness_factory):
    # Arrange
    harness = harness_factory("Перевод (русский → английский):\nHello")
    await harness.send(ALICE, "/translate")
    # Act
    await harness.send(ALICE, "на английский: Привет")
    # Assert
    messages = harness.llm.calls[0]["messages"]
    mode = MODES["translate"]
    assert messages[0] == ChatMessage("system", mode.system_prompt)
    assert messages[1 : 1 + len(mode.few_shot)] == list(mode.few_shot)
    assert messages[-1] == ChatMessage("user", "на английский: Привет")
    assert len(mode.few_shot) >= 2


# 2. Истории двух пользователей не смешиваются.
async def test_histories_of_two_users_are_separate(harness_factory):
    # Arrange
    harness = harness_factory("Ответ Алисе 1", "Ответ Бобу 1", "Ответ Алисе 2")
    # Act
    await harness.send(ALICE, "Вопрос Алисы 1")
    await harness.send(BOB, "Вопрос Боба 1")
    await harness.send(ALICE, "Вопрос Алисы 2")
    # Assert
    bob_call, alice_second = harness.llm.calls[1], harness.llm.calls[2]
    assert bob_call["messages"][1:] == [ChatMessage("user", "Вопрос Боба 1")]
    assert alice_second["messages"][1:] == [
        ChatMessage("user", "Вопрос Алисы 1"),
        ChatMessage("assistant", "Ответ Алисе 1"),
        ChatMessage("user", "Вопрос Алисы 2"),
    ]
    assert harness.sent_texts(BOB) == ["Ответ Бобу 1"]


# 3. Сокращение истории без потери текущего запроса и нарушения порядка.
async def test_history_is_trimmed_by_message_count(harness_factory):
    # Arrange
    harness = harness_factory(history_max_messages=4)
    for i in range(1, 6):
        await harness.repository.append_exchange(ALICE, f"вопрос {i}", f"ответ {i}")
    # Act
    await harness.send(ALICE, "текущий вопрос")
    # Assert
    assert harness.llm.calls[0]["messages"][1:] == [
        ChatMessage("user", "вопрос 4"),
        ChatMessage("assistant", "ответ 4"),
        ChatMessage("user", "вопрос 5"),
        ChatMessage("assistant", "ответ 5"),
        ChatMessage("user", "текущий вопрос"),
    ]


async def test_history_is_trimmed_by_size_and_fits_limit(harness_factory):
    # Arrange
    limit = 8000
    harness = harness_factory(history_max_messages=50, history_max_chars=limit)
    for i in range(10):
        await harness.repository.append_exchange(ALICE, f"q{i} " + "в" * 700, f"a{i} " + "о" * 700)
    current = "текущий " + "я" * 3000
    # Act
    await harness.send(ALICE, current)
    # Assert
    messages = harness.llm.calls[0]["messages"]
    history = messages[1:-1]
    assert messages[-1] == ChatMessage("user", current)
    assert sum(len(m.content) for m in messages) <= limit
    assert history, "часть истории должна остаться"
    assert history[0].role == "user"
    assert [m.content[:3] for m in history] == [
        f"{role}{i} " for i in range(10 - len(history) // 2, 10) for role in ("q", "a")
    ]


def test_trim_history_never_starts_with_assistant_message():
    # Arrange
    history = [
        ChatMessage("user", "длинный вопрос " * 10),
        ChatMessage("assistant", "a1"),
        ChatMessage("user", "q2"),
        ChatMessage("assistant", "a2"),
    ]
    # Act
    trimmed = trim_history(history, max_messages=10, budget_chars=10)
    # Assert
    assert trimmed == [ChatMessage("user", "q2"), ChatMessage("assistant", "a2")]


def test_trim_history_with_no_budget_keeps_nothing():
    # Arrange
    history = [ChatMessage("user", "q"), ChatMessage("assistant", "a")]
    # Act / Assert
    assert trim_history(history, max_messages=10, budget_chars=-5) == []
    assert trim_history(history, max_messages=0, budget_chars=1000) == []


# 4. Смена режима очищает историю и сохраняет новую настройку.
async def test_mode_switch_clears_history_and_saves_mode(harness_factory):
    # Arrange
    harness = harness_factory()
    await harness.repository.append_exchange(ALICE, "старый вопрос", "старый ответ")
    await harness.repository.set_temperature(ALICE, 0.3)
    # Act
    await harness.send(ALICE, "/quiz")
    # Assert
    assert harness.repository.settings[ALICE].mode == "quiz"
    assert harness.repository.settings[ALICE].temperature == 0.3
    assert ALICE not in harness.repository.history
    assert harness.llm.calls == []
    assert harness.sent_texts(ALICE) == [
        "Режим «Тренажёр» (/quiz) включён. История очищена.\n"
        "Назовите тему, например «SQL JOIN», — я задам первый вопрос."
    ]


async def test_next_message_after_switch_uses_new_prompt(harness_factory):
    # Arrange
    harness = harness_factory()
    await harness.send(ALICE, "/quiz")
    # Act
    await harness.send(ALICE, "рекурсия")
    # Assert
    assert harness.llm.calls[0]["messages"][0].content == MODES["quiz"].system_prompt


# 5. /reset очищает только историю вызвавшего пользователя.
async def test_reset_clears_only_caller_history(harness_factory):
    # Arrange
    harness = harness_factory()
    await harness.repository.append_exchange(ALICE, "qa", "aa")
    await harness.repository.append_exchange(BOB, "qb", "ab")
    await harness.repository.set_mode(ALICE, "translate")
    await harness.repository.append_exchange(ALICE, "qa2", "aa2")
    await harness.repository.set_temperature(ALICE, 1.0)
    # Act
    await harness.send(ALICE, "/reset")
    # Assert
    assert ALICE not in harness.repository.history
    assert harness.repository.history[BOB] == [
        ChatMessage("user", "qb"),
        ChatMessage("assistant", "ab"),
    ]
    assert harness.repository.settings[ALICE].mode == "translate"
    assert harness.repository.settings[ALICE].temperature == 1.0
    assert harness.sent_texts(ALICE) == ["История диалога очищена. Режим и temperature сохранены."]


# 6. /settings принимает разрешённое и отклоняет некорректное значение.
async def test_settings_shows_current_values(harness_factory):
    # Arrange
    harness = harness_factory()
    # Act
    await harness.send(ALICE, "/settings")
    # Assert
    message = harness.sent()[0]
    assert message.text == (
        "Текущие настройки:\n"
        "Режим: Учёба (/study)\n"
        "Модель: test-model\n"
        "Temperature: 0.7\n\n"
        "Выберите temperature кнопкой ниже или командой /settings 0.3 "
        "(допустимо: 0.0, 0.3, 0.7, 1.0)."
    )
    buttons = message.reply_markup.inline_keyboard[0]
    assert [b.text for b in buttons] == ["0.0", "0.3", "0.7", "1.0"]


async def test_settings_accepts_allowed_value_and_applies_it(harness_factory):
    # Arrange
    harness = harness_factory()
    # Act
    await harness.send(ALICE, "/settings 0,3")
    await harness.send(ALICE, "вопрос")
    # Assert
    assert harness.sent_texts(ALICE)[0] == (
        "Temperature = 0.3 сохранена и применится со следующего запроса."
    )
    assert harness.repository.settings[ALICE].temperature == 0.3
    assert harness.llm.calls[0]["temperature"] == 0.3
    assert BOB not in harness.repository.settings


@pytest.mark.parametrize("value", ["0.5", "2", "-0.3", "abc", "nan", "inf"])
async def test_settings_rejects_invalid_value_before_api(harness_factory, value):
    # Arrange
    harness = harness_factory()
    # Act
    await harness.send(ALICE, f"/settings {value}")
    # Assert
    assert harness.sent_texts(ALICE) == [
        "Недопустимое значение temperature. Выберите одно из: 0.0, 0.3, 0.7, 1.0."
    ]
    assert ALICE not in harness.repository.settings
    assert harness.llm.calls == []


async def test_settings_button_saves_value(harness_factory):
    # Arrange
    harness = harness_factory()
    # Act
    await harness.press(ALICE, "temperature:1.0")
    await harness.press(ALICE, "temperature:0.9")
    # Assert
    assert harness.repository.settings[ALICE].temperature == 1.0
    assert harness.sent_texts(ALICE) == [
        "Temperature = 1.0 сохранена и применится со следующего запроса.",
        "Недопустимое значение temperature. Выберите одно из: 0.0, 0.3, 0.7, 1.0.",
    ]
    assert len(harness.callback_answers()) == 2


# 7. Ошибка LLM преобразуется в безопасное пользовательское сообщение.
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LLMTimeoutError("нет ответа за 60 с"), ERROR_TIMEOUT),
        (LLMUnavailableError("HTTP 503"), ERROR_UNAVAILABLE),
        (LLMRequestError("HTTP 401: invalid key sk-test-secret", 401), ERROR_REJECTED),
        (LLMEmptyResponseError("в ответе нет текста"), ERROR_EMPTY),
    ],
)
async def test_llm_error_becomes_safe_message(harness_factory, error, expected):
    # Arrange
    harness = harness_factory(error)
    await harness.repository.append_exchange(ALICE, "прошлый вопрос", "прошлый ответ")
    # Act
    await harness.send(ALICE, "новый вопрос")
    # Assert
    assert harness.sent_texts(ALICE) == [expected]
    assert harness.repository.history[ALICE] == [
        ChatMessage("user", "прошлый вопрос"),
        ChatMessage("assistant", "прошлый ответ"),
    ]
    assert "sk-test-secret" not in expected
    assert MODES["study"].system_prompt[:40] not in expected


async def test_unexpected_error_hides_details(harness_factory):
    # Arrange
    harness = harness_factory(RuntimeError("password=secret traceback"))
    # Act
    await harness.send(ALICE, "вопрос")
    # Assert
    assert harness.sent_texts(ALICE) == [INTERNAL_ERROR]


async def test_truncated_answer_is_marked(harness_factory):
    # Arrange
    result = LLMResult(text="Начало ответа", model="test-model", finish_reason="length")
    harness = harness_factory(result)
    # Act
    await harness.send(ALICE, "вопрос")
    # Assert
    assert harness.sent_texts(ALICE) == ["Начало ответа" + TRUNCATED_NOTE]


# 8. Ответ длиннее 4096 символов разбивается и отправляется без потери текста.
async def test_long_answer_is_split_without_loss(harness_factory):
    # Arrange
    paragraphs = [f"Абзац {i}: " + "слово " * 150 for i in range(12)]
    answer = "\n\n".join(p.strip() for p in paragraphs) + " 🎉" * 50
    harness = harness_factory(answer)
    # Act
    await harness.send(ALICE, "длинный вопрос")
    # Assert
    parts = harness.sent_texts(ALICE)
    assert len(parts) >= 3
    assert all(0 < utf16_len(p) <= TELEGRAM_LIMIT for p in parts)
    assert "".join(parts) == answer
    assert parts[0].startswith("Абзац 0:")


# Команды и фильтры.
async def test_start_lists_commands(harness_factory):
    # Arrange
    harness = harness_factory()
    # Act
    await harness.send(ALICE, "/start")
    # Assert
    assert harness.sent_texts(ALICE) == [start_text()]
    for command in ("/study", "/translate", "/quiz", "/settings", "/reset"):
        assert command in start_text()
    assert harness.llm.calls == []


async def test_unknown_command_is_not_sent_to_llm(harness_factory):
    # Arrange
    harness = harness_factory()
    # Act
    await harness.send(ALICE, "/weather")
    # Assert
    assert harness.sent_texts(ALICE) == [UNKNOWN_COMMAND]
    assert harness.llm.calls == []


async def test_group_messages_are_ignored(harness_factory):
    # Arrange
    harness = harness_factory()
    # Act
    await harness.send(-100500, "вопрос в группе", chat_type="group")
    # Assert
    assert harness.sent() == []
    assert harness.llm.calls == []


async def test_non_text_message_gets_hint(harness_factory):
    # Arrange
    harness = harness_factory()
    # Act
    await harness.send(ALICE, None)
    # Assert
    assert harness.sent_texts(ALICE) == [TEXT_ONLY]
    assert harness.llm.calls == []
