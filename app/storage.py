"""Хранение пользовательских настроек и истории диалога в PostgreSQL.

Схема создаётся идемпотентно при запуске (CREATE ... IF NOT EXISTS), поэтому
повторный запуск не меняет и не удаляет существующие данные.
"""

from dataclasses import dataclass
from typing import Protocol

import asyncpg

from app.config import DEFAULT_TEMPERATURE
from app.llm import ChatMessage
from app.prompts import DEFAULT_MODE

SCHEMA = """
CREATE TABLE IF NOT EXISTS user_settings (
    chat_id     BIGINT PRIMARY KEY,
    mode        TEXT NOT NULL,
    temperature NUMERIC(2, 1) NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dialog_messages (
    id         BIGSERIAL PRIMARY KEY,
    chat_id    BIGINT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS dialog_messages_chat_id_id_idx
    ON dialog_messages (chat_id, id);
"""


@dataclass(frozen=True)
class UserSettings:
    mode: str = DEFAULT_MODE
    temperature: float = DEFAULT_TEMPERATURE


class Repository(Protocol):
    async def get_settings(self, chat_id: int) -> UserSettings: ...

    async def set_mode(self, chat_id: int, mode: str) -> None:
        """Сохраняет режим и очищает историю пользователя."""

    async def set_temperature(self, chat_id: int, temperature: float) -> None: ...

    async def clear_history(self, chat_id: int) -> None: ...

    async def get_history(self, chat_id: int, limit: int) -> list[ChatMessage]:
        """Последние limit сообщений в хронологическом порядке."""

    async def append_exchange(self, chat_id: int, user_text: str, assistant_text: str) -> None:
        """Сохраняет пару «запрос — успешный ответ» атомарно."""


class PostgresRepository:
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def ensure_schema(self) -> None:
        async with self._pool.acquire() as connection:
            # Блокировка защищает от гонки, если два процесса стартуют одновременно.
            async with connection.transaction():
                await connection.execute("SELECT pg_advisory_xact_lock(471001)")
                await connection.execute(SCHEMA)

    async def get_settings(self, chat_id: int) -> UserSettings:
        row = await self._pool.fetchrow(
            "SELECT mode, temperature FROM user_settings WHERE chat_id = $1", chat_id
        )
        if row is None:
            return UserSettings()
        return UserSettings(mode=row["mode"], temperature=float(row["temperature"]))

    async def set_mode(self, chat_id: int, mode: str) -> None:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    """
                    INSERT INTO user_settings (chat_id, mode, temperature)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (chat_id)
                    DO UPDATE SET mode = EXCLUDED.mode, updated_at = now()
                    """,
                    chat_id,
                    mode,
                    DEFAULT_TEMPERATURE,
                )
                await connection.execute("DELETE FROM dialog_messages WHERE chat_id = $1", chat_id)

    async def set_temperature(self, chat_id: int, temperature: float) -> None:
        await self._pool.execute(
            """
            INSERT INTO user_settings (chat_id, mode, temperature)
            VALUES ($1, $2, $3)
            ON CONFLICT (chat_id)
            DO UPDATE SET temperature = EXCLUDED.temperature, updated_at = now()
            """,
            chat_id,
            DEFAULT_MODE,
            temperature,
        )

    async def clear_history(self, chat_id: int) -> None:
        await self._pool.execute("DELETE FROM dialog_messages WHERE chat_id = $1", chat_id)

    async def get_history(self, chat_id: int, limit: int) -> list[ChatMessage]:
        if limit <= 0:
            return []
        rows = await self._pool.fetch(
            """
            SELECT role, content FROM (
                SELECT id, role, content FROM dialog_messages
                WHERE chat_id = $1
                ORDER BY id DESC
                LIMIT $2
            ) AS recent
            ORDER BY id
            """,
            chat_id,
            limit,
        )
        return [ChatMessage(row["role"], row["content"]) for row in rows]

    async def append_exchange(self, chat_id: int, user_text: str, assistant_text: str) -> None:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.executemany(
                    "INSERT INTO dialog_messages (chat_id, role, content) VALUES ($1, $2, $3)",
                    [(chat_id, "user", user_text), (chat_id, "assistant", assistant_text)],
                )
