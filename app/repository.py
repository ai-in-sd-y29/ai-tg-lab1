"""Постоянные настройки и история; все значения SQL параметризованы."""

from dataclasses import dataclass

import asyncpg


@dataclass(frozen=True)
class Profile:
    mode: str = "study"
    temperature: float = 0.3


SCHEMA = """
CREATE TABLE IF NOT EXISTS lab1_users (
    chat_id bigint PRIMARY KEY,
    mode text NOT NULL DEFAULT 'study' CHECK (mode IN ('study', 'translate', 'quiz')),
    temperature double precision NOT NULL DEFAULT 0.3
        CHECK (temperature IN (0.0, 0.3, 0.7, 1.0))
);
CREATE TABLE IF NOT EXISTS lab1_turns (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    chat_id bigint NOT NULL REFERENCES lab1_users(chat_id) ON DELETE CASCADE,
    user_text text NOT NULL,
    assistant_text text
);
ALTER TABLE lab1_turns ALTER COLUMN assistant_text DROP NOT NULL;
CREATE INDEX IF NOT EXISTS lab1_turns_chat_id ON lab1_turns(chat_id, id);
"""


class Repository:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def initialize(self) -> None:
        await self.pool.execute(SCHEMA)

    async def profile(self, chat_id: int) -> Profile:
        await self.pool.execute(
            "INSERT INTO lab1_users(chat_id) VALUES ($1) ON CONFLICT DO NOTHING",
            chat_id,
        )
        row = await self.pool.fetchrow(
            "SELECT mode, temperature FROM lab1_users WHERE chat_id=$1",
            chat_id,
        )
        return Profile(row["mode"], row["temperature"])

    async def history(self, chat_id: int, max_turns: int) -> list[dict[str, str]]:
        rows = await self.pool.fetch(
            "SELECT user_text, assistant_text FROM "
            "(SELECT id, user_text, assistant_text FROM lab1_turns "
            "WHERE chat_id=$1 AND assistant_text IS NOT NULL "
            "ORDER BY id DESC LIMIT $2) recent ORDER BY id",
            chat_id,
            max_turns,
        )
        return [
            message
            for row in rows
            for message in (
                {"role": "user", "content": row["user_text"]},
                {"role": "assistant", "content": row["assistant_text"]},
            )
        ]

    async def save_turn(
        self, chat_id: int, question: str, answer: str | None, max_turns: int
    ) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "INSERT INTO lab1_turns(chat_id, user_text, assistant_text) VALUES ($1,$2,$3)",
                chat_id,
                question,
                answer,
            )
            await connection.execute(
                "DELETE FROM lab1_turns WHERE chat_id=$1 AND id NOT IN "
                "(SELECT id FROM lab1_turns WHERE chat_id=$1 ORDER BY id DESC LIMIT $2)",
                chat_id,
                max_turns,
            )

    async def reset(self, chat_id: int) -> None:
        await self.pool.execute("DELETE FROM lab1_turns WHERE chat_id=$1", chat_id)

    async def change_mode(self, chat_id: int, mode: str) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "UPDATE lab1_users SET mode=$2 WHERE chat_id=$1", chat_id, mode
            )
            await connection.execute("DELETE FROM lab1_turns WHERE chat_id=$1", chat_id)

    async def set_temperature(self, chat_id: int, temperature: float) -> None:
        await self.pool.execute(
            "UPDATE lab1_users SET temperature=$2 WHERE chat_id=$1",
            chat_id,
            temperature,
        )
