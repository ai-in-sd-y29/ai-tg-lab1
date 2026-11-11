"""Транзакционные подтверждения и идемпотентность в PostgreSQL."""

import json
import uuid
from datetime import UTC, datetime, timedelta

import asyncpg

from app.lab2_models import AgentError, ReminderArgs, ReminderResult, normalize_reminder

SCHEMA = """
CREATE TABLE IF NOT EXISTS lab2_profiles (
    chat_id bigint PRIMARY KEY, timezone text
);
CREATE TABLE IF NOT EXISTS lab2_actions (
    id uuid PRIMARY KEY, chat_id bigint NOT NULL, update_id bigint NOT NULL,
    text text NOT NULL CHECK (length(text) BETWEEN 1 AND 500),
    remind_at timestamptz NOT NULL, timezone text NOT NULL,
    expires_at timestamptz NOT NULL,
    status text NOT NULL
        CHECK (status IN ('pending','confirmed','completed','cancelled','expired')),
    UNIQUE(chat_id, update_id)
);
CREATE INDEX IF NOT EXISTS lab2_actions_owner ON lab2_actions(chat_id, status);
CREATE TABLE IF NOT EXISTS lab2_reminders (
    id uuid PRIMARY KEY, action_id uuid UNIQUE NOT NULL REFERENCES lab2_actions(id),
    chat_id bigint NOT NULL, text text NOT NULL,
    remind_at timestamptz NOT NULL, timezone text NOT NULL
);
CREATE TABLE IF NOT EXISTS lab2_audit (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    request_id text NOT NULL DEFAULT '',
    chat_id bigint NOT NULL, occurred_at timestamptz NOT NULL,
    action text NOT NULL, tool_name text, arguments jsonb NOT NULL,
    validation text NOT NULL, outcome text NOT NULL, reason text NOT NULL,
    duration_ms integer NOT NULL
);
CREATE INDEX IF NOT EXISTS lab2_audit_owner ON lab2_audit(chat_id,id DESC);
ALTER TABLE lab2_audit ADD COLUMN IF NOT EXISTS request_id text NOT NULL DEFAULT '';
"""


class AgentRepository:
    def __init__(self, pool: asyncpg.Pool, clock=None):
        self.pool = pool
        self.clock = clock or (lambda: datetime.now(UTC))

    async def initialize(self):
        await self.pool.execute(SCHEMA)

    async def timezone(self, owner: int) -> str | None:
        return await self.pool.fetchval(
            "SELECT timezone FROM lab2_profiles WHERE chat_id=$1", owner
        )

    async def set_timezone(self, owner: int, zone: str):
        await self.pool.execute(
            "INSERT INTO lab2_profiles(chat_id,timezone) VALUES ($1,$2) "
            "ON CONFLICT(chat_id) DO UPDATE SET timezone=EXCLUDED.timezone",
            owner,
            zone,
        )

    async def prepare(self, owner: int, update_id: int, args: ReminderArgs, zone: str):
        args = normalize_reminder(args, zone, self.clock())
        async with self.pool.acquire() as c, c.transaction():
            await c.execute("SELECT pg_advisory_xact_lock($1)", owner)
            old = await c.fetchrow(
                "SELECT * FROM lab2_actions WHERE chat_id=$1 AND update_id=$2", owner, update_id
            )
            if old:
                return dict(old)
            await c.execute(
                "UPDATE lab2_actions SET status='cancelled' WHERE chat_id=$1 "
                "AND status IN ('pending','confirmed')",
                owner,
            )
            row = await c.fetchrow(
                "INSERT INTO lab2_actions(id,chat_id,update_id,text,remind_at,timezone,"
                "expires_at,status) "
                "VALUES($1,$2,$3,$4,$5,$6,$7,'pending') RETURNING *",
                uuid.uuid4(),
                owner,
                update_id,
                args.text,
                datetime.fromisoformat(args.remind_at),
                zone,
                self.clock() + timedelta(seconds=300),
            )
            return dict(row)

    async def confirm(self, owner: int, action_id: str, cancel: bool = False):
        try:
            identity = uuid.UUID(action_id)
        except ValueError:
            raise AgentError("confirmation_missing") from None
        async with self.pool.acquire() as c, c.transaction():
            await c.execute("SELECT pg_advisory_xact_lock($1)", owner)
            row = await c.fetchrow(
                "SELECT * FROM lab2_actions WHERE id=$1 AND chat_id=$2 FOR UPDATE", identity, owner
            )
            if not row:
                raise AgentError("confirmation_missing")
            status = row["status"]
            if status == "completed":
                return dict(row)
            if row["expires_at"] <= self.clock() or row["remind_at"] <= self.clock():
                # Ошибка возвращается после commit, чтобы expired сохранился.
                status = "expired"
            elif cancel and status in {"pending", "confirmed"}:
                status = "cancelled"
            elif status == "pending":
                status = "confirmed"
            await c.execute("UPDATE lab2_actions SET status=$2 WHERE id=$1", identity, status)
            result = dict(row)
            result["status"] = status
        if status in {"expired", "cancelled"}:
            raise AgentError("confirmation_" + status)
        return result

    async def complete(self, action_id: str, args: ReminderArgs) -> ReminderResult:
        async with self.pool.acquire() as c, c.transaction():
            identity = uuid.UUID(action_id)
            # Тот же порядок блокировок, что при prepare/confirm.
            owner = await c.fetchval("SELECT chat_id FROM lab2_actions WHERE id=$1", identity)
            if owner is None:
                raise AgentError("confirmation_missing")
            await c.execute("SELECT pg_advisory_xact_lock($1)", owner)
            row = await c.fetchrow("SELECT * FROM lab2_actions WHERE id=$1 FOR UPDATE", identity)
            if row["status"] not in {"confirmed", "completed"}:
                raise AgentError("confirmation_required")
            if row["text"] != args.text or row["remind_at"] != datetime.fromisoformat(
                args.remind_at
            ):
                raise AgentError("confirmation_mismatch")
            previous = await c.fetchrow("SELECT * FROM lab2_reminders WHERE action_id=$1", identity)
            if previous:
                return self.result(previous, False)
            if row["expires_at"] <= self.clock():
                raise AgentError("confirmation_expired")
            normalize_reminder(args, row["timezone"], self.clock())
            reminder = await c.fetchrow(
                "INSERT INTO lab2_reminders(id,action_id,chat_id,text,remind_at,timezone) "
                "VALUES($1,$2,$3,$4,$5,$6) RETURNING *",
                uuid.uuid4(),
                identity,
                row["chat_id"],
                row["text"],
                row["remind_at"],
                row["timezone"],
            )
            await c.execute("UPDATE lab2_actions SET status='completed' WHERE id=$1", identity)
            return self.result(reminder, True)

    @staticmethod
    def result(row, created: bool):
        from zoneinfo import ZoneInfo

        return ReminderResult(
            id=str(row["id"]),
            timezone=row["timezone"],
            created=created,
            remind_at=row["remind_at"].astimezone(ZoneInfo(row["timezone"])).isoformat(),
        )

    async def audit(
        self, owner, action, tool, arguments, validation, outcome, reason, duration, request_id=None
    ):
        await self.pool.execute(
            "INSERT INTO lab2_audit(chat_id,occurred_at,action,tool_name,arguments,validation,"
            "outcome,reason,duration_ms,request_id) VALUES($1,$2,$3,$4,$5::jsonb,$6,$7,$8,$9,$10)",
            owner,
            self.clock(),
            action,
            tool,
            json.dumps(arguments, ensure_ascii=False),
            validation,
            outcome,
            reason,
            int(duration),
            request_id or uuid.uuid4().hex,
        )

    async def last_audit(self, owner):
        row = await self.pool.fetchrow(
            "SELECT occurred_at,action,tool_name,arguments,validation,outcome,reason,duration_ms "
            "FROM lab2_audit WHERE chat_id=$1 ORDER BY id DESC LIMIT 1",
            owner,
        )
        return dict(row) if row else None
