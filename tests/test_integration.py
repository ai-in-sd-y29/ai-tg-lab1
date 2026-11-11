"""RUN_INTEGRATION=1 python -m pytest -m integration -v.

Тест создаёт собственные контейнер и volume; пользовательскую БД не использует.
"""

import asyncio
import json
import os
import secrets
import socket
import time
from dataclasses import replace

import asyncpg
import pytest

from app.config import Settings
from app.db import create_pool
from app.health import HealthState, health_result
from scripts.common import CommandError, run_command

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Включите RUN_INTEGRATION=1 для Docker-тестов",
    ),
]


@pytest.fixture
def database():
    name = "itmo-test-" + secrets.token_hex(6)
    volume = name + "-data"
    password = secrets.token_urlsafe(24)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    def docker(*args):
        return run_command(["docker", *args], label="Интеграционный Docker-тест", timeout=180)

    def wait_until_ready():
        deadline = time.monotonic() + 60
        while True:
            try:
                docker("exec", name, "pg_isready", "-h", "127.0.0.1", "-U", "bot", "-d", "bot")
                return
            except CommandError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)

    docker("info")
    docker("volume", "create", volume)
    try:
        docker(
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{port}:5432",
            "-e",
            "POSTGRES_DB=bot",
            "-e",
            "POSTGRES_USER=bot",
            "-e",
            f"POSTGRES_PASSWORD={password}",
            "--mount",
            f"type=volume,source={volume},target=/var/lib/postgresql/data",
            "postgres:16-bookworm",
        )
        info = json.loads(docker("inspect", name))[0]
        assert int(info["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"]) == port
        wait_until_ready()
        yield (
            Settings(
                bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                postgres_password=password,
                postgres_port=port,
            ),
            docker,
            wait_until_ready,
            name,
        )
    finally:
        try:
            docker("rm", "-f", name)
        finally:
            docker("volume", "rm", volume)


async def test_real_db_password_restart_health_and_persistence(database):
    # Arrange
    settings, docker, wait_until_ready, name = database
    pool = await create_pool(settings)
    task = asyncio.create_task(asyncio.Event().wait())
    state = HealthState(pool=pool, initialized=True, polling_task=task)
    try:
        await pool.execute("CREATE TABLE integration_probe (value text NOT NULL)")
        await pool.execute("INSERT INTO integration_probe VALUES ($1)", "Сохранилось 👋")
        # Act / Assert: реальное подключение и ошибка пароля.
        assert (await health_result(state))[0] == 200
        with pytest.raises(asyncpg.InvalidPasswordError):
            await create_pool(replace(settings, postgres_password="wrong-password"))
        # Act / Assert: потеря соединения и сохранность данных после запуска.
        await asyncio.to_thread(docker, "stop", name)
        assert (await health_result(state))[0] == 503
        await asyncio.to_thread(docker, "start", name)
        await asyncio.to_thread(wait_until_ready)
        for _ in range(60):
            if (await health_result(state))[0] == 200:
                break
            await asyncio.sleep(1)
        assert (await health_result(state))[0] == 200
        assert await pool.fetchval("SELECT value FROM integration_probe") == "Сохранилось 👋"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await pool.close()


async def test_dialogue_repository_isolation_restart_and_pruning(database):
    from app.repository import Profile, Repository

    settings, docker, wait_until_ready, name = database
    pool = await create_pool(settings)
    try:
        repo = Repository(pool)
        await repo.initialize()
        await repo.initialize()  # Повторный запуск не уничтожает данные.
        assert await repo.profile(11) == Profile()
        assert await repo.profile(22) == Profile()
        await repo.set_temperature(11, 0.7)
        await repo.change_mode(11, "translate")
        await repo.save_turn(11, "Привет", "Hello", 2)
        await repo.save_turn(22, "Вопрос 2", "Ответ 2", 2)
        await pool.close()
        await asyncio.to_thread(docker, "restart", name)
        await asyncio.to_thread(wait_until_ready)
        pool = await create_pool(settings)
        repo = Repository(pool)
        await repo.initialize()
        assert await repo.profile(11) == Profile("translate", 0.7)
        assert await repo.profile(22) == Profile()
        assert await repo.history(11, 10) == [
            {"role": "user", "content": "Привет"},
            {"role": "assistant", "content": "Hello"},
        ]
        await repo.save_turn(11, "Ещё", "More", 2)
        await repo.save_turn(11, "Последний", "Last", 2)
        assert [m["content"] for m in await repo.history(11, 10)] == [
            "Ещё",
            "More",
            "Последний",
            "Last",
        ]
        await repo.change_mode(11, "quiz")
        assert await repo.history(11, 10) == []
        assert await repo.profile(11) == Profile("quiz", 0.7)
        assert len(await repo.history(22, 10)) == 2
        await repo.reset(22)
        assert await repo.history(22, 10) == []
        assert await repo.profile(22) == Profile()
        await repo.save_turn(22, "Вопрос при сбое", None, 2)
        assert await repo.history(22, 10) == []
        assert (
            await pool.fetchval(
                "SELECT user_text FROM lab1_turns WHERE chat_id=$1",
                22,
            )
            == "Вопрос при сбое"
        )
    finally:
        await pool.close()


async def test_lab2_real_postgres_concurrency_isolation_expiry_and_restart(database):
    from datetime import UTC, datetime, timedelta

    from app.lab2_models import AgentError, ReminderArgs
    from app.lab2_repository import AgentRepository

    settings, docker, wait_until_ready, name = database
    pool = await create_pool(settings)
    try:
        repo = AgentRepository(pool)
        await repo.initialize()
        await repo.initialize()
        await repo.set_timezone(11, "Europe/Moscow")
        assert await repo.timezone(22) is None
        future = datetime.now(UTC) + timedelta(days=1)
        from zoneinfo import ZoneInfo

        args = ReminderArgs(
            text="Учебное напоминание",
            remind_at=future.astimezone(ZoneInfo("Europe/Moscow")).isoformat(),
        )
        prepared = await asyncio.gather(
            *(repo.prepare(11, 100, args, "Europe/Moscow") for _ in range(8))
        )
        assert len({str(r["id"]) for r in prepared}) == 1
        assert await pool.fetchval("SELECT count(*) FROM lab2_actions") == 1
        identity = str(prepared[0]["id"])
        with pytest.raises(AgentError, match="confirmation_missing"):
            await repo.confirm(22, identity)
        with pytest.raises(AgentError, match="confirmation_required"):
            await repo.complete(identity, args)
        await pool.close()
        await asyncio.to_thread(docker, "restart", name)
        await asyncio.to_thread(wait_until_ready)
        pool = await create_pool(settings)
        repo = AgentRepository(pool)
        assert await repo.timezone(11) == "Europe/Moscow"
        await asyncio.gather(*(repo.confirm(11, identity) for _ in range(8)))
        results = await asyncio.gather(*(repo.complete(identity, args) for _ in range(8)))
        assert sum(r.created for r in results) == 1
        assert len({r.id for r in results}) == 1
        assert await pool.fetchval("SELECT count(*) FROM lab2_reminders") == 1
        with pytest.raises(AgentError, match="confirmation_mismatch"):
            await repo.complete(identity, args.model_copy(update={"text": "Другой текст"}))
        expired = await repo.prepare(11, 101, args, "Europe/Moscow")
        await pool.execute(
            "UPDATE lab2_actions SET expires_at=now()-interval '1 second' WHERE id=$1",
            expired["id"],
        )
        with pytest.raises(AgentError, match="confirmation_expired"):
            await repo.confirm(11, str(expired["id"]))
        assert (
            await pool.fetchval("SELECT status FROM lab2_actions WHERE id=$1", expired["id"])
            == "expired"
        )
        cancelled = await repo.prepare(11, 102, args, "Europe/Moscow")
        with pytest.raises(AgentError, match="confirmation_cancelled"):
            await repo.confirm(11, str(cancelled["id"]), True)
        with pytest.raises(AgentError, match="confirmation_required"):
            await repo.complete(str(cancelled["id"]), args)
        await repo.audit(11, "respond", None, {}, "passed", "success", "Обычный ответ", 1)
        assert await repo.last_audit(22) is None
        assert (await repo.last_audit(11))["reason"] == "Обычный ответ"
        assert await pool.fetchval("SELECT count(*) FROM lab2_reminders") == 1
    finally:
        await pool.close()


async def test_lab2_real_stdio_mcp_with_postgres(database):
    from datetime import UTC, datetime, timedelta
    from zoneinfo import ZoneInfo

    from mcp import Client

    from app.lab2_models import ReminderArgs
    from app.lab2_repository import AgentRepository
    from app.mcp_client import ToolGateway, server_parameters

    settings, *_ = database
    pool = await create_pool(settings)
    secret = secrets.token_urlsafe(48)
    try:
        repo = AgentRepository(pool)
        await repo.initialize()
        async with Client(server_parameters(settings, secret), read_timeout_seconds=10) as client:
            gateway = ToolGateway(client, secret, 10)
            await gateway.discover()
            assert len(gateway.tools) == 3
            resource = await client.read_resource("schedule://current-week")
            assert len(json.loads(resource.contents[0].text)["days"]) == 7
            result = await gateway.call("get_schedule", {"date": "2026-11-09"})
            assert result["lessons"] and result["lessons"][0]["title"]
            future = datetime.now(UTC) + timedelta(days=1)
            args = ReminderArgs(
                text="Тест MCP", remind_at=future.astimezone(ZoneInfo("Europe/Moscow")).isoformat()
            )
            denied = await client.call_tool("add_reminder", args.model_dump())
            assert denied.is_error
            row = await repo.prepare(11, 100, args, "Europe/Moscow")
            await repo.confirm(11, str(row["id"]))
            results = await asyncio.gather(
                *(gateway.call("add_reminder", args.model_dump(), str(row["id"])) for _ in range(2))
            )
            assert sum(r["created"] for r in results) == 1
            assert results[0]["id"] == results[1]["id"]
            assert await pool.fetchval("SELECT count(*) FROM lab2_reminders") == 1
    finally:
        await pool.close()
