"""Независимый клиент MCP SDK: discovery, ресурс, чтение и отказ записи без capability.

Не использует ToolGateway или Telegram; создаёт и удаляет отдельную тестовую PostgreSQL.
python -m scripts.lab2_inspect [--live-weather]
"""

import argparse
import asyncio
import json
import secrets
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mcp import Client

from app.config import Settings
from app.mcp_client import server_parameters
from scripts.lab2_environment import temporary_database


async def inspect(settings, live_weather):
    result = {
        "date": datetime.now(UTC).isoformat(),
        "client": "mcp.Client, independent of ToolGateway",
    }
    async with Client(
        server_parameters(settings, secrets.token_urlsafe(48)), read_timeout_seconds=30
    ) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        result["tools"] = [t.model_dump(mode="json") for t in tools.tools]
        result["resources"] = [r.model_dump(mode="json") for r in resources.resources]
        week = await client.read_resource("schedule://current-week")
        result["week"] = json.loads(week.contents[0].text)
        schedule = await client.call_tool("get_schedule", {"date": "2026-11-09"})
        result["schedule"] = schedule.structured_content
        denied = await client.call_tool(
            "add_reminder",
            {
                "text": "Учебная проверка отказа",
                "remind_at": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            },
        )
        result["write_without_trusted_context_denied"] = denied.is_error
        if not denied.is_error:
            raise RuntimeError("Unexpected untrusted write")
        if live_weather:
            weather = await client.call_tool("get_weather", {"city": "Тбилиси, Грузия, Тбилиси"})
            result["weather"] = {"is_error": weather.is_error, "data": weather.structured_content}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-weather", action="store_true")
    args = parser.parse_args()
    settings = Settings.load()
    with temporary_database(settings) as isolated:
        result = asyncio.run(inspect(isolated, args.live_weather))
    Path("docs/lab2-mcp-check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Independent MCP: discovery, resource, schedule, denied write verified.")
    if args.live_weather:
        print(
            "Live weather:",
            "error" if result["weather"]["is_error"] else result["weather"]["data"]["status"],
        )


if __name__ == "__main__":
    main()
