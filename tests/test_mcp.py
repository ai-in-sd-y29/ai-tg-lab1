import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from mcp import Client

from app.config import Settings
from app.mcp_client import ToolGateway
from app.mcp_server import make_server
from app.tool_services import ScheduleService


@pytest.fixture
def mcp_server(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text(json.dumps({"days": {}}), encoding="utf-8")
    repo = AsyncMock()
    server = make_server(
        Settings(bot_token="unused", postgres_password="unused"),
        "s" * 48,
        repository=repo,
        schedule=ScheduleService(
            str(path), "Europe/Moscow", lambda: datetime(2027, 1, 1, tzinfo=UTC)
        ),
    )
    return server, repo


async def test_independent_mcp_discovery_resource_and_denied_write(mcp_server):
    server, repo = mcp_server
    async with Client(server) as client:
        gateway = ToolGateway(client, "s" * 48, 5)
        await gateway.discover()
        assert set(gateway.tools) == {"get_weather", "get_schedule", "add_reminder"}
        for tool in gateway.tools.values():
            assert tool.input_schema["additionalProperties"] is False
            assert tool.output_schema is not None
        result = await client.call_tool("get_schedule", {"date": "2026-10-12"})
        assert result.structured_content["lessons"] == []
        resource = await client.read_resource("schedule://current-week")
        week = json.loads(resource.contents[0].text)
        assert week["monday"] == "2026-12-28" and week["sunday"] == "2027-01-03"
        denied = await client.call_tool(
            "add_reminder", {"text": "test", "remind_at": "2027-01-02T09:00:00+03:00"}
        )
        assert denied.is_error
        repo.complete.assert_not_awaited()


@pytest.mark.parametrize(
    "name,args",
    [
        ("get_schedule", {"date": "not-a-date"}),
        ("get_schedule", {"date": "2026-10-12", "path": "C:/secret"}),
        ("add_reminder", {"text": "test", "remind_at": "2027-01-02T09:00:00+03:00", "owner": 22}),
    ],
)
async def test_server_revalidates_arguments(mcp_server, name, args):
    server, repo = mcp_server
    async with Client(server) as client:
        await client.list_tools()
        try:
            result = await client.call_tool(name, args)
            assert result.is_error
        except Exception as exc:
            # SDK вправе отклонить по опубликованной схеме ещё до dispatch.
            assert type(exc).__name__ in {"MCPError", "ValidationError"}
        repo.complete.assert_not_awaited()
