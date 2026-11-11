"""MCP discovery и проверка контрактов; доверенные metadata отделены от arguments."""

import logging
import secrets
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

import jsonschema
from mcp import Client, StdioServerParameters

from app.config import Settings
from app.lab2_models import AgentError
from app.mcp_auth import capability

logger = logging.getLogger(__name__)


class ToolGateway:
    def __init__(self, client, secret: str, timeout: int):
        self.client, self.secret, self.timeout = client, secret, timeout
        self.tools = {}
        self.resources = []
        self.available = False

    async def discover(self):
        tools = await self.client.list_tools()
        resources = await self.client.list_resources()
        self.tools = {tool.name: tool for tool in tools.tools}
        for tool in self.tools.values():
            if not tool.output_schema:
                raise AgentError("missing_output_schema")
            jsonschema.Draft202012Validator.check_schema(tool.input_schema)
            jsonschema.Draft202012Validator.check_schema(tool.output_schema)
        self.resources = [str(r.uri) for r in resources.resources]
        if "schedule://current-week" not in self.resources:
            raise AgentError("missing_resource")
        self.available = True

    def schemas(self):
        if not self.available:
            return []
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in self.tools.values()
        ]

    def validate(self, name, args):
        if not self.available:
            raise AgentError("mcp_unavailable")
        if name not in self.tools:
            raise AgentError("unknown_tool")
        try:
            jsonschema.validate(args, self.tools[name].input_schema)
        except jsonschema.ValidationError:
            raise AgentError("invalid_arguments") from None

    async def call(self, name, args, action_id=None):
        self.validate(name, args)
        meta = {"lab2": capability(self.secret, action_id)} if action_id else None
        try:
            result = await self.client.call_tool(
                name, args, read_timeout_seconds=self.timeout, meta=meta
            )
            if result.is_error:
                raise AgentError("tool_failed")
            data = result.structured_content
            jsonschema.validate(data, self.tools[name].output_schema)
            return data
        except AgentError:
            raise
        except jsonschema.ValidationError:
            raise AgentError("invalid_tool_result") from None
        except Exception:
            self.available = False
            logger.warning("MCP request failed; code=connection_or_protocol")
            raise AgentError("mcp_unavailable") from None


def server_parameters(settings: Settings, secret: str):
    env = {
        "POSTGRES_HOST": settings.postgres_host,
        "POSTGRES_PORT": str(settings.postgres_port),
        "POSTGRES_DB": settings.postgres_db,
        "POSTGRES_USER": settings.postgres_user,
        "POSTGRES_PASSWORD": settings.postgres_password,
        "SCHEDULE_PATH": settings.schedule_path,
        "SCHEDULE_TIMEZONE": settings.schedule_timezone,
        "WEATHER_TIMEOUT": str(settings.weather_timeout),
        "MCP_ACTION_SECRET": secret,
        "PYTHONUTF8": "1",
    }
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
    )


@asynccontextmanager
async def connect_gateway(settings: Settings):
    secret = secrets.token_urlsafe(48)
    gateway = ToolGateway(None, secret, settings.mcp_timeout)
    # Стек открывается/закрывается одной задачей: важно для anyio stdio.
    async with AsyncExitStack() as stack:
        try:
            client = await stack.enter_async_context(
                Client(
                    server_parameters(settings, secret), read_timeout_seconds=settings.mcp_timeout
                )
            )
            gateway.client = client
            await gateway.discover()
        except Exception:
            logger.warning("MCP connection failed; code=startup_or_discovery")
        yield gateway
