"""Собственный MCP 2 сервер stdio; stdout зарезервирован протоколом."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from pydantic import ValidationError

from app.config import Settings
from app.db import create_pool
from app.lab2_models import INPUT_MODELS, OUTPUT_MODELS, AgentError
from app.lab2_repository import AgentRepository
from app.mcp_auth import verify_capability
from app.tool_services import ScheduleService, WeatherService

DESCRIPTIONS = {
    "get_weather": "Текущая погода, температура, ветер и условия в городе. "
    "Также для вопроса о зонте. city: именительный падеж; если указана страна, "
    "сохраняй её через запятую. Не угадывай город. Только чтение.",
    "get_schedule": "Занятия пользователя на конкретную дату YYYY-MM-DD. "
    "Сегодня/завтра вычисляй по доверенному текущему времени. Только чтение.",
    "add_reminder": "Подготовить напоминание: text — только содержание без слова Напомни; "
    "remind_at — точные ISO дата и время с offset сохранённого часового пояса. "
    "Неполную дату/время уточни. Приложение запросит подтверждение перед созданием.",
}


def tool_definitions() -> list[types.Tool]:
    return [
        types.Tool(
            name=name,
            description=description,
            input_schema=INPUT_MODELS[name].model_json_schema(),
            output_schema=OUTPUT_MODELS[name].model_json_schema(),
            annotations=types.ToolAnnotations(
                read_only_hint=name != "add_reminder", destructive_hint=False, idempotent_hint=True
            ),
        )
        for name, description in DESCRIPTIONS.items()
    ]


def make_server(settings: Settings, secret: str, *, weather=None, schedule=None, repository=None):
    weather = weather or WeatherService(settings.weather_timeout)
    schedule = schedule or ScheduleService(settings.schedule_path, settings.schedule_timezone)

    @asynccontextmanager
    async def lifespan(server):
        if repository is not None:
            yield repository
        else:
            pool = await create_pool(settings)
            try:
                repo = AgentRepository(pool)
                await repo.initialize()
                yield repo
            finally:
                await pool.close()

    async def list_tools(ctx, params):
        return types.ListToolsResult(tools=tool_definitions())

    async def call_tool(ctx, params):
        try:
            if params.name not in INPUT_MODELS:
                raise AgentError("unknown_tool")
            args = INPUT_MODELS[params.name].model_validate(params.arguments or {}, strict=True)
            if params.name == "get_weather":
                result = await weather.weather(args.city)
            elif params.name == "get_schedule":
                result = schedule.day(args.date)
            else:
                meta = (ctx.meta or {}).get("lab2", {})
                action_id = verify_capability(secret, meta)
                result = await ctx.lifespan_context.complete(action_id, args)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=result.model_dump_json())],
                structured_content=result.model_dump(mode="json"),
            )
        except (AgentError, ValidationError):
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="invalid_or_denied")], is_error=True
            )
        except Exception:
            # Не включаем аргументы/исключение/DSN в stderr MCP-библиотеки.
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="service_unavailable")], is_error=True
            )

    async def list_resources(ctx, params):
        return types.ListResourcesResult(
            resources=[
                types.Resource(
                    uri="schedule://current-week",
                    name="Расписание текущей недели",
                    mime_type="application/json",
                )
            ]
        )

    async def read_resource(ctx, params):
        if str(params.uri) != "schedule://current-week":
            raise ValueError("unknown_resource")
        return types.ReadResourceResult(
            contents=[
                types.TextResourceContents(
                    uri=params.uri,
                    mime_type="application/json",
                    text=schedule.week().model_dump_json(),
                )
            ]
        )

    return Server(
        "student-tools",
        version="2.0",
        lifespan=lifespan,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_resources=list_resources,
        on_read_resource=read_resource,
    )


async def main():
    # Не загружаем .env и LLM/Telegram ключи в дочерний процесс.
    values = dict(os.environ)
    values["BOT_TOKEN"] = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"
    settings = Settings.load(os.devnull, environ=values)
    server = make_server(settings, os.environ.get("MCP_ACTION_SECRET", ""))
    logging.basicConfig(level=logging.CRITICAL)
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
