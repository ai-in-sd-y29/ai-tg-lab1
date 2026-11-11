"""Малый настоящий запрос для проверки function calling; никаких секретов в выводе."""

import asyncio
import json
from datetime import datetime

from app.agent import prompt
from app.config import Settings
from app.llm import LLMClient, LLMError
from app.mcp_server import tool_definitions


async def main():
    s = Settings.load(require_llm=True)
    client = LLMClient(s)
    schemas = [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tool_definitions()
    ]
    rows = []
    try:
        for query in [
            "Какая сейчас погода в Санкт-Петербурге?",
            "Напомни вечером про лабораторную.",
            "Объясни что такое список Python.",
        ]:
            try:
                msg = await client.agent_turn(
                    [
                        {
                            "role": "system",
                            "content": prompt(
                                "study",
                                datetime.fromisoformat("2026-10-12T09:00:00+03:00"),
                                "Europe/Moscow",
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    0.0,
                    schemas,
                )
                calls = msg.get("tool_calls") or []
                rows.append(
                    {
                        "query": query,
                        "tools": [c["function"]["name"] for c in calls],
                        "content": msg.get("content"),
                    }
                )
            except LLMError as e:
                rows.append({"query": query, "error": str(e)})
    finally:
        await client.close()
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
