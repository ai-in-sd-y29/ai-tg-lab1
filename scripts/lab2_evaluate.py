"""Настоящие LLM-запросы к неизменённому набору. Инструменты обнаружены по stdio MCP.

python -m scripts.lab2_evaluate --output docs/lab2-routing-baseline.json
Для повторной проверки ошибок: --retry-errors docs/lab2-routing-baseline.json.
Напоминания в эксперименте только подготавливаются, записей и Telegram-отправок нет.
"""

import argparse
import asyncio
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from app.agent import TextDecision, prompt
from app.config import Settings
from app.lab2_models import INPUT_MODELS, normalize_reminder
from app.llm import LLMClient
from app.mcp_client import connect_gateway
from scripts.lab2_environment import temporary_database


async def evaluate(settings, args):
    path = Path("docs/evaluation/lab2-routing.json")
    raw = await asyncio.to_thread(path.read_bytes)
    dataset = json.loads(raw)
    context = dataset["evaluation_context"]
    now = datetime.fromisoformat(context["now"])
    cases = dataset["cases"]
    if args.retry_errors:
        previous = json.loads(
            await asyncio.to_thread(Path(args.retry_errors).read_text, encoding="utf-8")
        )
        failed = {row["id"] for row in previous["results"] if not row["success"]}
        cases = [case for case in cases if case["id"] in failed]
    client = LLMClient(settings)
    results = []
    async with connect_gateway(settings) as gateway:
        if not gateway.available:
            raise RuntimeError("MCP discovery failed")
        try:
            for case in cases:
                row = {
                    **case,
                    "actual_action": None,
                    "actual_arguments": {},
                    "action_correct": False,
                    "arguments_correct": False,
                    "success": False,
                }
                try:
                    message = await client.agent_turn(
                        [
                            {
                                "role": "system",
                                "content": prompt("study", now, context["timezone"]),
                            },
                            {"role": "user", "content": case["query"]},
                        ],
                        0.0,
                        gateway.schemas(),
                    )
                    row["model_message"] = message
                    calls = message.get("tool_calls") or []
                    if calls:
                        if len(calls) != 1:
                            raise ValueError("multiple_calls")
                        call = calls[0]["function"]
                        name = call["name"]
                        arguments = json.loads(call["arguments"])
                        gateway.validate(name, arguments)
                        parsed = INPUT_MODELS[name].model_validate(arguments, strict=True)
                        if name == "add_reminder":
                            parsed = normalize_reminder(parsed, context["timezone"], now)
                        row.update(actual_action=name, actual_arguments=parsed.model_dump())
                    else:
                        decision = TextDecision.model_validate_json(message["content"])
                        row.update(actual_action=decision.action)
                    row["action_correct"] = row["actual_action"] == case["expected_action"]
                    row["arguments_correct"] = row["actual_arguments"] == case["expected_arguments"]
                    row["success"] = row["action_correct"] and row["arguments_correct"]
                except Exception as exc:
                    row["error"] = type(exc).__name__  # тела/ключи/DSN не сохраняются
                results.append(row)
                print(
                    case["id"],
                    row["actual_action"],
                    "PASS" if row["success"] else "FAIL",
                    flush=True,
                )
        finally:
            await client.close()
    correct = sum(r["action_correct"] for r in results)
    correct_args = sum(
        r["action_correct"] and r["arguments_correct"]
        for r in results
        if r["expected_action"] in INPUT_MODELS
    )
    report = {
        "date": datetime.now(UTC).isoformat(),
        "model": "/".join(settings.llm_model.rsplit("/", 2)[-2:]),
        "api": "Chat Completions v1; native tools then JSON schema for text",
        "temperature": 0.0,
        "max_tokens": settings.llm_max_tokens,
        "evaluation_context": context,
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "system_prompt": prompt("study", now, context["timezone"]),
        "source": "https://github.com/LegionerSV/itmo-tg-template/blob/main/docs/evaluation/lab2-routing.json",
        "tools_source": "MCP tools/list over actual stdio subprocess",
        "results": results,
        "correct_actions": correct,
        "correct_arguments": correct_args,
        "routing_accuracy": correct / 12,
        "argument_accuracy": correct_args / 9,
    }
    await asyncio.to_thread(
        Path(args.output).write_text,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Actions:", correct, "/12. Arguments:", correct_args, "/9.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/lab2-routing-baseline.json")
    parser.add_argument("--retry-errors")
    args = parser.parse_args()
    settings = Settings.load(require_llm=True)
    with temporary_database(settings) as isolated:
        asyncio.run(evaluate(isolated, args))


if __name__ == "__main__":
    main()
