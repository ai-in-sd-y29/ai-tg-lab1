"""Эксперимент лабораторной № 1: влияние temperature и качества промпта.

Запуск (нужны настройки LLM в .env и запущенный PostgreSQL):
    .venv/bin/python -m scripts.experiment_lab1

Часть 1. Запрос проходит через тот же Assistant, что и в боте, со служебным
chat_id. Перед каждым запуском история сбрасывается, как командой /reset.
Часть 2. Неполная и улучшенная системные инструкции сравниваются по заранее
заданным признакам на запросе, которого нет в few-shot-примерах.

Результаты: docs/lab1/experiment-results.json и docs/lab1/experiment.md.
"""

import argparse
import asyncio
import json
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from app.assistant import Assistant
from app.config import ALLOWED_TEMPERATURES, Settings
from app.db import create_pool
from app.llm import ChatMessage, LLMError, OpenAICompatibleClient
from app.prompts import MODES
from app.storage import PostgresRepository

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "lab1"
# Отрицательный id не совпадает с личными чатами Telegram (они положительные).
EXPERIMENT_CHAT_ID = -424242
RUNS_PER_TEMPERATURE = 3

EXPERIMENT_QUERY = "Чем список отличается от кортежа в Python и когда что лучше использовать?"
EXPERIMENT_MODE = "study"

COMPARISON_QUERY = "Почему в Python выражение 0.1 + 0.2 == 0.3 возвращает False?"
COMPARISON_TEMPERATURE = 0.3
WEAK_PROMPT = "Ты помощник. Ответь на вопрос."


@dataclass
class FormatCheck:
    first_line_is_short_answer: bool
    steps_2_to_5: bool
    has_code_block: bool
    ends_with_self_check: bool
    words_without_code_le_250: bool

    @property
    def passed(self) -> bool:
        return all(asdict(self).values())

    @property
    def score(self) -> int:
        return sum(asdict(self).values())


def check_study_format(text: str) -> FormatCheck:
    """Признаки формата из STUDY_PROMPT, проверяемые автоматически."""
    without_code = re.sub(r"```.*?```", "", text, flags=re.S)
    lines = [line.strip() for line in without_code.splitlines() if line.strip()]
    first = lines[0] if lines else ""
    bullets = [line for line in lines if line.startswith(("- ", "• ", "* "))]
    return FormatCheck(
        first_line_is_short_answer=bool(first)
        and not first.startswith(("-", "#", "1.", "*"))
        and len(first) <= 300,
        steps_2_to_5=2 <= len(bullets) <= 5,
        has_code_block=text.count("```") >= 2,
        ends_with_self_check=bool(lines) and lines[-1].startswith("Проверь себя:"),
        words_without_code_le_250=len(without_code.split()) <= 250,
    )


async def one_run(assistant: Assistant, number: int, temperature: float) -> dict:
    await assistant.repository.set_temperature(EXPERIMENT_CHAT_ID, temperature)
    await assistant.reset(EXPERIMENT_CHAT_ID)  # аналог /reset
    reply = await assistant.ask(
        EXPERIMENT_CHAT_ID, EXPERIMENT_QUERY, request_id=f"exp-{uuid.uuid4().hex[:8]}"
    )
    result = reply.result
    check = check_study_format(reply.text) if result else None
    print(f"Запуск {number:2d}: T={temperature} длина={len(reply.text)}", flush=True)
    return {
        "run": number,
        "temperature": temperature,
        "ok": result is not None,
        "length_chars": len(reply.text),
        "format_ok": check.passed if check else False,
        "format_score": check.score if check else 0,
        "format_details": asdict(check) if check else None,
        "finish_reason": result.finish_reason if result else None,
        "prompt_tokens": result.prompt_tokens if result else None,
        "completion_tokens": result.completion_tokens if result else None,
        "total_tokens": result.total_tokens if result else None,
        "model": result.model if result else None,
        "text": reply.text,
        "failed_attempts": [],
    }


async def temperature_runs(assistant: Assistant, repository: PostgresRepository) -> list[dict]:
    await repository.set_mode(EXPERIMENT_CHAT_ID, EXPERIMENT_MODE)
    rows = []
    for temperature in ALLOWED_TEMPERATURES:
        for _ in range(RUNS_PER_TEMPERATURE):
            rows.append(await one_run(assistant, len(rows) + 1, temperature))
    await assistant.reset(EXPERIMENT_CHAT_ID)
    return rows


async def retry_failed_runs(assistant: Assistant, repository: PostgresRepository, runs: list[dict]):
    """Повторяет только запуски, где сервис не вернул ответ (таймаут, 429, 5xx).

    Ответы модели не переигрываются: отбирать удачные результаты нельзя. Неудачная
    попытка остаётся в failed_attempts и выводится в отчёте.
    """
    await repository.set_mode(EXPERIMENT_CHAT_ID, EXPERIMENT_MODE)
    for index, row in enumerate(runs):
        if row["ok"]:
            continue
        new_row = await one_run(assistant, row["run"], row["temperature"])
        new_row["failed_attempts"] = [*row.get("failed_attempts", []), row["text"]]
        runs[index] = new_row
    await assistant.reset(EXPERIMENT_CHAT_ID)


async def prompt_comparison(client: OpenAICompatibleClient) -> list[dict]:
    rows = []
    variants = [("неполная", WEAK_PROMPT), ("улучшенная (R.C.T.F.)", MODES["study"].system_prompt)]
    for name, prompt in variants:
        for run in range(1, RUNS_PER_TEMPERATURE + 1):
            messages = [ChatMessage("system", prompt), ChatMessage("user", COMPARISON_QUERY)]
            try:
                result = await client.complete(
                    messages,
                    temperature=COMPARISON_TEMPERATURE,
                    request_id=f"cmp-{uuid.uuid4().hex[:8]}",
                )
                text, usage = result.text, result.completion_tokens
            except LLMError as error:
                text, usage = f"[ошибка: {type(error).__name__}]", None
            check = check_study_format(text)
            rows.append(
                {
                    "prompt": name,
                    "run": run,
                    "length_chars": len(text),
                    "format_score": check.score,
                    "format_details": asdict(check),
                    "completion_tokens": usage,
                    "text": text,
                }
            )
            print(f"Сравнение: {name} #{run} баллы формата={check.score}/5", flush=True)
    return rows


def yes_no(value: bool) -> str:
    return "да" if value else "нет"


def render_markdown(meta: dict, runs: list[dict], comparison: list[dict]) -> str:
    out = [
        "# Эксперимент лабораторной № 1 (сырые результаты)",
        "",
        "Файл создан скриптом `scripts/experiment_lab1.py`. Анализ — в `docs/lab1/report.md`.",
        "",
        f"- Дата: {meta['date']}",
        f"- Модель (запрошенная): `{meta['model']}`; модель в ответе API: "
        f"`{meta['response_model']}`",
        f"- API: OpenAI-совместимый Chat Completions, `{meta['base_url_host']}`",
        f"- Режим: `/{EXPERIMENT_MODE}`, системный промпт — `STUDY_PROMPT` из `app/prompts.py`",
        f"- Постоянные настройки: max_tokens={meta['max_tokens']}, история сбрасывается "
        "перед каждым запуском, прочие параметры случайности не передаются (значения провайдера "
        "по умолчанию)",
        f"- Запрос: «{EXPERIMENT_QUERY}»",
        "",
        "Формат считается соблюдённым, если выполнены все 5 признаков: краткий ответ в первой "
        "строке; 2–5 пунктов «- »; блок кода; последняя строка «Проверь себя:»; не больше 250 "
        "слов без кода.",
        "",
        "## Таблица 12 запусков",
        "",
        "| № | T | Длина, симв. | Формат (баллы из 5) | prompt / completion / total токенов |"
        " finish_reason |",
        "|---:|---:|---:|---|---|---|",
    ]
    for r in runs:
        tokens = f"{r['prompt_tokens']} / {r['completion_tokens']} / {r['total_tokens']}"
        out.append(
            f"| {r['run']} | {r['temperature']:.1f} | {r['length_chars']} | "
            f"{yes_no(r['format_ok'])} ({r['format_score']}) | {tokens} | {r['finish_reason']} |"
        )
    retried = [r for r in runs if r.get("failed_attempts")]
    if retried:
        out += ["", "Повторённые запуски (сервис не вернул ответ, ответ модели не переигрывался):"]
        for r in retried:
            errors = "; ".join(r["failed_attempts"])
            out.append(f"- запуск {r['run']} (T={r['temperature']:.1f}): {errors}")
    out += ["", "## Сравнение инструкций", ""]
    out += [
        f"Запрос: «{COMPARISON_QUERY}», temperature={COMPARISON_TEMPERATURE}.",
        f"Неполная инструкция: «{WEAK_PROMPT}». Улучшенная — `STUDY_PROMPT`.",
        "",
        "| Инструкция | № | Длина | Баллы формата | Токены ответа |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in comparison:
        out.append(
            f"| {r['prompt']} | {r['run']} | {r['length_chars']} | {r['format_score']}/5 | "
            f"{r['completion_tokens']} |"
        )
    out += ["", "## Полные ответы", ""]
    for r in runs:
        out += [f"### Запуск {r['run']}, T={r['temperature']:.1f}", "", "````text", r["text"]]
        out += ["````", ""]
    for r in comparison:
        out += [f"### Инструкция «{r['prompt']}», запуск {r['run']}", "", "````text", r["text"]]
        out += ["````", ""]
    return "\n".join(out)


async def main_async(env_file: str, retry_failed: bool) -> None:
    settings = Settings.load(env_file)
    pool = await create_pool(settings)
    client = OpenAICompatibleClient(settings)
    results_path = OUT_DIR / "experiment-results.json"
    try:
        repository = PostgresRepository(pool)
        await repository.ensure_schema()
        assistant = Assistant(repository, client, settings)
        if retry_failed:
            saved = json.loads(results_path.read_text(encoding="utf-8"))
            runs, comparison = saved["runs"], saved["comparison"]
            await retry_failed_runs(assistant, repository, runs)
        else:
            runs = await temperature_runs(assistant, repository)
            comparison = await prompt_comparison(client)
    finally:
        await client.close()
        await pool.close()

    if retry_failed:
        meta = saved["meta"]
        meta["retried_at"] = datetime.now().astimezone().isoformat(timespec="minutes")
        _write(meta, runs, comparison)
        return
    meta = {
        "date": datetime.now().astimezone().isoformat(timespec="minutes"),
        "model": settings.llm_model,
        "response_model": next((r["model"] for r in runs if r["model"]), None),
        "base_url_host": urlsplit(settings.llm_base_url).hostname,
        "max_tokens": settings.llm_max_output_tokens,
        "query": EXPERIMENT_QUERY,
        "mode": EXPERIMENT_MODE,
        "comparison_query": COMPARISON_QUERY,
        "comparison_temperature": COMPARISON_TEMPERATURE,
        "weak_prompt": WEAK_PROMPT,
    }
    _write(meta, runs, comparison)


def _write(meta: dict, runs: list[dict], comparison: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "experiment-results.json").write_text(
        json.dumps(
            {"meta": meta, "runs": runs, "comparison": comparison}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "experiment.md").write_text(
        render_markdown(meta, runs, comparison), encoding="utf-8"
    )
    print(f"Готово: {OUT_DIR / 'experiment.md'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Эксперимент лабораторной № 1")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="повторить только запуски, завершившиеся ошибкой сервиса",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args.env_file, args.retry_failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
