# Правила разработки

Проект — асинхронный Telegram-ассистент, Python 3.12. Лабораторная 2 продолжает функции lab1.

- `app/handlers` принимает Telegram updates и отправляет ответы; HTTP и SQL здесь запрещены.
- `app/agent.py` управляет выбором действий, ограничениями и подтверждением.
- `app/mcp_server.py` предоставляет инструменты и ресурс через stdio; stdout только MCP.
- `app/mcp_client.py` обнаруживает схемы сервера; обработчики не дублируют их.
- `app/lab2_repository.py` хранит настройки, действия, напоминания и аудит PostgreSQL.
- `data` содержит исключительно вымышленное расписание; `docs/evaluation` — исходный набор.
- Миграции идемпотентны, не удаляют данные lab1; SQL параметризован.
- Настройки читаются централизованно; новые имена документируются в `.env.example`.
- Не публиковать `.env`, `.env.cloud`, `.deploy`, токены, закрытые ключи, персональные данные.
- Не выводить HTTP-тела, Telegram ID, историю и системные инструкции в текстовые логи.
- Модель не назначает владельца, ключ повторов или подтверждение действия.
- До изменения прочитать `spec.md` и `plan.md`, обновлять их вместе с реализацией.

Команды из корня, с интерпретатором `.venv/Scripts/python.exe` (Windows) или `.venv/bin/python`:

```
python -m app
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m pip check
```

Интеграционные тесты: `RUN_INTEGRATION=1 python -m pytest -m integration -v`.
Подготовка локальной среды: `scripts/start.ps1 --setup-only` или `bash scripts/start.sh --setup-only`.
