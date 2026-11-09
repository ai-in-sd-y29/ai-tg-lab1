"""HTTP-клиент LLM проверяется на локальном aiohttp-сервере вместо настоящего API."""

import asyncio
import logging

import pytest
from aiohttp import web

from app.config import ConfigError, Settings
from app.llm import (
    ChatMessage,
    LLMEmptyResponseError,
    LLMRequestError,
    LLMTimeoutError,
    LLMUnavailableError,
    OpenAICompatibleClient,
)
from app.logging_setup import SecretFilter
from tests.conftest import TOKEN, make_settings

API_KEY = "sk-test-secret"
MESSAGES = [ChatMessage("system", "Инструкция"), ChatMessage("user", "Привет")]


def ok_body(text="Здравствуйте!", finish_reason="stop"):
    return {
        "model": "test-model-2026",
        "choices": [
            {"message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
    }


@pytest.fixture
async def server(unused_tcp_port):
    """Сервер отвечает по очереди заданными (status, body, delay) и запоминает запросы."""
    state = {"responses": [], "requests": []}

    async def handler(request: web.Request) -> web.Response:
        state["requests"].append(
            {"path": request.path, "headers": dict(request.headers), "json": await request.json()}
        )
        status, body, delay = state["responses"].pop(0)
        if delay:
            await asyncio.sleep(delay)
        if isinstance(body, str):
            return web.Response(status=status, text=body)
        return web.json_response(body, status=status)

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    state["url"] = f"http://127.0.0.1:{unused_tcp_port}/v1"
    yield state
    await runner.cleanup()


@pytest.fixture
async def client_factory(server):
    clients = []

    def factory(**overrides):
        settings = make_settings(llm_base_url=server["url"], llm_api_key=API_KEY, **overrides)
        client = OpenAICompatibleClient(settings)
        client.retry_delay = 0
        clients.append(client)
        return client

    yield factory
    for client in clients:
        await client.close()


async def test_successful_request_sends_roles_parameters_and_key(server, client_factory):
    # Arrange
    server["responses"].append((200, ok_body(), 0))
    client = client_factory(llm_max_output_tokens=321)
    # Act
    result = await client.complete(MESSAGES, temperature=0.3, request_id="r1")
    # Assert
    request = server["requests"][0]
    assert request["headers"]["Authorization"] == f"Bearer {API_KEY}"
    assert request["json"] == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "Инструкция"},
            {"role": "user", "content": "Привет"},
        ],
        "temperature": 0.3,
        "max_tokens": 321,
    }
    assert result.text == "Здравствуйте!"
    assert result.model == "test-model-2026"
    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (12, 5, 17)
    assert result.truncated is False


async def test_length_finish_reason_marks_truncated(server, client_factory):
    # Arrange
    server["responses"].append((200, ok_body(finish_reason="length"), 0))
    # Act
    result = await client_factory().complete(MESSAGES, temperature=0.0, request_id="r")
    # Assert
    assert result.truncated is True


async def test_transient_error_is_retried_once(server, client_factory):
    # Arrange
    server["responses"] += [(503, {"error": {"message": "overloaded"}}, 0), (200, ok_body(), 0)]
    # Act
    result = await client_factory(llm_max_retries=1).complete(
        MESSAGES, temperature=0.7, request_id="r"
    )
    # Assert
    assert result.text == "Здравствуйте!"
    assert len(server["requests"]) == 2


async def test_retries_are_limited(server, client_factory):
    # Arrange
    server["responses"] += [(429, {}, 0), (500, "oops", 0)]
    # Act / Assert
    with pytest.raises(LLMUnavailableError, match="HTTP 500"):
        await client_factory(llm_max_retries=1).complete(MESSAGES, temperature=0.7, request_id="r")
    assert len(server["requests"]) == 2


async def test_client_error_is_not_retried(server, client_factory):
    # Arrange
    error = {"error": {"message": "Incorrect API key provided: sk-or-v1****abcd."}}
    server["responses"].append((401, error, 0))
    # Act / Assert
    with pytest.raises(LLMRequestError) as exc:
        await client_factory(llm_max_retries=3).complete(MESSAGES, temperature=0.7, request_id="r")
    assert exc.value.status == 401
    assert "Incorrect API key" in str(exc.value)
    assert "sk-or-v1" not in str(exc.value) and "abcd" not in str(exc.value)
    assert len(server["requests"]) == 1


@pytest.mark.parametrize(
    "body",
    [
        ok_body(text=""),
        ok_body(text="   \n"),
        {"choices": []},
        {"choices": [{"message": {"content": None}}]},
        "не JSON",
    ],
)
async def test_empty_or_malformed_answer(server, client_factory, body):
    # Arrange
    server["responses"].append((200, body, 0))
    # Act / Assert
    with pytest.raises(LLMEmptyResponseError):
        await client_factory().complete(MESSAGES, temperature=0.7, request_id="r")


async def test_timeout(server, client_factory):
    # Arrange
    server["responses"].append((200, ok_body(), 2))
    # Act / Assert
    with pytest.raises(LLMTimeoutError):
        await client_factory(llm_timeout_seconds=0.2).complete(
            MESSAGES, temperature=0.7, request_id="r"
        )


async def test_unreachable_service(unused_tcp_port_factory):
    # Arrange
    settings = make_settings(llm_base_url=f"http://127.0.0.1:{unused_tcp_port_factory()}/v1")
    client = OpenAICompatibleClient(settings)
    client.retry_delay = 0
    try:
        # Act / Assert
        with pytest.raises(LLMUnavailableError, match="сетевая ошибка"):
            await client.complete(MESSAGES, temperature=0.7, request_id="r")
    finally:
        await client.close()


async def test_logs_have_no_key_prompt_or_user_text(server, client_factory, caplog):
    # Arrange
    server["responses"] += [(500, {}, 0), (200, ok_body(), 0)]
    caplog.set_level(logging.INFO, logger="app.llm")
    # Act
    await client_factory(llm_max_retries=1).complete(MESSAGES, temperature=0.7, request_id="abc")
    # Assert
    text = caplog.text
    assert "request_id=abc" in text and "model=test-model" in text and "duration=" in text
    for forbidden in (API_KEY, "Инструкция", "Привет"):
        assert forbidden not in text


def test_secret_filter_hides_llm_key():
    # Arrange
    secret_filter = SecretFilter([API_KEY])
    record = logging.LogRecord("t", logging.ERROR, "", 1, "Ошибка: ключ %s", (API_KEY,), None)
    # Act
    secret_filter.filter(record)
    # Assert
    assert API_KEY not in record.getMessage()


# Конфигурация LLM.
BASE = {"BOT_TOKEN": TOKEN, "POSTGRES_PASSWORD": "db"}
LLM = {"LLM_BASE_URL": "https://llm.example/v1/", "LLM_API_KEY": "sk-x", "LLM_MODEL": "m"}


def test_llm_settings_are_loaded(tmp_path):
    # Act
    config = Settings.load(tmp_path / ".env", environ={**BASE, **LLM, "HISTORY_MAX_MESSAGES": "10"})
    # Assert
    assert config.llm_base_url == "https://llm.example/v1"
    assert config.llm_model == "m"
    assert config.history_max_messages == 10
    assert "sk-x" not in repr(config)


@pytest.mark.parametrize(
    ("missing", "expected"),
    [("LLM_BASE_URL", "LLM_BASE_URL"), ("LLM_API_KEY", "LLM_API_KEY"), ("LLM_MODEL", "LLM_MODEL")],
)
def test_missing_llm_setting_has_clear_error(tmp_path, missing, expected):
    # Arrange
    environ = {**BASE, **LLM}
    del environ[missing]
    # Act / Assert
    with pytest.raises(ConfigError, match=expected):
        Settings.load(tmp_path / ".env", environ=environ)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("LLM_BASE_URL", "ftp://llm.example"),
        ("LLM_BASE_URL", "https://user:secret-pass@llm.example/v1"),
        ("HISTORY_MAX_CHARS", "100"),
        ("HISTORY_MAX_MESSAGES", "-1"),
        ("LLM_TIMEOUT_SECONDS", "abc"),
        ("LLM_PROXY_URL", "ftp://proxy:1"),
    ],
)
def test_invalid_llm_setting_is_rejected_without_leak(tmp_path, key, value):
    # Act / Assert
    with pytest.raises(ConfigError, match=key) as exc:
        Settings.load(tmp_path / ".env", environ={**BASE, **LLM, key: value})
    assert "secret-pass" not in str(exc.value)
