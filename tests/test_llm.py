from dataclasses import replace

import pytest
from aiohttp import web

from app.config import Settings
from app.llm import LLMClient, LLMError


@pytest.fixture
async def api(unused_tcp_port):
    requests = []
    response = {
        "choices": [{"message": {"content": "Ответ"}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 12},
        "model": "test-model",
    }
    state = {"status": 200, "response": response}

    async def complete(request):
        requests.append((await request.json(), dict(request.headers)))
        if isinstance(state["response"], str):
            return web.Response(text=state["response"], status=state["status"])
        return web.json_response(state["response"], status=state["status"])

    app = web.Application()
    app.router.add_post("/v1/chat/completions", complete)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    settings = Settings(
        bot_token="test",
        postgres_password="test",
        llm_api_key="test-key",
        llm_base_url=f"http://127.0.0.1:{unused_tcp_port}/v1",
        llm_model="gpt://test-folder/yandexgpt/latest",
    )
    client = LLMClient(settings)
    try:
        yield client, state, requests
    finally:
        await client.close()
        await runner.cleanup()


async def test_client_actual_payload_and_result(api):
    client, _, requests = api
    messages = [{"role": "system", "content": "Инструкция"}, {"role": "user", "content": "Вопрос"}]
    result = await client.complete(messages, 0.0)
    assert result.text == "Ответ"
    assert result.usage == {"total_tokens": 12}
    body, headers = requests[0]
    assert body == {
        "model": client.settings.llm_model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 1000,
        "stream": False,
    }
    assert headers["Authorization"] == "Bearer test-key"
    assert headers["OpenAI-Project"] == "test-folder"


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": "  "}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
        {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
        "not-json",
        "x" * 1_000_001,
    ],
    ids=[
        "missing",
        "no_choices",
        "empty",
        "whitespace",
        "truncated",
        "null",
        "invalid_json",
        "oversized",
    ],
)
async def test_empty_malformed_and_incomplete_response(api, response):
    client, state, _ = api
    state["response"] = response
    with pytest.raises(LLMError):
        await client.complete([], 0.3)


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 503])
async def test_http_error_is_safe_and_not_retried(api, status, caplog):
    client, state, requests = api
    state.update(status=status, response="private-dialogue-and-key")
    with pytest.raises(LLMError, match=f"http_{status}"):
        await client.complete([], 0.3)
    assert len(requests) == 1
    assert "private-dialogue-and-key" not in caplog.text
    assert "test-key" not in caplog.text


async def test_timeout_becomes_safe_llm_error(api, monkeypatch):
    client, _, _ = api
    import aiohttp

    def timeout(*args, **kwargs):
        raise TimeoutError("private-url")

    monkeypatch.setattr(aiohttp.ClientSession, "post", timeout)
    with pytest.raises(LLMError, match="transport_or_format"):
        await client.complete([], 0.3)


async def test_connection_refused_is_safe(api):
    client, _, _ = api
    client.settings = replace(client.settings, llm_base_url="http://127.0.0.1:1/v1")
    with pytest.raises(LLMError, match="transport_or_format"):
        await client.complete([], 0.3)


async def test_native_tool_calls_preserve_arguments_and_do_not_use_response_format(api):
    client, state, requests = api
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "index": 0,
                "type": "function",
                "function": {"name": "get_schedule", "arguments": '{"date":"2026-10-12"}'},
            }
        ],
    }
    state["response"] = {"choices": [{"message": message, "finish_reason": "tool_calls"}]}
    tools = [
        {"type": "function", "function": {"name": "get_schedule", "parameters": {"type": "object"}}}
    ]
    assert await client.agent_turn([], 0.0, tools) == message
    assert requests[0][0]["tools"] == tools
    assert "response_format" not in requests[0][0]
    assert len(requests) == 1


async def test_text_response_uses_separate_json_schema_request(api):
    client, state, requests = api
    state["response"] = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"action":"clarify","text":"Уточните время"}',
                },
                "finish_reason": "stop",
            }
        ]
    }
    tools = [
        {"type": "function", "function": {"name": "get_schedule", "parameters": {"type": "object"}}}
    ]
    result = await client.agent_turn([], 0.3, tools)
    assert "clarify" in result["content"]
    assert len(requests) == 2
    assert requests[0][0]["tools"] == tools
    assert requests[1][0]["response_format"]["json_schema"]["strict"] is True
    assert "tools" not in requests[1][0]
