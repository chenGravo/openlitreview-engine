import httpx
import pytest

from openlitreview.llm import (
    PROVIDERS,
    ModelResponseError,
    _build_request,
    _extract_content_and_usage,
    _parse_json_content,
    _safe_http_error_detail,
)


def test_chat_completion_request_uses_json_mode() -> None:
    endpoint, body = _build_request(
        PROVIDERS["kimi"],
        model="kimi-k2.6",
        system="Return JSON",
        prompt="Test",
        max_output_tokens=123,
        temperature=0.1,
    )
    assert endpoint == "chat/completions"
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_completion_tokens"] == 123
    assert "max_tokens" not in body
    assert body["temperature"] == 0.6
    assert body["thinking"] == {"type": "disabled"}


def test_kimi_uses_current_official_api_base_url() -> None:
    assert PROVIDERS["kimi"].base_url == "https://api.moonshot.cn/v1"


def test_deepseek_keeps_max_tokens_parameter() -> None:
    _, body = _build_request(
        PROVIDERS["deepseek"],
        model="deepseek-v4-pro",
        system="Return JSON",
        prompt="Test",
        max_output_tokens=123,
        temperature=0.1,
    )
    assert body["max_tokens"] == 123
    assert "max_completion_tokens" not in body


def test_deepseek_uses_ark_route_and_configured_model() -> None:
    provider = PROVIDERS["deepseek"]
    assert provider.base_url == "https://ark.cn-beijing.volces.com/api/v3"
    assert provider.api_key_envs == ("ARK_DEEPSEEK_API_KEY",)
    assert provider.model_envs == ("ARK_DEEPSEEK_MODEL_ID",)
    assert provider.require_configured_model is True
    assert provider.thinking_mode == "disabled"

    _, body = _build_request(
        provider,
        model="deepseek-v4-pro-260425",
        system="Return JSON",
        prompt="Test",
        max_output_tokens=64,
        temperature=0.0,
    )
    assert body["thinking"] == {"type": "disabled"}


def test_http_error_detail_excludes_message_and_contact_data() -> None:
    response = httpx.Response(
        402,
        json={
            "error": {
                "type": "insufficient_balance",
                "code": "billing_error",
                "message": "contact private@example.com and include secret data",
            }
        },
    )
    detail = _safe_http_error_detail(response)
    assert detail == " (type=insufficient_balance, code=billing_error)"
    assert "private@example.com" not in detail


def test_responses_request_does_not_store_provider_state() -> None:
    endpoint, body = _build_request(
        PROVIDERS["doubao"],
        model="ep-verified",
        system="Return JSON",
        prompt="Test",
        max_output_tokens=456,
        temperature=0.1,
    )
    assert endpoint == "responses"
    assert body["model"] == "ep-verified"
    assert body["store"] is False
    assert body["max_output_tokens"] == 456


def test_extract_chat_content_and_usage() -> None:
    content, input_tokens, output_tokens = _extract_content_and_usage(
        PROVIDERS["deepseek"],
        {
            "choices": [
                {"finish_reason": "stop", "message": {"content": '{"ok":true}'}}
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        },
        fallback_input_tokens=100,
        fallback_output_tokens=200,
    )
    assert _parse_json_content(content) == {"ok": True}
    assert (input_tokens, output_tokens) == (12, 7)


def test_extract_responses_content_and_usage() -> None:
    content, input_tokens, output_tokens = _extract_content_and_usage(
        PROVIDERS["doubao"],
        {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"ok":true}'}],
                }
            ],
            "usage": {"input_tokens": 15, "output_tokens": 8},
        },
        fallback_input_tokens=100,
        fallback_output_tokens=200,
    )
    assert _parse_json_content(content) == {"ok": True}
    assert (input_tokens, output_tokens) == (15, 8)


def test_truncated_chat_completion_is_rejected() -> None:
    with pytest.raises(ModelResponseError, match="stopped early"):
        _extract_content_and_usage(
            PROVIDERS["kimi"],
            {
                "choices": [
                    {"finish_reason": "length", "message": {"content": '{"partial":'}}
                ]
            },
            fallback_input_tokens=100,
            fallback_output_tokens=200,
        )
