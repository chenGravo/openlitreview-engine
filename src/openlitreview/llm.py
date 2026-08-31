from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from .budget import BudgetLedger
from .pricing import estimate_tokens, get_price


@dataclass(frozen=True)
class ProviderConfig:
    base_url: str
    api_style: Literal["chat_completions", "responses"]
    api_key_envs: tuple[str, ...]
    model_envs: tuple[str, ...] = ()
    require_configured_model: bool = False
    max_tokens_field: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    temperature_override: float | None = None
    thinking_mode: Literal["enabled", "disabled"] | None = None


PROVIDERS: dict[str, ProviderConfig] = {
    "deepseek": ProviderConfig(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_style="chat_completions",
        api_key_envs=("ARK_DEEPSEEK_API_KEY",),
        model_envs=("ARK_DEEPSEEK_MODEL_ID",),
        require_configured_model=True,
    ),
    "kimi": ProviderConfig(
        base_url="https://api.moonshot.cn/v1",
        api_style="chat_completions",
        api_key_envs=("KIMI_API_KEY", "MOONSHOT_API_KEY"),
        max_tokens_field="max_completion_tokens",
        temperature_override=0.6,
        thinking_mode="disabled",
    ),
    "doubao": ProviderConfig(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_style="responses",
        api_key_envs=("DOUBAO_API_KEY", "ARK_API_KEY"),
        model_envs=("DOUBAO_MODEL_ENDPOINT", "ARK_MODEL_ID"),
        require_configured_model=True,
    ),
}


class ModelResponseError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, ledger: BudgetLedger, task_id: str) -> None:
        self.ledger = ledger
        self.task_id = task_id

    async def complete_json(
        self,
        *,
        model_alias: str,
        system: str,
        prompt: str,
        max_output_tokens: int,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        price = get_price(model_alias)
        provider = PROVIDERS[price.provider]
        api_key = _first_environment_value(provider.api_key_envs)
        if not api_key:
            raise ModelResponseError(
                f"Missing secret: {' or '.join(provider.api_key_envs)}"
            )
        configured_model = _first_environment_value(provider.model_envs)
        if provider.require_configured_model and not configured_model:
            raise ModelResponseError(
                f"{price.provider} requires a configured model ID in "
                f"{' or '.join(provider.model_envs)}"
            )
        api_model = configured_model or price.api_model
        base_url = os.getenv(
            f"OPENLITREVIEW_{price.provider.upper()}_BASE_URL", provider.base_url
        ).rstrip("/")
        input_tokens = estimate_tokens(system + "\n" + prompt)
        call_id, _ = self.ledger.authorize_call(
            self.task_id,
            model_alias,
            input_tokens=input_tokens,
            max_output_tokens=max_output_tokens,
        )
        endpoint, body = _build_request(
            provider,
            model=api_model,
            system=system,
            prompt=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(180.0, connect=20.0), follow_redirects=True
            ) as client:
                response = await client.post(
                    f"{base_url}/{endpoint}",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=body,
                )
                response.raise_for_status()
                payload = response.json()
            content, actual_input, actual_output = _extract_content_and_usage(
                provider,
                payload,
                fallback_input_tokens=input_tokens,
                fallback_output_tokens=max_output_tokens,
            )
            parsed = _parse_json_content(content)
            self.ledger.reconcile_call(call_id, actual_input, actual_output)
            return parsed
        except httpx.HTTPStatusError as exc:
            self.ledger.reconcile_call(
                call_id,
                input_tokens,
                max_output_tokens,
                status="failed_unknown",
            )
            detail = _safe_http_error_detail(exc.response)
            raise ModelResponseError(
                f"Model request failed: HTTP {exc.response.status_code}{detail}"
            ) from exc
        except ModelResponseError:
            self.ledger.reconcile_call(
                call_id,
                input_tokens,
                max_output_tokens,
                status="failed_unknown",
            )
            raise
        except Exception as exc:
            self.ledger.reconcile_call(
                call_id,
                input_tokens,
                max_output_tokens,
                status="failed_unknown",
            )
            raise ModelResponseError(f"Model request failed: {type(exc).__name__}") from exc


def _first_environment_value(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def _build_request(
    provider: ProviderConfig,
    *,
    model: str,
    system: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    if provider.api_style == "chat_completions":
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": (
                provider.temperature_override
                if provider.temperature_override is not None
                else temperature
            ),
            "response_format": {"type": "json_object"},
        }
        body[provider.max_tokens_field] = max_output_tokens
        if provider.thinking_mode is not None:
            body["thinking"] = {"type": provider.thinking_mode}
        return "chat/completions", body
    return "responses", {
        "model": model,
        "input": [
            {"type": "message", "role": "system", "content": system},
            {"type": "message", "role": "user", "content": prompt},
        ],
        "max_output_tokens": max_output_tokens,
        "store": False,
    }


def _extract_content_and_usage(
    provider: ProviderConfig,
    payload: dict[str, Any],
    *,
    fallback_input_tokens: int,
    fallback_output_tokens: int,
) -> tuple[Any, int, int]:
    usage = payload.get("usage") or {}
    if provider.api_style == "chat_completions":
        choices = payload.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise ModelResponseError("Chat completion contains no choice")
        finish_reason = choices[0].get("finish_reason")
        if finish_reason in {"length", "content_filter"}:
            raise ModelResponseError(f"Chat completion stopped early: {finish_reason}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        actual_input = int(usage.get("prompt_tokens") or fallback_input_tokens)
        actual_output = int(usage.get("completion_tokens") or fallback_output_tokens)
        return content, actual_input, actual_output

    if payload.get("status") not in {None, "completed"}:
        raise ModelResponseError(f"Responses request is not complete: {payload.get('status')}")
    texts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "output_text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text)
    if not texts:
        raise ModelResponseError("Responses request contains no output text")
    actual_input = int(usage.get("input_tokens") or fallback_input_tokens)
    actual_output = int(usage.get("output_tokens") or fallback_output_tokens)
    return "\n".join(texts), actual_input, actual_output


def _parse_json_content(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise ModelResponseError("Model response content is not text or an object")
    stripped = content.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ModelResponseError("Model response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ModelResponseError("Model JSON response must be an object")
    return payload


def _safe_http_error_detail(response: httpx.Response) -> str:
    """Return provider error type/code only; never persist messages or request data."""
    try:
        payload = response.json()
    except (ValueError, TypeError):
        return ""
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return ""
    safe_parts: list[str] = []
    for label, key in (("type", "type"), ("code", "code")):
        value = error.get(key)
        if isinstance(value, (str, int)):
            cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "", str(value))[:80]
            if cleaned:
                safe_parts.append(f"{label}={cleaned}")
    return f" ({', '.join(safe_parts)})" if safe_parts else ""
