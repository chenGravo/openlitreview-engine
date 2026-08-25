from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_UP, Decimal


@dataclass(frozen=True)
class ModelPrice:
    alias: str
    provider: str
    api_model: str
    input_cny_per_million: Decimal
    output_cny_per_million: Decimal
    effective_date: str
    source_url: str

    def estimate(self, input_tokens: int, max_output_tokens: int) -> Decimal:
        amount = (
            Decimal(input_tokens) * self.input_cny_per_million
            + Decimal(max_output_tokens) * self.output_cny_per_million
        ) / Decimal(1_000_000)
        return amount.quantize(Decimal("0.0001"), rounding=ROUND_UP)


MODEL_PRICES: dict[str, ModelPrice] = {
    "deepseek-v4-flash": ModelPrice(
        alias="deepseek-v4-flash",
        provider="deepseek",
        api_model="deepseek-v4-flash",
        input_cny_per_million=Decimal("1"),
        output_cny_per_million=Decimal("2"),
        effective_date="2026-08-20",
        source_url="https://api-docs.deepseek.com/zh-cn/quick_start/pricing",
    ),
    "deepseek-v4-pro": ModelPrice(
        alias="deepseek-v4-pro",
        provider="deepseek",
        api_model="deepseek-v4-pro",
        input_cny_per_million=Decimal("3"),
        output_cny_per_million=Decimal("6"),
        effective_date="2026-08-20",
        source_url="https://api-docs.deepseek.com/zh-cn/quick_start/pricing",
    ),
    "kimi-k2.6": ModelPrice(
        alias="kimi-k2.6",
        provider="kimi",
        api_model="kimi-k2.6",
        input_cny_per_million=Decimal("6.5"),
        output_cny_per_million=Decimal("27"),
        effective_date="2026-08-20",
        source_url="https://platform.kimi.com/docs/pricing/chat-k26",
    ),
    "kimi-k3": ModelPrice(
        alias="kimi-k3",
        provider="kimi",
        api_model="kimi-k3",
        input_cny_per_million=Decimal("20"),
        output_cny_per_million=Decimal("100"),
        effective_date="2026-08-20",
        source_url="https://platform.kimi.com/docs/pricing/chat-k3",
    ),
    "doubao-seed-2.1-pro": ModelPrice(
        alias="doubao-seed-2.1-pro",
        provider="doubao",
        api_model="doubao-seed-2.1-pro",
        input_cny_per_million=Decimal("6"),
        output_cny_per_million=Decimal("30"),
        effective_date="2026-08-20",
        source_url="https://www.volcengine.com/product/doubao",
    ),
}


def get_price(model_alias: str) -> ModelPrice:
    try:
        return MODEL_PRICES[model_alias]
    except KeyError as exc:
        raise ValueError(
            f"Unknown or unpriced model {model_alias!r}; request blocked until price is reviewed"
        ) from exc


def estimate_tokens(text: str) -> int:
    """Conservative preflight estimate; provider usage is authoritative after the call."""
    if not text:
        return 0
    utf8_bytes = len(text.encode("utf-8"))
    word_based = int(len(text.split()) * 1.6)
    byte_based = math.ceil(utf8_bytes / 2.5)
    return max(word_based, byte_based, 1)
