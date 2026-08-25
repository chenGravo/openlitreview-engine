# Model price snapshot

This project blocks unknown model aliases and reserves the maximum estimated cost before every
request. Prices below were checked against provider-owned pages on 2026-08-25. Input estimates use
the uncached price, so cache discounts are never assumed during authorization.

| Alias | Provider API model | Official input / 1M | Official output / 1M | Ledger CNY input / output | Official source |
|---|---|---:|---:|---:|---|
| `deepseek-v4-flash` | `deepseek-v4-flash` | ¥1 | ¥2 | ¥1 / ¥2 | [DeepSeek](https://api-docs.deepseek.com/zh-cn/quick_start/pricing) |
| `deepseek-v4-pro` | `deepseek-v4-pro` | ¥3 | ¥6 | ¥3 / ¥6 | [DeepSeek](https://api-docs.deepseek.com/zh-cn/quick_start/pricing) |
| `kimi-k2.6` | `kimi-k2.6` | US$0.95 | US$4 | ¥8 / ¥32 | [Kimi](https://platform.kimi.ai/docs/pricing/chat-k26) |
| `kimi-k3` | `kimi-k3` | US$3 | US$15 | ¥24 / ¥120 | [Kimi](https://platform.kimi.ai/docs/pricing/chat-k3) |
| `doubao-seed-2.1-pro` | user-configured Ark endpoint/model ID | ¥6 | ¥30 | ¥6 / ¥30 | [Volcengine](https://www.volcengine.com/product/ark) |

Kimi is billed in US dollars and its listed prices exclude applicable taxes. The local ledger uses
a fixed conservative planning rate of ¥8 per US dollar, so authorization rounds the current K2.6
rates up to ¥8/¥32 and K3 rates to ¥24/¥120 per million tokens. This planning conversion is not a
claim about the provider's settlement exchange rate; it is a fail-safe buffer for the user's CNY
monthly cap.

Provider prices can change. A release must recheck this table, the corresponding values in
`src/openlitreview/pricing.py`, and the user's provider-side prepaid balances before it is marked
price-ready.
