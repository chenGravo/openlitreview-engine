# Model price snapshot

This project blocks unknown model aliases and reserves the maximum estimated cost before every
request. Prices below were checked against provider-owned pages on 2026-08-27. Input estimates use
the uncached price, so cache discounts are never assumed during authorization.

| Alias | Provider API model | Official input / 1M | Official output / 1M | Ledger CNY input / output | Official source |
|---|---|---:|---:|---:|---|
| `deepseek-v4-flash` | `deepseek-v4-flash` | US$0.14 | US$0.28 | ¥2 / ¥3 | [DeepSeek](https://api-docs.deepseek.com/quick_start/pricing) |
| `deepseek-v4-pro` | `deepseek-v4-pro` | US$0.435 | US$0.87 | ¥4 / ¥7 | [DeepSeek](https://api-docs.deepseek.com/quick_start/pricing) |
| `kimi-k2.6` | `kimi-k2.6` | ¥6.50 | ¥27 | ¥8 / ¥32 | [Kimi](https://platform.kimi.com/docs/pricing/chat-k26) |
| `kimi-k3` | `kimi-k3` | ¥20 | ¥100 | ¥24 / ¥120 | [Kimi](https://platform.kimi.com/docs/pricing/chat-k3) |
| `doubao-seed-2.1-pro` | user-configured Ark endpoint/model ID | ¥6 | ¥30 | ¥6 / ¥30 | [Volcengine](https://www.volcengine.com/product/ark) |

DeepSeek publishes the current prices in US dollars. The ledger converts them at a deliberately
conservative planning rate of ¥8 per US dollar and rounds up to whole CNY amounts. Kimi publishes
prices in CNY; its ledger prices are also rounded upward. These buffers are fail-safe estimates for
the user's CNY cap, not claims about the providers' final invoices.

Provider prices can change. A release must recheck this table, the corresponding values in
`src/openlitreview/pricing.py`, and the user's provider-side prepaid balances before it is marked
price-ready.
