# Model price snapshot

This project blocks unknown model aliases and reserves the maximum estimated cost before every
request. Prices below were checked against provider-owned pages on 2026-08-27. Input estimates use
the uncached price, so cache discounts are never assumed during authorization.

| Alias | Provider API model | Official input / 1M | Official output / 1M | Ledger CNY input / output | Official source |
|---|---|---:|---:|---:|---|
| `deepseek-v4-flash` | `ARK_DEEPSEEK_MODEL_ID` | n/a | n/a | ¥12 / ¥24 conservative ceiling | [Volcengine Ark](https://www.volcengine.com/docs/82379/1544106) |
| `deepseek-v4-pro` | `ARK_DEEPSEEK_MODEL_ID` | n/a | n/a | ¥12 / ¥24 conservative ceiling | [Volcengine Ark](https://www.volcengine.com/docs/82379/1544106) |
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
