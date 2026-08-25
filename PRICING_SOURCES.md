# Model price snapshot

This project blocks unknown model aliases and reserves the maximum estimated cost before every
request. Prices below were checked against provider-owned pages on 2026-08-20. Input estimates use
the uncached price, so cache discounts are never assumed during authorization.

| Alias | Provider API model | Input / 1M tokens | Output / 1M tokens | Official source |
|---|---|---:|---:|---|
| `deepseek-v4-flash` | `deepseek-v4-flash` | ¥1 | ¥2 | [DeepSeek](https://api-docs.deepseek.com/zh-cn/quick_start/pricing) |
| `deepseek-v4-pro` | `deepseek-v4-pro` | ¥3 | ¥6 | [DeepSeek](https://api-docs.deepseek.com/zh-cn/quick_start/pricing) |
| `kimi-k2.6` | `kimi-k2.6` | ¥6.50 | ¥27 | [Kimi](https://platform.kimi.com/docs/pricing/chat-k26) |
| `kimi-k3` | `kimi-k3` | ¥20 | ¥100 | [Kimi](https://platform.kimi.com/docs/pricing/chat-k3) |
| `doubao-seed-2.1-pro` | user-configured Ark endpoint/model ID | ¥6 | ¥30 | [Volcengine](https://www.volcengine.com/product/doubao) |

Provider prices can change. A release must recheck this table, the corresponding values in
`src/openlitreview/pricing.py`, and the user's provider-side prepaid balances before it is marked
price-ready.
