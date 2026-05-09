# cachebench

[![crates.io](https://img.shields.io/crates/v/cachebench.svg)](https://crates.io/crates/cachebench)
[![docs.rs](https://docs.rs/cachebench/badge.svg)](https://docs.rs/cachebench)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Prompt-cache observability for LLM APIs. Per-call hit ratio, cost saved, regression alerts. Anthropic, OpenAI, Bedrock.

```toml
[dependencies]
cachebench = "0.1"
```

## Why

Prompt caching saves 50–90% of input tokens on Anthropic and OpenAI, but per-request hit rate is invisible from the SDK. Misses are silent. A deploy that appends a timestamp to a system prompt can quietly halve your cache hit rate and double your bill — and you'll find out from the invoice. Anthropic's SDK [silently misses ~40% on back-to-back requests](https://github.com/anthropics/anthropic-sdk-python/issues/1451) at certain windows.

`cachebench` wraps your LLM call site, takes the `Usage` returned by the provider, and tells you per call what hit and what didn't.

## Quick start

```rust
use cachebench::{CacheTracker, Provider, Usage, fingerprint};
use serde_json::json;
use std::time::Duration;

let tracker = CacheTracker::new(Provider::Anthropic)
    .with_alert_threshold(0.6)
    .with_alert_hook(|m| {
        eprintln!("cache regression: prefix={} hit_ratio={:?}", m.prefix_id, m.hit_ratio());
    });

// After your Anthropic call returns, hand the usage to the tracker:
let messages = vec![json!({"role": "user", "content": "Hello"})];
let prefix = fingerprint(&messages, &"You are helpful", &json!(null), Some("claude-sonnet-4"));

let usage = Usage {
    input_tokens: 50,
    cache_read_tokens: 4123,
    cache_creation_tokens: 0,
    output_tokens: 200,
};

let m = tracker.record(prefix, usage, Duration::from_millis(420));
println!("hit_ratio = {:?}", m.hit_ratio());
println!("cost_saved = ${:.4}", m.cost_saved_usd(&Provider::Anthropic.default_pricing()));

let agg = tracker.aggregate();
println!("{:#?}", agg);
```

## Features

- **Per-call attribution.** Stable `prefix_id` (sha256 of system + tools + model + prefix messages, trailing user turn excluded) lets you group calls by what was supposed to be cached.
- **Regression alerts.** Configurable threshold; fires the alert hook when a cacheable call hits below it.
- **Multi-provider pricing.** `DEFAULT_ANTHROPIC_PRICING`, `DEFAULT_OPENAI_PRICING`, `DEFAULT_BEDROCK_PRICING` constants; pass your own `Pricing` if rates change.
- **Per-prefix grouping.** `tracker.by_prefix()` shows hit rate per prefix — instantly tells you which system prompt regressed.
- **Cheap to share.** `CacheTracker` is `Clone` and shares one inner history across tasks; safe to hand to many spawned tasks.

## What it doesn't do

- Not a proxy. Not a router. Not a cache itself — it observes the provider's cache, doesn't store responses.
- Doesn't make HTTP calls; you do, then hand the `Usage` to `record()`.
- No HTTP middleware in this crate (yet). For automatic capture from `reqwest`-based clients, watch for a future `cachebench-reqwest` companion crate or hook your own middleware.

## Sibling: Python `cachebench`

Python users: same library, same fingerprinting, same metrics — see [MukundaKatta/cachebench](https://github.com/MukundaKatta/cachebench) (Python).

## License

MIT
