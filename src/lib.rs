//! Prompt-cache observability for LLM APIs.
//!
//! Wrap your LLM call site with a [`CacheTracker`], feed it the [`Usage`]
//! returned by the provider, and get per-call cache hit ratio, cost saved,
//! and regression alerts. Cross-provider (Anthropic, OpenAI, Bedrock).
//!
//! # Quick start
//!
//! ```
//! use cachebench::{CacheTracker, Provider, Usage};
//! use std::time::Duration;
//!
//! let tracker = CacheTracker::new(Provider::Anthropic)
//!     .with_alert_threshold(0.6);
//!
//! // After your LLM call:
//! let usage = Usage {
//!     input_tokens: 100,
//!     cache_read_tokens: 800,
//!     cache_creation_tokens: 0,
//!     output_tokens: 50,
//! };
//! let metrics = tracker.record("prefix-abc".into(), usage, Duration::from_millis(420));
//! assert_eq!(metrics.hit_ratio(), Some(1.0));
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

mod core;
mod fingerprint;

pub use crate::core::{
    Aggregate, CacheTracker, CallMetrics, PrefixStats, Pricing, Provider, Usage,
    DEFAULT_ANTHROPIC_PRICING, DEFAULT_BEDROCK_PRICING, DEFAULT_OPENAI_PRICING,
};
pub use crate::fingerprint::fingerprint;
