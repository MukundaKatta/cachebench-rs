use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

/// LLM provider whose cache mechanics we're tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Provider {
    /// Anthropic / Claude direct API.
    Anthropic,
    /// OpenAI direct API.
    OpenAI,
    /// AWS Bedrock (Claude or Llama family).
    Bedrock,
}

/// Per-million-token USD prices for one provider's tier.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Pricing {
    /// Plain (uncached) input tokens.
    pub input_per_mtok: f64,
    /// Tokens read from cache.
    pub cache_read_per_mtok: f64,
    /// Tokens written into cache (5-minute tier).
    pub cache_write_5m_per_mtok: f64,
    /// Tokens written into cache (1-hour tier).
    pub cache_write_1h_per_mtok: f64,
    /// Output tokens.
    pub output_per_mtok: f64,
}

/// Anthropic Claude Sonnet 4 default pricing as of late 2025.
pub const DEFAULT_ANTHROPIC_PRICING: Pricing = Pricing {
    input_per_mtok: 3.00,
    cache_read_per_mtok: 0.30,
    cache_write_5m_per_mtok: 3.75,
    cache_write_1h_per_mtok: 6.00,
    output_per_mtok: 15.00,
};

/// OpenAI GPT-4o default pricing as of late 2025.
pub const DEFAULT_OPENAI_PRICING: Pricing = Pricing {
    input_per_mtok: 2.50,
    cache_read_per_mtok: 1.25,
    cache_write_5m_per_mtok: 2.50,
    cache_write_1h_per_mtok: 2.50,
    output_per_mtok: 10.00,
};

/// Bedrock Claude default pricing as of late 2025.
pub const DEFAULT_BEDROCK_PRICING: Pricing = Pricing {
    input_per_mtok: 3.00,
    cache_read_per_mtok: 0.30,
    cache_write_5m_per_mtok: 3.75,
    cache_write_1h_per_mtok: 6.00,
    output_per_mtok: 15.00,
};

impl Provider {
    /// Returns the default per-million-token pricing for this provider.
    pub fn default_pricing(&self) -> Pricing {
        match self {
            Provider::Anthropic => DEFAULT_ANTHROPIC_PRICING,
            Provider::OpenAI => DEFAULT_OPENAI_PRICING,
            Provider::Bedrock => DEFAULT_BEDROCK_PRICING,
        }
    }
}

/// Token usage breakdown extracted from a provider response.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Plain uncached input tokens.
    pub input_tokens: u64,
    /// Tokens served from cache (a "hit").
    pub cache_read_tokens: u64,
    /// Tokens written to cache on this call (a "miss" that populated cache).
    pub cache_creation_tokens: u64,
    /// Output tokens generated.
    pub output_tokens: u64,
}

/// One recorded LLM call with cache metrics attached.
#[derive(Debug, Clone)]
pub struct CallMetrics {
    /// Which provider this call hit.
    pub provider: Provider,
    /// Stable hash of the cacheable prefix; same prefix = same id.
    pub prefix_id: String,
    /// Token usage breakdown.
    pub usage: Usage,
    /// Wall-clock latency of the call.
    pub elapsed: Duration,
    /// Recorded at this moment.
    pub timestamp: SystemTime,
}

impl CallMetrics {
    /// Cache hit ratio over the cacheable portion of this call.
    /// Returns `None` if no cacheable prefix was sent.
    pub fn hit_ratio(&self) -> Option<f64> {
        let cacheable = self.usage.cache_read_tokens + self.usage.cache_creation_tokens;
        if cacheable == 0 {
            None
        } else {
            Some(self.usage.cache_read_tokens as f64 / cacheable as f64)
        }
    }

    /// USD cost of this call given a pricing config.
    pub fn cost_usd(&self, pricing: &Pricing) -> f64 {
        (self.usage.input_tokens as f64 * pricing.input_per_mtok
            + self.usage.cache_read_tokens as f64 * pricing.cache_read_per_mtok
            + self.usage.cache_creation_tokens as f64 * pricing.cache_write_5m_per_mtok
            + self.usage.output_tokens as f64 * pricing.output_per_mtok)
            / 1_000_000.0
    }

    /// USD saved versus paying full input price for everything.
    pub fn cost_saved_usd(&self, pricing: &Pricing) -> f64 {
        let cacheable_total = self.usage.input_tokens
            + self.usage.cache_read_tokens
            + self.usage.cache_creation_tokens;
        let full = cacheable_total as f64 * pricing.input_per_mtok;
        let actual = self.usage.input_tokens as f64 * pricing.input_per_mtok
            + self.usage.cache_read_tokens as f64 * pricing.cache_read_per_mtok
            + self.usage.cache_creation_tokens as f64 * pricing.cache_write_5m_per_mtok;
        (full - actual) / 1_000_000.0
    }
}

/// Aggregate statistics across many calls.
#[derive(Debug, Clone, Copy, Default)]
pub struct Aggregate {
    /// Total calls recorded in window.
    pub calls: usize,
    /// Hit ratio across all cacheable tokens. `None` if no cacheable traffic.
    pub hit_ratio: Option<f64>,
    /// Total tokens read from cache.
    pub tokens_read_from_cache: u64,
    /// Total tokens written to cache.
    pub tokens_written_to_cache: u64,
    /// Total USD cost.
    pub cost_usd: f64,
    /// Total USD saved versus full-input pricing.
    pub cost_saved_usd: f64,
}

/// Per-prefix stats group; lets you spot which system prompt regressed.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrefixStats {
    /// Calls with this prefix id.
    pub calls: usize,
    /// Hit ratio for this prefix only.
    pub hit_ratio: Option<f64>,
    /// USD saved across calls with this prefix.
    pub cost_saved_usd: f64,
}

type AlertHook = Arc<dyn Fn(&CallMetrics) + Send + Sync>;

/// Records per-call cache metrics and exposes aggregate / per-prefix views.
///
/// `CacheTracker` is `Clone` and cheap to share across tasks; clones share
/// the same internal history but carry their own config (which is rarely
/// changed post-construction anyway).
#[derive(Clone)]
pub struct CacheTracker {
    provider: Provider,
    pricing: Pricing,
    miss_alert_threshold: f64,
    on_miss_alert: Option<AlertHook>,
    history_size: usize,
    history: Arc<Mutex<VecDeque<CallMetrics>>>,
}

impl CacheTracker {
    /// Construct a tracker for the given provider, with default pricing.
    pub fn new(provider: Provider) -> Self {
        Self {
            provider,
            pricing: provider.default_pricing(),
            miss_alert_threshold: 0.6,
            on_miss_alert: None,
            history_size: 10_000,
            history: Arc::new(Mutex::new(VecDeque::with_capacity(1024))),
        }
    }

    /// Override pricing.
    pub fn with_pricing(mut self, pricing: Pricing) -> Self {
        self.pricing = pricing;
        self
    }

    /// Set the hit-ratio threshold below which `on_miss_alert` fires.
    pub fn with_alert_threshold(mut self, threshold: f64) -> Self {
        self.miss_alert_threshold = threshold;
        self
    }

    /// Register a callback fired when a cacheable call hits below threshold.
    pub fn with_alert_hook<F>(mut self, hook: F) -> Self
    where
        F: Fn(&CallMetrics) + Send + Sync + 'static,
    {
        self.on_miss_alert = Some(Arc::new(hook));
        self
    }

    /// Set how many recent calls to retain in memory.
    pub fn with_history_size(mut self, size: usize) -> Self {
        self.history_size = size;
        self
    }

    /// Record a call's usage and return the resulting metrics.
    pub fn record(&self, prefix_id: String, usage: Usage, elapsed: Duration) -> CallMetrics {
        let m = CallMetrics {
            provider: self.provider,
            prefix_id,
            usage,
            elapsed,
            timestamp: SystemTime::now(),
        };
        {
            let mut history = self.history.lock();
            history.push_back(m.clone());
            while history.len() > self.history_size {
                history.pop_front();
            }
        }

        let cacheable = usage.cache_read_tokens + usage.cache_creation_tokens;
        if cacheable > 0 {
            if let Some(ratio) = m.hit_ratio() {
                if ratio < self.miss_alert_threshold {
                    if let Some(hook) = self.on_miss_alert.as_ref() {
                        hook(&m);
                    }
                }
            }
        }
        m
    }

    /// All recorded calls, oldest first.
    pub fn calls(&self) -> Vec<CallMetrics> {
        self.history.lock().iter().cloned().collect()
    }

    /// Drop all recorded history.
    pub fn reset(&self) {
        self.history.lock().clear();
    }

    /// The provider this tracker is configured for.
    pub fn provider(&self) -> Provider {
        self.provider
    }

    /// Aggregate across all recorded calls.
    pub fn aggregate(&self) -> Aggregate {
        let calls = self.calls();
        if calls.is_empty() {
            return Aggregate::default();
        }
        let read: u64 = calls.iter().map(|c| c.usage.cache_read_tokens).sum();
        let write: u64 = calls.iter().map(|c| c.usage.cache_creation_tokens).sum();
        let cacheable = read + write;
        let hit_ratio = if cacheable == 0 {
            None
        } else {
            Some(read as f64 / cacheable as f64)
        };
        Aggregate {
            calls: calls.len(),
            hit_ratio,
            tokens_read_from_cache: read,
            tokens_written_to_cache: write,
            cost_usd: calls.iter().map(|c| c.cost_usd(&self.pricing)).sum(),
            cost_saved_usd: calls.iter().map(|c| c.cost_saved_usd(&self.pricing)).sum(),
        }
    }

    /// Group recorded calls by `prefix_id`.
    pub fn by_prefix(&self) -> HashMap<String, PrefixStats> {
        let mut groups: HashMap<String, Vec<CallMetrics>> = HashMap::new();
        for c in self.calls() {
            groups.entry(c.prefix_id.clone()).or_default().push(c);
        }
        groups
            .into_iter()
            .map(|(id, ms)| {
                let read: u64 = ms.iter().map(|c| c.usage.cache_read_tokens).sum();
                let write: u64 = ms.iter().map(|c| c.usage.cache_creation_tokens).sum();
                let cacheable = read + write;
                let hit_ratio = if cacheable == 0 {
                    None
                } else {
                    Some(read as f64 / cacheable as f64)
                };
                (
                    id,
                    PrefixStats {
                        calls: ms.len(),
                        hit_ratio,
                        cost_saved_usd: ms.iter().map(|c| c.cost_saved_usd(&self.pricing)).sum(),
                    },
                )
            })
            .collect()
    }
}
