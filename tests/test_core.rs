use cachebench::{fingerprint, CacheTracker, Provider, Usage};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn fingerprint_is_stable() {
    let msgs = vec![json!({"role": "user", "content": "hi"})];
    let a = fingerprint(&msgs, &"sys", &json!(null), Some("m"));
    let b = fingerprint(&msgs, &"sys", &json!(null), Some("m"));
    assert_eq!(a, b);
    assert_eq!(a.len(), 16);
}

#[test]
fn fingerprint_excludes_trailing_user_turn() {
    let a_msgs = vec![json!({"role": "user", "content": "A"})];
    let b_msgs = vec![json!({"role": "user", "content": "B"})];
    let a = fingerprint(&a_msgs, &"sys", &json!(null), None);
    let b = fingerprint(&b_msgs, &"sys", &json!(null), None);
    assert_eq!(a, b, "trailing user turn must not affect fingerprint");
}

#[test]
fn fingerprint_changes_with_system_or_model() {
    let msgs: Vec<serde_json::Value> = vec![];
    let a = fingerprint(&msgs, &"sys1", &json!(null), Some("m"));
    let b = fingerprint(&msgs, &"sys2", &json!(null), Some("m"));
    let c = fingerprint(&msgs, &"sys1", &json!(null), Some("m2"));
    assert_ne!(a, b);
    assert_ne!(a, c);
}

#[test]
fn record_and_aggregate() {
    let t = CacheTracker::new(Provider::Anthropic);
    t.record(
        "p1".into(),
        Usage {
            input_tokens: 100,
            cache_read_tokens: 800,
            cache_creation_tokens: 0,
            output_tokens: 50,
        },
        Duration::from_millis(420),
    );
    let agg = t.aggregate();
    assert_eq!(agg.calls, 1);
    assert_eq!(agg.hit_ratio, Some(1.0));
    assert_eq!(agg.tokens_read_from_cache, 800);
    assert!(agg.cost_saved_usd > 0.0);
}

#[test]
fn miss_alert_fires_below_threshold() {
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();
    let t = CacheTracker::new(Provider::Anthropic)
        .with_alert_threshold(0.5)
        .with_alert_hook(move |_| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        });
    t.record(
        "p".into(),
        Usage {
            input_tokens: 10,
            cache_read_tokens: 0,
            cache_creation_tokens: 900,
            output_tokens: 5,
        },
        Duration::from_millis(100),
    );
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn no_alert_when_no_cacheable_prefix() {
    let count = Arc::new(AtomicUsize::new(0));
    let count_clone = count.clone();
    let t = CacheTracker::new(Provider::Anthropic)
        .with_alert_threshold(0.99)
        .with_alert_hook(move |_| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        });
    t.record(
        "p".into(),
        Usage {
            input_tokens: 100,
            cache_read_tokens: 0,
            cache_creation_tokens: 0,
            output_tokens: 50,
        },
        Duration::from_millis(50),
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
}

#[test]
fn by_prefix_groups_correctly() {
    let t = CacheTracker::new(Provider::Anthropic);
    let usage = Usage {
        input_tokens: 10,
        cache_read_tokens: 90,
        cache_creation_tokens: 0,
        output_tokens: 5,
    };
    t.record("A".into(), usage, Duration::from_millis(10));
    t.record("A".into(), usage, Duration::from_millis(10));
    t.record("B".into(), usage, Duration::from_millis(10));
    let by = t.by_prefix();
    assert_eq!(by.len(), 2);
    assert_eq!(by.get("A").unwrap().calls, 2);
    assert_eq!(by.get("B").unwrap().calls, 1);
}

#[test]
fn history_trims_to_size() {
    let t = CacheTracker::new(Provider::Anthropic).with_history_size(3);
    let usage = Usage::default();
    for i in 0..5 {
        t.record(format!("p{i}"), usage, Duration::from_millis(1));
    }
    let calls = t.calls();
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0].prefix_id, "p2");
    assert_eq!(calls[2].prefix_id, "p4");
}

#[test]
fn cost_math_matches_python_sibling() {
    // Same inputs as the Python tests: input=100, cache_read=800, output=50
    // Expected (Anthropic Sonnet 4 default pricing):
    //   cost = 100*3 + 800*0.30 + 0*3.75 + 50*15  / 1e6 = 0.00129
    //   saved = (900-actual_input_paid) reflecting 10x discount on read
    let t = CacheTracker::new(Provider::Anthropic);
    let m = t.record(
        "p".into(),
        Usage {
            input_tokens: 100,
            cache_read_tokens: 800,
            cache_creation_tokens: 0,
            output_tokens: 50,
        },
        Duration::from_millis(0),
    );
    let pricing = Provider::Anthropic.default_pricing();
    let cost = m.cost_usd(&pricing);
    let saved = m.cost_saved_usd(&pricing);
    assert!((cost - 0.00129).abs() < 1e-9, "cost mismatch: {cost}");
    // saved = 900*input - (100*input + 800*cache_read) = 900*3 - (300+240) = 2160 / 1e6
    assert!((saved - 0.00216).abs() < 1e-9, "saved mismatch: {saved}");
}

#[test]
fn clones_share_history() {
    let t = CacheTracker::new(Provider::Anthropic);
    let t2 = t.clone();
    t.record("x".into(), Usage::default(), Duration::from_millis(0));
    assert_eq!(t2.calls().len(), 1);
}
