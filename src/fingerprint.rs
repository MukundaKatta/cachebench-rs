use serde::Serialize;
use sha2::{Digest, Sha256};

/// Stable hash of the cacheable prefix portion of a call.
///
/// Excludes the trailing user turn (which fragments the prefix space).
/// Same `(system, tools, model, prefix_messages)` ⇒ same id; perfect for
/// grouping calls in [`CacheTracker::by_prefix`](crate::CacheTracker::by_prefix).
///
/// `messages`, `system`, `tools` accept any `serde::Serialize` value. Pass
/// `&serde_json::Value::Null` for `system`/`tools` if absent.
pub fn fingerprint<M, S, T>(messages: &[M], system: &S, tools: &T, model: Option<&str>) -> String
where
    M: Serialize,
    S: Serialize,
    T: Serialize,
{
    // Drop the trailing user turn; cacheable prefix is everything before it.
    let prefix = if messages.is_empty() {
        &messages[..]
    } else {
        &messages[..messages.len() - 1]
    };

    #[derive(Serialize)]
    struct Payload<'a, M, S, T>
    where
        M: Serialize,
        S: Serialize,
        T: Serialize,
    {
        system: &'a S,
        tools: &'a T,
        model: Option<&'a str>,
        prefix_messages: &'a [M],
    }

    let payload = Payload {
        system,
        tools,
        model,
        prefix_messages: prefix,
    };
    let blob = serde_json::to_vec(&payload).unwrap_or_default();
    let mut hasher = Sha256::new();
    hasher.update(&blob);
    let digest = hasher.finalize();
    // First 8 bytes hex = 16 chars, matches the Python sibling lib.
    format!(
        "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7]
    )
}
