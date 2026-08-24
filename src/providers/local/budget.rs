//! Output-token budgeting for a fixed local context window.
//!
//! Claude Code asks for a big `max_tokens` (64K by default) regardless of how
//! long the prompt is. Anthropic's own API and llama.cpp silently clamp that to
//! whatever is left in the window; vLLM does not — it rejects the whole request:
//!
//! ```text
//! This model's maximum context length is 131072 tokens. However, you requested
//! 64000 output tokens and your prompt contains at least 67073 input tokens...
//! ```
//!
//! which kills exactly the long repo-review tasks the big window was for. So we
//! do the clamping the server won't: count the prompt, subtract it from the
//! window, and cap `max_tokens` at what is actually left.

/// Headroom left below the context window when clamping proactively. Covers
/// the drift between our token count and the server's own templating.
pub const PROACTIVE_RESERVE: u32 = 512;

/// Headroom used when recomputing from the server's own error message. The
/// server told us the exact input count there, so less slack is needed.
pub const REACTIVE_RESERVE: u32 = 256;

/// Below this many free tokens a turn is not worth attempting: the model has no
/// room for a useful answer, and the caller should compact instead.
pub const MIN_USABLE_OUTPUT: u32 = 1024;

/// Very rough token estimate for when the tokenizer endpoint is unavailable.
/// Three bytes per token runs short of real tokenizers on English + code, so
/// it overestimates the prompt and under-allocates output — the safe direction.
pub fn estimate_tokens_from_bytes(bytes: usize) -> u32 {
    u32::try_from(bytes / 3).unwrap_or(u32::MAX)
}

/// How many output tokens are left in `context` once `prompt_tokens` and
/// `reserve` are accounted for. `None` when the prompt does not fit at all.
pub fn available_output_tokens(context: u32, prompt_tokens: u32, reserve: u32) -> Option<u32> {
    context
        .checked_sub(prompt_tokens)
        .and_then(|left| left.checked_sub(reserve))
        .filter(|left| *left > 0)
}

/// The clamp itself: never ask for more than the caller wanted, never ask for
/// more than the window has left.
pub fn clamp_max_tokens(requested: u32, context: u32, prompt_tokens: u32, reserve: u32) -> Clamp {
    match available_output_tokens(context, prompt_tokens, reserve) {
        None => Clamp::PromptTooLong {
            prompt_tokens,
            context,
        },
        Some(available) if available < MIN_USABLE_OUTPUT => Clamp::PromptTooLong {
            prompt_tokens,
            context,
        },
        Some(available) if available < requested => Clamp::Clamped(available),
        Some(_) => Clamp::Unchanged,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Clamp {
    /// The request already fits; leave `max_tokens` alone.
    Unchanged,
    /// Send this smaller `max_tokens` instead.
    Clamped(u32),
    /// Nothing useful fits. The caller should surface a "prompt is too long"
    /// error so Claude Code compacts the conversation.
    PromptTooLong { prompt_tokens: u32, context: u32 },
}

/// Phrased the way the Anthropic API phrases it, because that is the wording
/// Claude Code recognizes as "compact and retry" rather than "the backend is
/// broken".
pub fn prompt_too_long_message(prompt_tokens: u32, context: u32) -> String {
    format!(
        "prompt is too long: {prompt_tokens} tokens > {context} maximum \
         (local model context window). Compact the conversation and retry."
    )
}

/// What a vLLM context-length rejection tells us.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextLengthError {
    pub context: u32,
    pub input_tokens: u32,
    pub requested_output_tokens: Option<u32>,
}

/// Pull the numbers out of vLLM's 400. The message is prose, so this matches on
/// the anchors rather than a rigid format:
///
/// ```text
/// This model's maximum context length is 131072 tokens. However, you requested
/// 64000 output tokens and your prompt contains at least 67073 input tokens.
/// ```
pub fn parse_context_length_error(message: &str) -> Option<ContextLengthError> {
    let lower = message.to_ascii_lowercase();
    if !lower.contains("maximum context length") {
        return None;
    }
    let context = number_after(&lower, "maximum context length is")?;
    let input_tokens = number_before(&lower, "input tokens")?;
    let requested_output_tokens = number_before(&lower, "output tokens");
    Some(ContextLengthError {
        context,
        input_tokens,
        requested_output_tokens,
    })
}

/// Recompute `max_tokens` from what the server just told us. `None` means the
/// prompt itself leaves no usable room.
pub fn retry_max_tokens(error: &ContextLengthError) -> Option<u32> {
    available_output_tokens(error.context, error.input_tokens, REACTIVE_RESERVE)
        .filter(|available| *available >= MIN_USABLE_OUTPUT)
}

/// First integer appearing after `anchor`.
fn number_after(haystack: &str, anchor: &str) -> Option<u32> {
    let start = haystack.find(anchor)? + anchor.len();
    let rest = &haystack[start..];
    let digits: String = rest
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// Last integer appearing before `anchor` (e.g. the "67073" in
/// "at least 67073 input tokens").
fn number_before(haystack: &str, anchor: &str) -> Option<u32> {
    let end = haystack.find(anchor)?;
    let head = &haystack[..end];
    let digits: String = head
        .chars()
        .rev()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(char::is_ascii_digit)
        .collect();
    digits.chars().rev().collect::<String>().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const VLLM_400: &str = "This model's maximum context length is 131072 tokens. However, you requested 64000 output tokens and your prompt contains at least 67073 input tokens. Please reduce the length of the messages or completion.";

    #[test]
    fn parses_the_production_vllm_rejection() {
        let parsed = parse_context_length_error(VLLM_400).expect("parsed");
        assert_eq!(parsed.context, 131072);
        assert_eq!(parsed.input_tokens, 67073);
        assert_eq!(parsed.requested_output_tokens, Some(64000));
    }

    #[test]
    fn retry_budget_matches_the_reported_numbers() {
        let parsed = parse_context_length_error(VLLM_400).expect("parsed");
        assert_eq!(retry_max_tokens(&parsed), Some(131072 - 67073 - 256));
    }

    #[test]
    fn unrelated_errors_are_not_treated_as_context_overflow() {
        assert!(parse_context_length_error("model 'qwen' not found").is_none());
        assert!(parse_context_length_error("").is_none());
    }

    #[test]
    fn retry_refuses_when_the_prompt_leaves_no_room() {
        let error = ContextLengthError {
            context: 131072,
            input_tokens: 130_900,
            requested_output_tokens: Some(64000),
        };
        assert_eq!(retry_max_tokens(&error), None);
    }

    #[test]
    fn clamp_shrinks_an_oversized_request() {
        assert_eq!(
            clamp_max_tokens(64000, 131072, 67073, PROACTIVE_RESERVE),
            Clamp::Clamped(131072 - 67073 - 512)
        );
    }

    #[test]
    fn clamp_leaves_a_request_that_already_fits() {
        assert_eq!(
            clamp_max_tokens(8192, 131072, 1000, PROACTIVE_RESERVE),
            Clamp::Unchanged
        );
    }

    #[test]
    fn clamp_reports_a_prompt_that_cannot_be_answered() {
        assert_eq!(
            clamp_max_tokens(64000, 131072, 131_000, PROACTIVE_RESERVE),
            Clamp::PromptTooLong {
                prompt_tokens: 131_000,
                context: 131072
            }
        );
        // Fits, but with less than a usable answer's worth of room left.
        assert!(matches!(
            clamp_max_tokens(64000, 131072, 130_000, PROACTIVE_RESERVE),
            Clamp::PromptTooLong { .. }
        ));
    }

    #[test]
    fn byte_estimate_is_conservative() {
        // 3 bytes/token overestimates the prompt, so the clamp errs small.
        assert_eq!(estimate_tokens_from_bytes(3000), 1000);
        assert_eq!(estimate_tokens_from_bytes(0), 0);
    }
}
