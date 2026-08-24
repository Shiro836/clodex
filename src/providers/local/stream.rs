use std::convert::Infallible;

use axum::body::Body;
use bytes::Bytes;
use futures_util::StreamExt;

use super::client::{LocalError, LocalResponse};
use crate::monitor::{MonitorHandle, usage_from_anthropic_sse};
use crate::providers::opencode::chat::{LiveStreamTranslator, stream_error};

/// Bridge the upstream OpenAI-compatible SSE stream to Anthropic SSE while it
/// is still arriving. Buffering the whole completion first (the way the Kimi
/// provider does) would hide every token of a local model until it finished,
/// which for 70-230 tok/s on one GPU is the difference between a usable and an
/// unusable agent loop.
pub fn stream_body(
    upstream: LocalResponse,
    message_id: String,
    model: String,
    monitor: Option<MonitorHandle>,
    req_id: String,
) -> Body {
    let state = LocalStreamState {
        upstream: upstream.into_stream(),
        translator: LiveStreamTranslator::new(message_id, model),
        terminal: false,
        error_sent: false,
        monitor,
        req_id,
        bytes: 0,
    };
    let stream = futures_util::stream::unfold(state, |mut state| async move {
        state
            .next_output()
            .await
            .map(|bytes| (Ok::<Bytes, Infallible>(Bytes::from(bytes)), state))
    });
    Body::from_stream(stream)
}

struct LocalStreamState<S> {
    upstream: S,
    translator: LiveStreamTranslator,
    terminal: bool,
    error_sent: bool,
    monitor: Option<MonitorHandle>,
    req_id: String,
    bytes: u64,
}

impl<S> LocalStreamState<S>
where
    S: futures_util::Stream<Item = Result<Bytes, LocalError>> + Unpin,
{
    async fn next_output(&mut self) -> Option<Vec<u8>> {
        if self.terminal {
            return None;
        }
        if self.error_sent {
            self.terminal = true;
            return None;
        }
        loop {
            let chunk = match self.upstream.next().await {
                Some(Ok(chunk)) => chunk,
                Some(Err(error)) => return Some(self.fail(&error.message)),
                None => {
                    let output = match self.translator.finish() {
                        Ok(output) => output,
                        Err(error) => {
                            return Some(
                                self.fail(&format!("local stream ended prematurely: {error}")),
                            );
                        }
                    };
                    self.terminal = true;
                    return (!output.is_empty()).then_some(output);
                }
            };
            if self.bytes == 0
                && let Some(monitor) = self.monitor.as_ref()
            {
                monitor.generation_started(&self.req_id);
            }
            self.bytes = self.bytes.saturating_add(chunk.len() as u64);

            let output = match self.translator.push(&chunk) {
                Ok(output) => output,
                Err(error) => {
                    return Some(self.fail(&format!("local stream is invalid: {error}")));
                }
            };
            if !output.is_empty()
                && let Some(monitor) = self.monitor.as_ref()
            {
                let (input_tokens, output_tokens) = usage_from_anthropic_sse(&output);
                monitor.stream_progress(
                    &self.req_id,
                    output.len() as u64,
                    count_sse_events(&output),
                    input_tokens,
                    output_tokens,
                );
            }
            if self.translator.is_finished() {
                if let Err(error) = self.translator.finish() {
                    return Some(self.fail(&format!("local stream is invalid: {error}")));
                }
                self.terminal = true;
                return (!output.is_empty()).then_some(output);
            }
            if !output.is_empty() {
                return Some(output);
            }
        }
    }

    fn fail(&mut self, message: &str) -> Vec<u8> {
        self.error_sent = true;
        stream_error(message)
    }
}

fn count_sse_events(bytes: &[u8]) -> u64 {
    String::from_utf8_lossy(bytes).matches("event:").count() as u64
}
