use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use http::StatusCode;
use serde::Serialize;

use crate::traffic::TrafficCapture;

const MAX_BUFFERED_RESPONSE_BYTES: usize = 32 * 1024 * 1024;

/// Minimal OpenAI-compatible HTTP client for a locally hosted inference server
/// (vLLM, llama.cpp, sglang, …). No auth handshake, no token refresh: the
/// server lives on loopback and either wants a static bearer token or nothing
/// at all.
pub struct LocalClient {
    client: Arc<reqwest::Client>,
    base_url: reqwest::Url,
    api_key: Option<String>,
}

pub struct LocalResponse {
    response: reqwest::Response,
}

#[derive(Debug)]
pub struct LocalError {
    pub status: StatusCode,
    pub retry_after: Option<String>,
    pub message: String,
}

impl LocalResponse {
    pub fn into_stream(
        self,
    ) -> impl futures_util::Stream<Item = Result<bytes::Bytes, LocalError>> + Send {
        self.response.bytes_stream().map(|chunk| {
            chunk.map_err(|error| LocalError {
                status: StatusCode::BAD_GATEWAY,
                retry_after: None,
                message: format!("local upstream stream failed: {error}"),
            })
        })
    }

    pub async fn into_bytes(self) -> Result<Vec<u8>, LocalError> {
        let mut stream = self.into_stream();
        let mut bytes = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            if bytes.len().saturating_add(chunk.len()) > MAX_BUFFERED_RESPONSE_BYTES {
                return Err(LocalError {
                    status: StatusCode::BAD_GATEWAY,
                    retry_after: None,
                    message: "local upstream response exceeds the size limit".to_string(),
                });
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }
}

impl LocalClient {
    pub fn new(base_url: String, api_key: Option<String>) -> anyhow::Result<Self> {
        let base_url = reqwest::Url::parse(base_url.trim_end_matches('/'))?;
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(10))
            // A local 27B model on a single consumer GPU can spend minutes on a
            // long prefill before the first token, so no total-request timeout.
            .pool_idle_timeout(Duration::from_secs(300))
            .build()?;
        Ok(Self {
            client: Arc::new(client),
            base_url,
            api_key,
        })
    }

    pub fn base_url(&self) -> &str {
        self.base_url.as_str()
    }

    pub async fn post_chat_completions<T: Serialize + ?Sized>(
        &self,
        body: &T,
        stream: bool,
        traffic: Option<Arc<TrafficCapture>>,
    ) -> Result<LocalResponse, LocalError> {
        let url = self.chat_completions_url();
        let accept = if stream {
            "text/event-stream"
        } else {
            "application/json"
        };

        if let Some(capture) = traffic.as_ref() {
            let value = serde_json::to_value(body).unwrap_or(serde_json::Value::Null);
            capture.write_json("020-upstream-request", &value);
            capture.write_json(
                "021-upstream-request-metadata",
                &serde_json::json!({
                    "method": "POST",
                    "url": url.as_str(),
                    "provider": "local",
                    "transport": "http",
                    "headers": {
                        "accept": accept,
                        "content-type": "application/json"
                    }
                }),
            );
        }

        let mut request = self
            .client
            .post(url.clone())
            .header(http::header::ACCEPT, accept)
            .header(http::header::CONTENT_TYPE, "application/json")
            .json(body);
        if let Some(key) = self.api_key.as_deref().filter(|key| !key.is_empty()) {
            request = request.header(http::header::AUTHORIZATION, format!("Bearer {key}"));
        }

        let response = request.send().await.map_err(|error| LocalError {
            status: StatusCode::BAD_GATEWAY,
            retry_after: None,
            message: format!(
                "local inference server at {url} is unreachable: {error}. Is the qwen-mode-* systemd unit up?"
            ),
        })?;

        let status = StatusCode::from_u16(response.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        if status.is_success() {
            return Ok(LocalResponse { response });
        }

        let retry_after = response
            .headers()
            .get(http::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let text = response.text().await.unwrap_or_default();
        Err(LocalError {
            status,
            retry_after,
            message: if text.is_empty() {
                format!("local upstream returned HTTP {status}")
            } else {
                format!("local upstream returned HTTP {status}: {text}")
            },
        })
    }

    fn chat_completions_url(&self) -> reqwest::Url {
        let mut url = self.base_url.clone();
        {
            let mut segments = match url.path_segments_mut() {
                Ok(segments) => segments,
                Err(()) => return self.base_url.clone(),
            };
            segments.pop_if_empty().push("chat").push("completions");
        }
        url
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_completions_url_appends_to_versioned_base() {
        let client = LocalClient::new("http://127.0.0.1:8000/v1".into(), None).expect("client");
        assert_eq!(
            client.chat_completions_url().as_str(),
            "http://127.0.0.1:8000/v1/chat/completions"
        );
    }

    #[test]
    fn chat_completions_url_tolerates_trailing_slash() {
        let client = LocalClient::new("http://127.0.0.1:8001/v1/".into(), None).expect("client");
        assert_eq!(
            client.chat_completions_url().as_str(),
            "http://127.0.0.1:8001/v1/chat/completions"
        );
    }
}
