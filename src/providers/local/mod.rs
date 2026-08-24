pub mod client;
pub mod stream;

use std::sync::Arc;

use async_trait::async_trait;
use axum::{
    Json,
    body::Body,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::anthropic::{
    error::json_error,
    schema::{CountTokensResponse, MessagesRequest},
};
use crate::provider::{
    CliHandlers, Generation, GenerationBody, Provider, ProviderError, ProviderErrorKind,
    RequestContext,
};
use crate::providers::{kimi::count_tokens, opencode::chat};
use crate::registry::{is_anthropic_alias, normalize_incoming_model};

use self::client::{LocalClient, LocalError};

/// Model id advertised for people who want to address the local backend
/// explicitly instead of going through an Anthropic alias.
pub const LOCAL_MODEL_ALIAS: &str = "local";

enum ClientState {
    Ready(Arc<LocalClient>),
    Invalid(String),
}

/// Provider for a self-hosted OpenAI-compatible server (vLLM / llama.cpp).
///
/// Everything about the endpoint is configuration: `CCP_LOCAL_BASE_URL`
/// selects which server (and therefore which mode: fast/long on :8000,
/// smart on :8001), `CCP_LOCAL_MODEL` the model id sent upstream.
pub struct LocalProvider {
    client: ClientState,
    model: String,
}

impl LocalProvider {
    pub fn new() -> Self {
        let model = crate::config::local_model();
        let client = LocalClient::new(
            crate::config::local_base_url(),
            crate::config::local_api_key(),
        )
        .map(Arc::new)
        .map(ClientState::Ready)
        .unwrap_or_else(|error| ClientState::Invalid(error.to_string()));
        Self { client, model }
    }

    fn client(&self) -> Result<Arc<LocalClient>, String> {
        match &self.client {
            ClientState::Ready(client) => Ok(client.clone()),
            ClientState::Invalid(error) => Err(error.clone()),
        }
    }

    /// Anthropic aliases (and the bare `local` id) collapse onto the
    /// configured model; anything else is forwarded verbatim so a server that
    /// hosts several models can still be addressed by name.
    fn resolve_model(&self, requested: &str) -> String {
        let normalized = normalize_incoming_model(requested);
        if normalized.is_empty()
            || normalized == LOCAL_MODEL_ALIAS
            || is_anthropic_alias(&normalized)
            || normalized.starts_with("claude-")
        {
            return self.model.clone();
        }
        normalized
    }

    async fn buffered_messages_response(
        &self,
        body: MessagesRequest,
        ctx: RequestContext,
    ) -> Response {
        let requested = body.model.clone().unwrap_or_default();
        let resolved = self.resolve_model(&requested);
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, resolved.clone());
        }
        let client = match self.client() {
            Ok(client) => client,
            Err(error) => return invalid_configuration_response(error),
        };
        let message_id = format!("msg_{}", uuid::Uuid::new_v4().simple());

        let translated = match chat::prepare_request(&body, &resolved) {
            Ok(translated) => translated,
            Err(error) => return invalid_request_response(error),
        };
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.upstream_started(&ctx.req_id);
        }
        let upstream = match client
            .post_chat_completions(&translated, true, ctx.traffic.clone())
            .await
        {
            Ok(upstream) => upstream,
            Err(error) => return map_error(error),
        };
        let bytes = match upstream.into_bytes().await {
            Ok(bytes) => bytes,
            Err(error) => return map_error(error),
        };
        if let Some(traffic) = ctx.traffic.as_ref() {
            traffic.write_bytes("032-upstream-response-body.sse", &bytes);
        }
        let value = match chat::accumulate_response(&bytes, &message_id, &requested) {
            Ok(value) => value,
            Err(error) => return invalid_upstream_response(error),
        };
        if let Some(traffic) = ctx.traffic.as_ref() {
            traffic.write_json("051-downstream-response", &value);
        }
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.usage_updated(
                &ctx.req_id,
                value
                    .pointer("/usage/input_tokens")
                    .and_then(serde_json::Value::as_u64),
                value
                    .pointer("/usage/output_tokens")
                    .and_then(serde_json::Value::as_u64),
            );
        }
        (StatusCode::OK, Json(value)).into_response()
    }
}

impl Default for LocalProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Model ids the registry routes to this provider.
pub fn advertised_models() -> Vec<String> {
    let mut out = vec![LOCAL_MODEL_ALIAS.to_string()];
    let configured = crate::config::local_model();
    if configured != LOCAL_MODEL_ALIAS {
        out.push(configured);
    }
    out
}

#[async_trait]
impl Provider for LocalProvider {
    fn name(&self) -> &'static str {
        "local"
    }

    fn supported_models(&self) -> Vec<String> {
        advertised_models()
    }

    fn cli(&self) -> &'static dyn CliHandlers {
        &LOCAL_CLI
    }

    async fn handle_messages(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        if !body.stream {
            return self.buffered_messages_response(body, ctx).await;
        }
        match self.generate_anthropic_stream(body, ctx).await {
            Ok(generation) => sse_response(generation.body),
            Err(error) => map_provider_error(error),
        }
    }

    async fn handle_count_tokens(&self, body: MessagesRequest, ctx: RequestContext) -> Response {
        let resolved = self.resolve_model(body.model.as_deref().unwrap_or_default());
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, resolved);
        }
        let tokens = count_tokens::count_tokens(&body);
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.usage_updated(&ctx.req_id, Some(tokens), None);
        }
        (
            StatusCode::OK,
            Json(CountTokensResponse {
                input_tokens: tokens,
            }),
        )
            .into_response()
    }

    async fn generate_anthropic_stream(
        &self,
        mut body: MessagesRequest,
        ctx: RequestContext,
    ) -> Result<Generation, ProviderError> {
        body.stream = true;
        let requested = body.model.clone().unwrap_or_default();
        let resolved = self.resolve_model(&requested);
        let client = self.client().map_err(|error| {
            ProviderError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                ProviderErrorKind::Api,
                format!("Invalid local provider configuration: {error}"),
            )
        })?;
        if let Some(monitor) = ctx.monitor.as_ref() {
            monitor.model_resolved(&ctx.req_id, resolved.clone());
            monitor.upstream_started(&ctx.req_id);
        }
        let message_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
        let translated =
            chat::prepare_request(&body, &resolved).map_err(invalid_request_provider_error)?;
        let upstream = client
            .post_chat_completions(&translated, true, ctx.traffic.clone())
            .await
            .map_err(local_provider_error)?;
        let body = stream::stream_body(
            upstream,
            message_id,
            requested,
            ctx.monitor.clone(),
            ctx.req_id.clone(),
        );
        Ok(Generation {
            body: GenerationBody::LiveSse(body),
            resolved_model: resolved,
        })
    }
}

fn invalid_request_provider_error(error: impl std::fmt::Display) -> ProviderError {
    ProviderError::new(
        StatusCode::BAD_REQUEST,
        ProviderErrorKind::InvalidRequest,
        error.to_string(),
    )
}

fn local_provider_error(error: LocalError) -> ProviderError {
    let (status, kind) = match error.status {
        StatusCode::UNAUTHORIZED => (StatusCode::UNAUTHORIZED, ProviderErrorKind::Authentication),
        StatusCode::FORBIDDEN => (error.status, ProviderErrorKind::Permission),
        StatusCode::TOO_MANY_REQUESTS => {
            (StatusCode::TOO_MANY_REQUESTS, ProviderErrorKind::RateLimit)
        }
        status if status.is_client_error() => (status, ProviderErrorKind::InvalidRequest),
        _ => (StatusCode::BAD_GATEWAY, ProviderErrorKind::Api),
    };
    let mut mapped = ProviderError::new(status, kind, error.message);
    mapped.retry_after = error.retry_after;
    mapped
}

fn map_error(error: LocalError) -> Response {
    map_provider_error(local_provider_error(error))
}

fn map_provider_error(error: ProviderError) -> Response {
    let response = json_error(error.status, error.error_type(), error.message);
    if let Some(retry_after) = error.retry_after {
        ([(http::header::RETRY_AFTER, retry_after)], response).into_response()
    } else {
        response
    }
}

fn invalid_configuration_response(error: impl std::fmt::Display) -> Response {
    json_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "api_error",
        format!("Invalid local provider configuration: {error}"),
    )
}

fn invalid_request_response(error: impl std::fmt::Display) -> Response {
    json_error(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        error.to_string(),
    )
}

fn invalid_upstream_response(error: impl std::fmt::Display) -> Response {
    json_error(
        StatusCode::BAD_GATEWAY,
        "api_error",
        format!("Local response translation failed: {error}"),
    )
}

fn sse_response(body: GenerationBody) -> Response {
    let body = match body {
        GenerationBody::BufferedSse(bytes) => Body::from(bytes),
        GenerationBody::LiveSse(body) => body,
    };
    (
        [
            (http::header::CONTENT_TYPE, "text/event-stream"),
            (http::header::CACHE_CONTROL, "no-cache"),
            (http::header::CONNECTION, "keep-alive"),
        ],
        body,
    )
        .into_response()
}

struct LocalCli;

impl CliHandlers for LocalCli {
    fn login(&self) -> anyhow::Result<()> {
        println!("local: no authentication needed; point CCP_LOCAL_BASE_URL at the server");
        Ok(())
    }

    fn device(&self) -> anyhow::Result<()> {
        self.login()
    }

    fn status(&self) -> anyhow::Result<()> {
        println!("Base URL: {}", crate::config::local_base_url());
        println!("Model: {}", crate::config::local_model());
        println!(
            "API key: {}",
            if crate::config::local_api_key()
                .filter(|key| !key.is_empty())
                .is_some()
            {
                "set"
            } else {
                "not set (loopback server)"
            }
        );
        Ok(())
    }

    fn logout(&self) -> anyhow::Result<()> {
        println!("local: nothing to log out of");
        Ok(())
    }
}

static LOCAL_CLI: LocalCli = LocalCli;

#[cfg(test)]
mod tests {
    use super::*;

    fn provider_with_model(model: &str) -> LocalProvider {
        LocalProvider {
            client: ClientState::Invalid("unused".into()),
            model: model.to_string(),
        }
    }

    #[test]
    fn anthropic_aliases_resolve_to_configured_model() {
        let provider = provider_with_model("qwen3.8-27b");
        for alias in [
            "opus",
            "sonnet",
            "haiku",
            "fable",
            "claude-opus-5",
            "local",
            "",
        ] {
            assert_eq!(provider.resolve_model(alias), "qwen3.8-27b");
        }
    }

    #[test]
    fn one_million_hint_is_stripped_before_resolving() {
        let provider = provider_with_model("qwen3.8-27b");
        assert_eq!(provider.resolve_model("claude-opus-5[1m]"), "qwen3.8-27b");
    }

    #[test]
    fn explicit_upstream_model_is_forwarded_verbatim() {
        let provider = provider_with_model("qwen3.8-27b");
        assert_eq!(
            provider.resolve_model("gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090"),
            "gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090"
        );
    }
}
