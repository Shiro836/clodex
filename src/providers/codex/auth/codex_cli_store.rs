//! clodex fork addition.
//!
//! Storage adapter that reads and writes the *official Codex CLI* credential
//! file (`$CODEX_HOME/auth.json`, default `~/.codex/auth.json`) instead of the
//! proxy's own credential store.
//!
//! This lets clodex reuse the ChatGPT/Codex subscription tokens the user
//! already obtained by logging into the Codex CLI — no second OAuth login. The
//! proxy already authenticates with Codex's own OAuth `CLIENT_ID` and issuer
//! (see `constants.rs`), so the existing refresh logic in `manager.rs` works
//! unchanged against these tokens.
//!
//! Both clodex and the Codex CLI point at the *same* file, so the rotating
//! refresh token stays a single source of truth and the two clients never
//! invalidate each other's session.

use anyhow::Result;
use serde_json::{Value, json};
use std::path::PathBuf;

use super::token_store::StoredAuth;
use crate::auth::{AuthStorage, load_auth_file_value, write_atomically};

/// Resolve the Codex CLI credential path: `$CODEX_HOME/auth.json`, else
/// `~/.codex/auth.json`.
pub fn codex_auth_path() -> PathBuf {
    if let Ok(dir) = std::env::var("CODEX_HOME") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("auth.json");
        }
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| "/".to_string());
    PathBuf::from(home).join(".codex").join("auth.json")
}

pub struct CodexCliAuthStore {
    path: PathBuf,
}

impl CodexCliAuthStore {
    pub fn new() -> Self {
        Self {
            path: codex_auth_path(),
        }
    }

    #[cfg(test)]
    pub fn with_path(path: PathBuf) -> Self {
        Self { path }
    }
}

impl Default for CodexCliAuthStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Decode the `exp` claim (seconds since epoch) from a JWT and return it in
/// milliseconds. Returns `None` when the token isn't a decodable JWT with an
/// `exp` claim.
fn jwt_exp_ms(token: &str) -> Option<u64> {
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    let payload_b64 = parts[1].replace('-', "+").replace('_', "/");
    let padded = match payload_b64.len() % 4 {
        2 => format!("{payload_b64}=="),
        3 => format!("{payload_b64}="),
        _ => payload_b64,
    };
    use base64::Engine;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(&padded)
        .ok()?;
    let claims: Value = serde_json::from_slice(&decoded).ok()?;
    let exp = claims.get("exp")?.as_u64()?;
    Some(exp.saturating_mul(1000))
}

fn rfc3339_now() -> String {
    use time::OffsetDateTime;
    use time::format_description::well_known::Rfc3339;
    OffsetDateTime::now_utc().format(&Rfc3339).unwrap_or_default()
}

impl AuthStorage<StoredAuth> for CodexCliAuthStore {
    fn load(&self) -> Result<Option<StoredAuth>> {
        let Some(root) = load_auth_file_value(&self.path) else {
            return Ok(None);
        };
        let Some(tokens) = root.get("tokens") else {
            return Ok(None);
        };
        let access = tokens
            .get("access_token")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let refresh = tokens
            .get("refresh_token")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if access.is_empty() && refresh.is_empty() {
            return Ok(None);
        }
        let account_id = tokens
            .get("account_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        // Fall back to 0 (immediately expired) when the access token has no
        // readable expiry, so the manager refreshes using the refresh token.
        let expires = jwt_exp_ms(&access).unwrap_or(0);
        Ok(Some(StoredAuth {
            access,
            refresh,
            expires,
            account_id,
        }))
    }

    fn save(&self, value: StoredAuth) -> Result<()> {
        // Preserve the Codex CLI schema and any fields we don't manage
        // (`id_token`, `auth_mode`, `OPENAI_API_KEY`, ...).
        let mut root = load_auth_file_value(&self.path).unwrap_or_else(|| json!({}));
        if !root.is_object() {
            root = json!({});
        }
        let obj = root.as_object_mut().expect("root is an object");
        obj.entry("auth_mode").or_insert_with(|| json!("chatgpt"));
        obj.entry("OPENAI_API_KEY").or_insert(Value::Null);

        let tokens = obj.entry("tokens").or_insert_with(|| json!({}));
        if !tokens.is_object() {
            *tokens = json!({});
        }
        let t = tokens.as_object_mut().expect("tokens is an object");
        t.insert("access_token".into(), json!(value.access));
        t.insert("refresh_token".into(), json!(value.refresh));
        if let Some(account_id) = value.account_id {
            t.insert("account_id".into(), json!(account_id));
        }
        // `id_token` is intentionally left untouched: it is only used to derive
        // the account id (which we persist explicitly) and the Codex CLI
        // refreshes it on its own turns.

        obj.insert("last_refresh".into(), json!(rfc3339_now()));

        write_atomically(&self.path.to_string_lossy(), &root)
    }

    fn clear(&self) -> Result<()> {
        // Intentional no-op. This file belongs to the official Codex CLI;
        // deleting it would log the user out of Codex as well. On an
        // unrecoverable refresh failure the user re-runs `codex login`.
        Ok(())
    }

    fn path(&self) -> String {
        self.path.to_string_lossy().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_codex_file(path: &std::path::Path, access: &str, refresh: &str, account: &str) {
        let doc = json!({
            "auth_mode": "chatgpt",
            "OPENAI_API_KEY": null,
            "tokens": {
                "id_token": "id.tok.en",
                "access_token": access,
                "refresh_token": refresh,
                "account_id": account,
            },
            "last_refresh": "2026-01-01T00:00:00Z",
        });
        std::fs::write(path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
    }

    #[test]
    fn load_maps_codex_schema_to_stored_auth() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("auth.json");
        write_codex_file(&path, "acc", "ref", "acct_123");

        let store = CodexCliAuthStore::with_path(path);
        let loaded = store.load().unwrap().unwrap();
        assert_eq!(loaded.access, "acc");
        assert_eq!(loaded.refresh, "ref");
        assert_eq!(loaded.account_id.as_deref(), Some("acct_123"));
        // Non-JWT access token => forced-refresh sentinel.
        assert_eq!(loaded.expires, 0);
    }

    #[test]
    fn load_returns_none_when_file_missing() {
        let dir = tempfile::TempDir::new().unwrap();
        let store = CodexCliAuthStore::with_path(dir.path().join("nope.json"));
        assert!(store.load().unwrap().is_none());
    }

    #[test]
    fn save_writes_back_codex_schema_and_preserves_id_token() {
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("auth.json");
        write_codex_file(&path, "old", "old-ref", "acct_1");

        let store = CodexCliAuthStore::with_path(path.clone());
        store
            .save(StoredAuth {
                access: "new-access".into(),
                refresh: "new-refresh".into(),
                expires: 123,
                account_id: Some("acct_2".into()),
            })
            .unwrap();

        let raw: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(raw["tokens"]["access_token"], "new-access");
        assert_eq!(raw["tokens"]["refresh_token"], "new-refresh");
        assert_eq!(raw["tokens"]["account_id"], "acct_2");
        // Untouched Codex fields survive the round-trip.
        assert_eq!(raw["tokens"]["id_token"], "id.tok.en");
        assert_eq!(raw["auth_mode"], "chatgpt");
        assert!(raw["OPENAI_API_KEY"].is_null());

        // And the manager can read exactly what we wrote back.
        let reloaded = store.load().unwrap().unwrap();
        assert_eq!(reloaded.access, "new-access");
        assert_eq!(reloaded.refresh, "new-refresh");
        assert_eq!(reloaded.account_id.as_deref(), Some("acct_2"));
    }

    #[test]
    fn jwt_exp_ms_reads_expiry() {
        // {"exp":2000000000} base64url payload, no padding needed here.
        let token = "eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjIwMDAwMDAwMDB9.sig";
        assert_eq!(jwt_exp_ms(token), Some(2_000_000_000_000));
    }
}
