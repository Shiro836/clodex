use serde::{Deserialize, Serialize};

use super::codex_cli_store::CodexCliAuthStore;
use crate::auth::{AuthStorage, KeychainFileAuthStore, SystemKeychain};
use crate::paths;

pub const KEYCHAIN_SERVICE: &str = "claude-code-proxy.codex";
pub const KEYCHAIN_ACCOUNT: &str = "auth";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoredAuth {
    pub access: String,
    pub refresh: String,
    pub expires: u64,
    #[serde(
        default,
        rename = "accountId",
        alias = "account_id",
        skip_serializing_if = "Option::is_none"
    )]
    pub account_id: Option<String>,
}

pub struct CodexTokenStore<S: AuthStorage<StoredAuth>> {
    store: S,
}

impl<S: AuthStorage<StoredAuth>> CodexTokenStore<S> {
    pub fn new(store: S) -> Self {
        Self { store }
    }

    pub fn load_auth(&self) -> Result<Option<StoredAuth>, anyhow::Error> {
        self.store.load()
    }

    pub fn save_auth(&self, value: StoredAuth) -> Result<(), anyhow::Error> {
        self.store.save(value)
    }

    pub fn clear_auth(&self) -> Result<(), anyhow::Error> {
        self.store.clear()
    }

    pub fn auth_path(&self) -> String {
        self.store.path()
    }
}

/// The default Codex credential store.
///
/// clodex fork: this used to be a plain alias for the proxy's own
/// keychain/file store. It is now an enum so `file_store()` can transparently
/// back onto the official Codex CLI credential file (`~/.codex/auth.json`) when
/// `CCP_USE_CODEX_CLI_AUTH` is set, without changing any call site that names
/// `DefaultCodexAuthStore`.
pub enum DefaultCodexAuthStore {
    /// Reuse the tokens the user already has from the Codex CLI login.
    CodexCli(CodexCliAuthStore),
    /// Upstream behaviour: the proxy's own keychain/file-backed store.
    Keychain(KeychainFileAuthStore<StoredAuth, SystemKeychain>),
}

impl AuthStorage<StoredAuth> for DefaultCodexAuthStore {
    fn load(&self) -> anyhow::Result<Option<StoredAuth>> {
        match self {
            Self::CodexCli(store) => store.load(),
            Self::Keychain(store) => store.load(),
        }
    }

    fn save(&self, value: StoredAuth) -> anyhow::Result<()> {
        match self {
            Self::CodexCli(store) => store.save(value),
            Self::Keychain(store) => store.save(value),
        }
    }

    fn clear(&self) -> anyhow::Result<()> {
        match self {
            Self::CodexCli(store) => store.clear(),
            Self::Keychain(store) => store.clear(),
        }
    }

    fn path(&self) -> String {
        match self {
            Self::CodexCli(store) => store.path(),
            Self::Keychain(store) => store.path(),
        }
    }
}

/// When `CCP_USE_CODEX_CLI_AUTH` is truthy, reuse the Codex CLI credential file
/// instead of the proxy's own store. clodex's launcher sets this.
fn use_codex_cli_auth() -> bool {
    match std::env::var("CCP_USE_CODEX_CLI_AUTH") {
        Ok(value) => {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        }
        Err(_) => false,
    }
}

pub fn file_store() -> CodexTokenStore<DefaultCodexAuthStore> {
    if use_codex_cli_auth() {
        return CodexTokenStore::new(DefaultCodexAuthStore::CodexCli(CodexCliAuthStore::new()));
    }
    let primary = paths::provider_auth_file("codex");
    let legacy = paths::provider_legacy_auth_file("codex");
    let store = KeychainFileAuthStore::new(
        primary.to_string_lossy().to_string(),
        legacy.to_string_lossy().to_string(),
        KEYCHAIN_SERVICE,
        KEYCHAIN_ACCOUNT,
        use_macos_keychain(),
        SystemKeychain,
    );
    CodexTokenStore::new(DefaultCodexAuthStore::Keychain(store))
}

fn use_macos_keychain() -> bool {
    cfg!(target_os = "macos") && std::env::var_os("CCP_CONFIG_DIR").is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::InMemoryAuthStore;
    use serde_json::json;

    #[test]
    fn stored_auth_reads_account_id_alias() {
        let auth: StoredAuth = serde_json::from_value(json!({
            "access": "a",
            "refresh": "r",
            "expires": 123,
            "accountId": "acct"
        }))
        .unwrap();
        assert_eq!(auth.account_id.as_deref(), Some("acct"));
    }

    #[test]
    fn stored_auth_writes_account_id_key() {
        let auth = StoredAuth {
            access: "a".into(),
            refresh: "r".into(),
            expires: 4102444800000,
            account_id: Some("acct_1".into()),
        };
        let value = serde_json::to_value(auth).unwrap();
        assert_eq!(value["accountId"], "acct_1");
        assert!(value.get("account_id").is_none());
    }

    #[test]
    fn stored_auth_roundtrip() {
        let store = CodexTokenStore::new(InMemoryAuthStore::new());
        let auth = StoredAuth {
            access: "token".into(),
            refresh: "refresh".into(),
            expires: 9999999999999,
            account_id: Some("acct_1".into()),
        };
        store.save_auth(auth.clone()).unwrap();
        let loaded = store.load_auth().unwrap().unwrap();
        assert_eq!(loaded.access, "token");
        assert_eq!(loaded.account_id.as_deref(), Some("acct_1"));
    }
}
