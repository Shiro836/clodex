# clodex

Launch an **unmodified** Claude Code that talks to your **ChatGPT / Codex
subscription** instead of the Anthropic API — while your normal `claude` keeps
working exactly as before, in parallel, untouched.

This is a fork of [`raine/claude-code-proxy`](https://github.com/raine/claude-code-proxy)
(MIT) with one addition: it reuses the tokens you already have from
`codex login` (`~/.codex/auth.json`), so there is **no separate OAuth step**.

## How it works

```
clodex ─► ensure local proxy is running (auto-started once, reused after)
       ─► exec /opt/claude-code/bin/claude with env vars scoped to THIS process:
            ANTHROPIC_BASE_URL   = http://127.0.0.1:18765   (the proxy)
            ANTHROPIC_CONFIG_DIR = ~/.clodex                (separate config)
                       │
                       ▼
       local proxy: Anthropic /v1/messages  ◄──►  chatgpt.com/backend-api/codex
                    auth via your own ~/.codex/auth.json (shared, single account)
```

- **Nothing is patched.** The Claude Code binary is never modified, so it can
  never break and survives updates. `clodex` is a sibling command.
- **No interference.** The env vars live only inside the process `clodex`
  spawns. Plain `claude` in another terminal sees none of them and keeps using
  `~/.claude` + Anthropic. `clodex` uses a separate `~/.clodex` config.
- **One source of truth for auth.** The fork reads *and writes back*
  `~/.codex/auth.json` in Codex's own schema, so the rotating refresh token
  stays in sync between clodex and the official Codex CLI.

### The fork's one change

[`src/providers/codex/auth/codex_cli_store.rs`](src/providers/codex/auth/codex_cli_store.rs)
— an `AuthStorage` adapter backed by `~/.codex/auth.json`, selected when
`CCP_USE_CODEX_CLI_AUTH=1` (the `clodex` launcher sets this). Everything else is
upstream. The proxy already used Codex's own OAuth `CLIENT_ID`, so token refresh
works unchanged. Upstream's original README is kept as
[`README.upstream.md`](README.upstream.md).

## Install (Arch / AUR)

```sh
# from the AUR (once published):
yay -S clodex

# or build the local checkout:
cd packaging && makepkg -si
```

Installs `/usr/bin/clodex` and `/usr/bin/clodex-proxy`. Uninstall: `pacman -R clodex`.

## Use

```sh
codex login          # once, if you aren't already logged into Codex
clodex               # opens Claude Code on your Codex subscription
clodex --proxy status   # status | stop | restart | log
```

## Model mapping

Claude Code's model picker maps automatically:

| Claude UI | Codex model |
|-----------|-------------|
| opus      | gpt-5.6-sol |
| sonnet    | gpt-5.6-terra |
| haiku     | gpt-5.6-luna |

Override the startup defaults via `CLODEX_MODEL` / `CLODEX_SMALL_MODEL`.

The launcher enables gateway model discovery, so Claude Code's `/model` picker
lists the proxy's **full catalog** — the whole 5.6 lineup (`gpt-5.6-luna`,
`gpt-5.6-sol`, `gpt-5.6-terra`, plus `-fast` variants) and the older
`gpt-5.2 … gpt-5.5`. Switch in-session with `/model` or `/model <id>`.

## Note on terms

This reuses **your own single** ChatGPT/Codex subscription in a third-party
client — which OpenAI has publicly said it supports. Do **not** pool multiple
accounts or share credentials; that violates OpenAI's usage policy. Anthropic
is never contacted by `clodex` and the Claude binary is never altered.

## License

MIT, inherited from the upstream project. See [`LICENSE`](LICENSE).
