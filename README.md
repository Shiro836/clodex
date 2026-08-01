# clodex

Launch an **unmodified** Claude Code against either your **ChatGPT / Codex
subscription** or the **DeepSeek API**, while plain `claude` keeps working in
parallel. All three commands share Claude Code's normal configuration.

This is a fork of [`raine/claude-code-proxy`](https://github.com/raine/claude-code-proxy)
(MIT). `clodex` reuses the tokens you already have from `codex login`
(`~/.codex/auth.json`), so there is **no separate OAuth step**. `claudeep`
connects directly to DeepSeek's Anthropic-compatible API.

## How it works

```
clodex ───► local proxy ───► chatgpt.com/backend-api/codex
              auth via your existing ~/.codex/auth.json

claudeep ──────────────────► api.deepseek.com/anthropic
              auth via DEEPSEEK_API_KEY

claude, clodex, and claudeep all exec the same unmodified Claude Code binary
and use its normal ~/.claude configuration. Provider variables are scoped to
the launched process.
```

- **Nothing is patched.** The Claude Code binary is never modified and survives
  updates. `clodex` and `claudeep` are sibling commands.
- **Shared Claude Code configuration.** Settings, plugins, MCP servers, history,
  and sessions remain under `~/.claude` for all three commands.
- **No provider interference.** Routing and authentication variables live only
  inside the launched process, so plain `claude` still talks to Anthropic.
- **One source of truth for Codex auth.** The fork reads *and writes back*
  `~/.codex/auth.json` in Codex's own schema, so the rotating refresh token
  stays in sync between clodex and the official Codex CLI.

### The fork's proxy change

[`src/providers/codex/auth/codex_cli_store.rs`](src/providers/codex/auth/codex_cli_store.rs)
— an `AuthStorage` adapter backed by `~/.codex/auth.json`, selected when
`CCP_USE_CODEX_CLI_AUTH=1` (the `clodex` launcher sets this). Everything else in
the proxy remains upstream. The proxy already used Codex's own OAuth
`CLIENT_ID`, so token refresh works unchanged. Upstream's original README is
kept as [`README.upstream.md`](README.upstream.md).

## Install (Arch / AUR)

```sh
# from the AUR (once published):
yay -S clodex

# or build the local checkout:
cd packaging && makepkg -si
```

Installs `/usr/bin/clodex`, `/usr/bin/claudeep`, and `/usr/bin/clodex-proxy`.
Uninstall with `pacman -R clodex`.

## Use

### Codex

```sh
codex login            # once, if you aren't already logged into Codex
clodex                 # opens Claude Code on your Codex subscription
clodex --proxy status  # status | stop | restart | log
```

### DeepSeek

Store a DeepSeek API key once, then launch:

```sh
install -d -m 700 ~/.config/claudeep
printf '%s' "$DEEPSEEK_API_KEY" > ~/.config/claudeep/api-key
chmod 600 ~/.config/claudeep/api-key
claudeep
```

The key lookup order is `CLAUDEEP_API_KEY`, `DEEPSEEK_API_KEY`, then
`~/.config/claudeep/api-key`. Override the file location with
`CLAUDEEP_API_KEY_FILE`. DeepSeek API usage is billed separately from Codex and
Anthropic subscriptions.

## Model mapping

Both launchers keep Claude Code's native Opus/Sonnet/Haiku picker.

| Claude Code picker | `clodex` | `claudeep` |
|--------------------|----------|------------|
| Opus   | `gpt-5.6-sol` | `deepseek-v4-pro` |
| Sonnet | `gpt-5.6-terra` | `deepseek-v4-flash` |
| Haiku  | `gpt-5.6-luna` | `deepseek-v4-flash` |

DeepSeek performs its Claude-name mapping at the API endpoint. Neither launcher
pins the startup model by default, so the picker stays intact.

Pin an exact startup model with `CLODEX_MODEL` or `CLAUDEEP_MODEL`:

```sh
CLODEX_MODEL=gpt-5.6-luna-fast clodex
CLAUDEEP_MODEL='deepseek-v4-pro[1m]' claudeep
```

The background/subagent overrides are `CLODEX_SMALL_MODEL` and
`CLAUDEEP_SMALL_MODEL`, respectively.

## Note on terms

This reuses **your own single** ChatGPT/Codex subscription in a third-party
client — which OpenAI has publicly said it supports. Do **not** pool multiple
accounts or share credentials; that violates OpenAI's usage policy. Anthropic
is never contacted by `clodex` and the Claude binary is never altered.

## License

MIT, inherited from the upstream project. See [`LICENSE`](LICENSE).
