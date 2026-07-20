# Hicortex memory plugin for Hermes

> **Install:** `hermes plugins install gamaze-labs/hicortex-hermes-plugin` → `hermes memory setup hicortex` → restart your gateway.
>
> The [gamaze-labs/hicortex-hermes-plugin](https://github.com/gamaze-labs/hicortex-hermes-plugin) repo is a **generated read-only mirror** of `hermes-plugin/hicortex/` in the main Hicortex repo — do not open PRs there. Requires a running [Hicortex server](https://hicortex.gamaze.com/docs/installation.html) (local or remote) for recall; capture of Hermes sessions is handled by the server machine's nightly job.

Gives [Hermes](https://github.com/nousresearch/hermes-agent) agents self-learning memory backed by a [Hicortex](https://hicortex.gamaze.com/) server: their experience is distilled into lessons overnight, and they wake up wiser. **Recall-only:** the plugin retrieves relevant memories on every turn and injects distilled lessons into the system prompt. It has **no local LLM, no capture, no cron** — it is a thin recall shim.

**Capture happens centrally.** A nightly reader on the Hicortex server distills each agent's own session store (Hermes keeps full history in `~/.hermes/profiles/<agent>/state.db`), so nothing needs to be captured in real time. See `specs/2026-07-01-memory-capture-architecture.md` in the main repo.

## How it works

| Hermes hook | What it does | Hicortex call |
|---|---|---|
| `prefetch(query)` | recall relevant memories before each turn | `GET /search` |
| `queue_prefetch(query)` | background recall for the next turn | `GET /search` |
| `system_prompt_block()` | inject per-agent standing context + distilled lessons + memory index | `GET /context`, `GET /lessons` |
| `get_tool_schemas()` | exposes the 8 unified tools | see tool table below |

That's the whole surface. No `sync_turn`, no compaction/session-end capture — those are intentionally absent.

### Per-agent standing context (0.13)

`system_prompt_block()` also injects the hand-edited **standing context layer** (`## Context`, above the lessons block) — "who you are + how to work", distinct from episodic memory. The server resolves it **per agent**: this profile's own sections override the global set (`override`), or it can be `global` or `off`. See the main repo's `/context` layer docs.

The plugin sends its **profile name** as `?agent=`, resolved in this order:

1. `agent_name` in the plugin config (explicit override);
2. the `HERMES_PROFILE` environment variable;
3. a `HERMES_HOME` ending in `profiles/<name>` (the per-profile install path);
4. none → the global context (backward compatible).

Leave `agent_name` blank to auto-derive (2–4). Context injection needs a Hicortex server **≥ 0.13**; against an older server the plugin detects the missing per-agent support and injects no context (lessons are unaffected). Context and lessons fail soft independently — a context failure never costs the lessons block.

### Tools (unified 8)

| Tool | REST call | Description |
|---|---|---|
| `hicortex_search` | `GET /search` | Semantic search over long-term memory |
| `hicortex_recent` | `GET /recent` | Recent memories by project (queryless recall; was `hicortex_context`/`hicortex_recall_recent` before 0.12) |
| `hicortex_ingest` | `POST /ingest` | Store a new memory |
| `hicortex_lessons` | `GET /lessons` | Get distilled lessons |
| `hicortex_index` | `GET /index` | Knowledge domain index |
| `hicortex_graph` | `GET /graph` | Graph queries (neighbors/hubs/path) |
| `hicortex_update` | `POST /update` | Update a memory (re-embeds on content change) |
| `hicortex_delete` | `POST /delete` | Permanently delete a memory and its links |

## Prerequisites

- A reachable Hicortex server (default `http://localhost:8787`). Stand one up with `npx @gamaze/hicortex init`.
- The server needs the REST `/search`, `/recent`, `/lessons` endpoints (Hicortex ≥ 0.12 — this plugin version does not talk to older servers; upgrade the server first).

## Install

Hermes discovers user-installed providers from `$HERMES_HOME/plugins/<name>/`:

```bash
cp -r hermes-plugin/hicortex "$HERMES_HOME/plugins/hicortex"
```

(No `pip install` — the plugin is stdlib-only.)

## Configure & activate

Use Hermes' own tooling — it discovers this plugin automatically and writes `config.yaml` correctly (**never hand-edit `config.yaml` with scripts/regex**):

```bash
hermes memory setup   # select "hicortex", enter the server URL/token when prompted
```

Run it once per profile if you use Hermes profiles. Hermes allows **one** external memory provider at a time, so disable Honcho (or any other) first, then restart the gateway.

Config fields (`hicortex_url`, `default_project`, `recall_limit`, `privacy_filter`, `agent_name`) can also be written to `$HERMES_HOME/plugins/hicortex/config.json` directly. `agent_name` pins the per-agent context id for this profile (leave blank to auto-derive — see [Per-agent standing context](#per-agent-standing-context-013)). The auth token is a **secret** — set it via env, not the JSON file:

```bash
export HICORTEX_AUTH_TOKEN=hctx-default-token   # or your custom token
```

Env overrides: `HICORTEX_URL`, `HICORTEX_AUTH_TOKEN`.

## Topology

- **Server host:** runs Hicortex. Set `hicortex_url: http://localhost:8787` (localhost bypasses auth).
- **Other Hermes boxes:** set `hicortex_url` to the server's Tailscale hostname (e.g. `http://memory-server:8787`) and `HICORTEX_AUTH_TOKEN` to the server's token. Each box recalls from the same shared brain.

## Notes

- Localhost requests skip auth; remote requests require the bearer token.
- Recall failures are non-fatal — the plugin returns empty context and the turn proceeds.
