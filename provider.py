"""Hicortex MemoryProvider for Hermes — recall-only.

Recall:   prefetch()          -> GET /search   (relevant memories before each turn)
          queue_prefetch()    -> GET /search   (background recall for the next turn)
          tools               -> hicortex_search / hicortex_recent
          system_prompt_block -> lessons + memory index injected into the prompt

Capture is NOT the plugin's job. A nightly reader on the Hicortex server
distills each agent's own session store (Hermes: ~/.hermes/profiles/<agent>/
state.db) centrally — see specs/2026-07-01-memory-capture-architecture.md. This
plugin has no local LLM, no spool, no timer, and no capture path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional

from agent.memory_provider import MemoryProvider

from .client import HicortexClient
from .config import CONFIG_SCHEMA, load_config

logger = logging.getLogger(__name__)

_INJECT_CONTENT_CAP = 500

# Agent ids are joined into a filesystem path server-side, so they share the
# section-name allowlist. \Z (NOT $) anchors the END OF STRING: Python's $ also
# matches just before a trailing "\n", so "nano\n" would pass and go out as
# agent=nano%0A → a 400 the fail-soft path silently swallows.
_AGENT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*\Z")


def _valid_agent_id(name: Optional[str]) -> bool:
    return bool(name) and len(name) <= 64 and bool(_AGENT_ID_RE.match(name))


def _sanitize_agent_id(raw: Optional[str]) -> Optional[str]:
    """Sanitize a raw identity (profile name / env value) into a valid agent id,
    or None when nothing valid remains — mirrors the TS ``sanitizeAgentId``
    EXACTLY so a profile resolves to the SAME id on both harnesses (a mismatch
    would make one honor the persona firewall and the other leak global context
    into an ``off``/``override`` persona): lowercase → collapse invalid runs to
    "-" → strip leading -/_ → truncate 64 → validate. "Lenny" → "lenny";
    "MacBook-Pro.local" → "macbook-pro-local"; all-symbols → None."""
    if not isinstance(raw, str):
        return None
    cleaned = re.sub(r"[^a-z0-9_-]+", "-", raw.lower())
    cleaned = re.sub(r"^[-_]+", "", cleaned)[:64]
    return cleaned if _valid_agent_id(cleaned) else None


def _profile_from_home(home: str) -> Optional[str]:
    """Parse a Hermes profile name from a ``HERMES_HOME`` ending in
    ``…/profiles/<name>``; None when the path is not profile-shaped."""
    home = (home or "").strip().rstrip("/")
    if not home:
        return None
    parent, name = os.path.split(home)
    return name if name and os.path.basename(parent) == "profiles" else None


def _resolve_agent_name(cfg: Dict[str, Any]) -> Optional[str]:
    """Resolve the per-agent context id (0.13), in priority order:
      1. config ``agent_name`` (explicit override);
      2. ``HERMES_PROFILE`` env;
      3. parse ``HERMES_HOME`` when it ends ``profiles/<name>``;
      4. None → bare fetch → the global set.
    Each source is stripped then SANITIZED (not rejected) so "Lenny" → "lenny"
    matches the TS contract; a source that sanitizes to None yields None (bare
    fetch), never a fall-through to another identity."""
    configured = (cfg.get("agent_name") or "").strip()
    if configured:
        return _sanitize_agent_id(configured)
    prof = (os.environ.get("HERMES_PROFILE") or "").strip()
    if prof:
        return _sanitize_agent_id(prof)
    parsed = _profile_from_home(os.environ.get("HERMES_HOME") or "")
    if parsed:
        return _sanitize_agent_id(parsed)
    return None


def _title_case_section(name: str) -> str:
    """"user" → "User", "my_notes" → "My Notes" (mirrors the CC/OC helper)."""
    words = [w for w in re.split(r"[-_]+", name) if w]
    return " ".join(w[:1].upper() + w[1:] for w in words)


def _order_section_names(names: Iterable[str]) -> List[str]:
    """Stable ordering: user, rules, then the rest alphabetically."""
    names = list(names)
    primaries = [p for p in ("user", "rules") if p in names]
    rest = sorted(n for n in names if n not in ("user", "rules"))
    return primaries + rest


def _render_context_block(sections: Dict[str, Any]) -> str:
    """Render the ``## Context`` block, or "" when every section is blank."""
    body_parts: List[str] = []
    for name in _order_section_names(sections.keys()):
        body = sections.get(name)
        if not isinstance(body, str) or not body.strip():
            continue
        body_parts.extend([f"### {_title_case_section(name)}", "", body.strip()])
    if not body_parts:
        return ""
    return "\n".join(["## Context", "", *body_parts])


class HicortexProvider(MemoryProvider):
    """Hicortex long-term memory backend for Hermes (recall-only)."""

    def __init__(self):
        self._client: Optional[HicortexClient] = None
        self._project: Optional[str] = None
        self._recall_limit: int = 5
        self._privacy: Optional[str] = "WORK,PERSONAL"
        self._agent_name: Optional[str] = None
        self._prefetch_cache: Dict[str, str] = {}
        self._bg_threads: List[threading.Thread] = []

    @property
    def name(self) -> str:
        return "hicortex"

    # ------------------------------------------------------------------ config
    def _build_client(self) -> Optional[HicortexClient]:
        cfg = load_config()
        url = cfg.get("hicortex_url")
        if not url:
            return None
        token = cfg.get("hicortex_auth_token")
        return HicortexClient(url, auth_token=token or None)

    def _client_or_none(self) -> Optional[HicortexClient]:
        if self._client is None:
            try:
                self._client = self._build_client()
            except Exception as e:
                logger.warning("hicortex: failed to build client: %s", e)
        return self._client

    def is_available(self) -> bool:
        """Configured and ready — NO network call (per MemoryProvider contract).

        ``is_available`` runs at agent init to decide whether to activate this
        provider. Pinging the server here would mean a slow or momentarily-down
        server silently disables memory for the whole session. Per the contract
        ("should not make network calls — just check config and installed deps")
        we only verify a server URL is configured; per-request failures are
        handled at use time.
        """
        return self._build_client() is not None

    def initialize(self, session_id: str, **kwargs) -> None:
        cfg = load_config()
        self._project = cfg.get("default_project") or None
        try:
            self._recall_limit = int(cfg.get("recall_limit", 5))
        except (TypeError, ValueError):
            self._recall_limit = 5
        self._privacy = cfg.get("privacy_filter", "WORK,PERSONAL")
        self._agent_name = _resolve_agent_name(cfg)
        try:
            self._client = self._build_client()
        except Exception as e:
            logger.warning("hicortex: init client build failed: %s", e)

    # ------------------------------------------------------------------- recall
    def _format_hits(self, hits: list[dict]) -> str:
        if not hits:
            return ""
        lines = [
            "Relevant prior context from your long-term memory "
            "(verify before relying on these — each shows date and project):"
        ]
        for h in hits[: self._recall_limit]:
            date = (h.get("created_at") or "")[:10]
            proj = h.get("project") or "global"
            content = (h.get("content") or "").strip().replace("\n", " ")
            if len(content) > _INJECT_CONTENT_CAP:
                content = content[:_INJECT_CONTENT_CAP] + "…"
            lines.append(f"- [{date}, {proj}] {content}")
        return "\n".join(lines)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        key = hashlib.sha1(query.encode("utf-8")).hexdigest()
        cached = self._prefetch_cache.pop(key, None)
        if cached is not None:
            return cached
        client = self._client_or_none()
        if client is None:
            return ""
        try:
            hits = client.search(
                query, limit=self._recall_limit, project=self._project, privacy=self._privacy
            )
            return self._format_hits(hits)
        except Exception as e:
            logger.debug("hicortex prefetch failed: %s", e)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        client = self._client_or_none()
        if client is None:
            return
        key = hashlib.sha1(query.encode("utf-8")).hexdigest()

        def _bg() -> None:
            try:
                hits = client.search(
                    query, limit=self._recall_limit, project=self._project, privacy=self._privacy
                )
                self._prefetch_cache[key] = self._format_hits(hits)
            except Exception as e:
                logger.debug("hicortex queue_prefetch failed: %s", e)

        self._spawn(_bg)

    # ------------------------------------------------------ system prompt/tools
    def system_prompt_block(self) -> str:
        client = self._client_or_none()
        if client is None:
            return ""
        # Standing context (L2, 0.13) is prepended ABOVE the lessons block. The
        # two fetches run CONCURRENTLY (matching the TS Promise.all paths): run
        # serially, a blackholed server would stall the turn for up to 2× the
        # client timeout. Each block fails soft independently — a context failure
        # must never cost the lessons block, and vice versa.
        with ThreadPoolExecutor(max_workers=2) as executor:
            f_context = executor.submit(self._context_block, client)
            f_lessons = executor.submit(self._lessons_block, client)
            blocks = [f_context.result(), f_lessons.result()]
        return "\n\n".join(b for b in blocks if b)

    def _context_block(self, client: HicortexClient) -> str:
        """Fetch the standing context layer and render a ``## Context`` block,
        or "" when nothing should be injected. Gates (ALL): "hermes" in the
        server-resolved ``clients``; when an agent id was SENT, the response
        echoes ``agent`` (old-server guard — a pre-0.13 server ignores ?agent=
        and returns global with no echo; injecting would push global context
        into every persona; the check is skipped on a bare fetch); and the
        resolved section set is non-empty (mode "off" → {}).

        Reference implementation for the gate: TS ``gateAndRenderContext`` in
        ``packages/hicortex/src/lessons-context.ts`` (keep the two in sync).

        The ENTIRE path — fetch, parse, gate, render — is inside the try: a
        malformed ``clients`` value (e.g. an int from a proxy error page) would
        otherwise raise during the ``in`` check, escape, and cost the lessons
        block too (mirrors the TS ``.catch(() => null)`` totality)."""
        try:
            data = client.context(agent=self._agent_name)
            if not isinstance(data, dict):
                return ""
            clients = data.get("clients") or []
            if "hermes" not in clients:
                return ""
            if self._agent_name is not None and not isinstance(data.get("agent"), str):
                return ""
            sections = data.get("sections") or {}
            if not isinstance(sections, dict):
                return ""
            return _render_context_block(sections)
        except Exception as e:
            logger.debug("hicortex context injection failed: %s", e)
            return ""

    def _lessons_block(self, client: HicortexClient) -> str:
        try:
            data = client.lessons()
        except Exception as e:
            logger.debug("hicortex lessons fetch failed: %s", e)
            return ""
        lessons = (data.get("lessons") or [])[:8]
        idx = data.get("index") or {}
        lines = [
            "## Hicortex long-term memory",
            "You have shared long-term memory across sessions. Use `hicortex_search` "
            "for specific recall and `hicortex_recent` for recent memories by project.",
        ]
        if lessons:
            lines.append("Lessons:")
            for l in lessons:
                c = (l.get("content") or "").strip().replace("\n", " ")
                lines.append(f"- {c[:200]}")
        if idx.get("total"):
            lines.append(
                f"({idx.get('total')} memories, {idx.get('lessonCount')} lessons "
                f"across {idx.get('sourceCount')} agents)"
            )
        return "\n".join(lines)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "hicortex_search",
                "description": (
                    "Search long-term memory using semantic similarity. Returns the most "
                    "relevant memories from past sessions."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query text"},
                        "limit": {
                            "type": "number",
                            "description": "Max results (default 5)",
                        },
                        "project": {"type": "string", "description": "Filter by project name"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "hicortex_recent",
                "description": (
                    "Get recent memories, optionally filtered by project. Queryless recall "
                    "of the latest memories by project, ranked by importance. Useful to "
                    "catch up on what happened recently."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project": {"type": "string", "description": "Filter by project name"},
                        "limit": {"type": "number", "description": "Max results (default 10)"},
                    },
                },
            },
            {
                "name": "hicortex_ingest",
                "description": (
                    "Store a new memory in long-term storage. "
                    "Use for important facts, decisions, or lessons."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory content to store"},
                        "project": {"type": "string", "description": "Project this memory belongs to"},
                        "memory_type": {
                            "type": "string",
                            "enum": ["episode", "lesson", "fact", "decision"],
                            "description": "Type of memory (default: episode)",
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "hicortex_lessons",
                "description": (
                    "Get actionable lessons learned from past sessions. "
                    "Auto-generated insights about mistakes to avoid."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project": {"type": "string", "description": "Filter by project name"},
                    },
                },
            },
            {
                "name": "hicortex_index",
                "description": (
                    "Get the knowledge domain index — shows what topics and projects "
                    "are stored in memory, grouped by domain."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "hicortex_graph",
                "description": (
                    "Query the memory knowledge graph — find connected memories, "
                    "hub nodes, or paths between memories."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["neighbors", "hubs", "path"],
                            "description": "Graph operation to perform",
                        },
                        "id": {"type": "string", "description": "Memory ID (required for neighbors and path operations)"},
                        "target_id": {"type": "string", "description": "Target memory ID (required for path operation)"},
                        "limit": {"type": "number", "description": "Max results (default 10)"},
                        "domain": {"type": "string", "description": "Filter hubs by domain"},
                        "relationship": {
                            "type": "string",
                            "description": "Filter neighbors by relationship type (e.g., CONTRADICTS, SUPERSEDES, derives)",
                        },
                    },
                    "required": ["operation"],
                },
            },
            {
                "name": "hicortex_update",
                "description": (
                    "Update an existing memory. Use after searching to fix incorrect information. "
                    "If content changes, the embedding is re-computed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Memory ID (from search results, first 8 chars or full UUID)"},
                        "content": {"type": "string", "description": "New content text"},
                        "project": {"type": "string", "description": "New project name"},
                        "memory_type": {
                            "type": "string",
                            "enum": ["episode", "lesson", "fact", "decision"],
                            "description": "New memory type",
                        },
                    },
                    "required": ["id"],
                },
            },
            {
                "name": "hicortex_delete",
                "description": (
                    "Permanently delete a memory and its links. "
                    "Use when a memory is incorrect and should be removed entirely."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Memory ID (from search results, first 8 chars or full UUID)"},
                    },
                    "required": ["id"],
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        client = self._client_or_none()
        if client is None:
            return json.dumps({"error": "hicortex server not configured"})
        try:
            if tool_name == "hicortex_search":
                hits = client.search(
                    args.get("query", ""),
                    limit=int(args.get("limit", 5)),
                    project=args.get("project") or self._project,
                )
                return json.dumps(hits)

            elif tool_name == "hicortex_recent":
                hits = client.recent(
                    project=args.get("project") or self._project,
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps(hits)

            elif tool_name == "hicortex_ingest":
                content = args.get("content", "")
                if not content:
                    return json.dumps({"error": "content is required"})
                status, resp = client.ingest(
                    content=content,
                    source_agent="hermes/manual",
                    project=args.get("project") or self._project,
                    memory_type=args.get("memory_type", "episode"),
                )
                if status not in (200, 201):
                    return json.dumps({"error": resp.get("error", f"HTTP {status}")})
                id_val = resp.get("id") or ""
                return json.dumps({"id": id_val, "message": f"Memory stored (id: {id_val[:8]})"})

            elif tool_name == "hicortex_lessons":
                data = client.lessons()
                lessons = (data.get("lessons") or [])
                if not lessons:
                    return json.dumps({"message": "No lessons found."})
                return json.dumps([{"content": l.get("content", "")[:500]} for l in lessons])

            elif tool_name == "hicortex_index":
                return json.dumps(client.index())

            elif tool_name == "hicortex_graph":
                op = args.get("operation", "")
                result = client.graph(
                    op=op,
                    id=args.get("id"),
                    target_id=args.get("target_id"),
                    limit=args.get("limit"),
                    domain=args.get("domain"),
                    relationship=args.get("relationship"),
                )
                return json.dumps(result)

            elif tool_name == "hicortex_update":
                id_val = args.get("id", "")
                if not id_val:
                    return json.dumps({"error": "id is required"})
                status, resp = client.update(
                    id=id_val,
                    content=args.get("content"),
                    project=args.get("project"),
                    memory_type=args.get("memory_type"),
                )
                if status == 404:
                    return json.dumps({"error": f"Memory not found: {id_val}"})
                if status not in (200, 201):
                    return json.dumps({"error": resp.get("error", f"HTTP {status}")})
                return json.dumps({"updated": True, "id": resp.get("id", id_val)})

            elif tool_name == "hicortex_delete":
                id_val = args.get("id", "")
                if not id_val:
                    return json.dumps({"error": "id is required"})
                status, resp = client.delete(id=id_val)
                if status == 404:
                    return json.dumps({"error": f"Memory not found: {id_val}"})
                if status not in (200, 201):
                    return json.dumps({"error": resp.get("error", f"HTTP {status}")})
                return json.dumps({"deleted": True, "id": resp.get("id", id_val)})

            else:
                return json.dumps({"error": f"unknown tool: {tool_name}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    # ---------------------------------------------------------------- lifecycle
    def _spawn(self, fn) -> None:
        self._bg_threads = [t for t in self._bg_threads if t.is_alive()]
        t = threading.Thread(target=fn, daemon=True)
        t.start()
        self._bg_threads.append(t)

    def shutdown(self) -> None:
        for t in self._bg_threads:
            t.join(timeout=2.0)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return CONFIG_SCHEMA

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        from .config import save_config as _save

        _save(values, hermes_home)
