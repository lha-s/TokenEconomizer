"""Claude Code's session transcript: tool calls, token usage, subagents, deferred tools.

The interactive transcript uses the same `message.content[]` envelope as `stream-json`, so one
reader serves both. Usage is the difference: the transcript writes ONE ENTRY PER CONTENT BLOCK,
each repeating its message's `usage`, so summing entries over-counts by the number of blocks;
`claude_session_usage` keys by message id. Subagents write their own files under
`<session>/subagents/`, and their tool calls and tokens belong to the session that spawned them.

Only the conversation's own entries (`assistant`, `user`) carry its calls. Claude Code 2.1.300
also writes `api-request-blob` entries that repeat each request's messages, earlier tool calls
and results included; read as calls, every call counted twice.
"""
import glob
import json
import os
import re
from datetime import datetime

from .jsonl import lines as _lines

_CLAUDE_KEYS = ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
                "output_tokens")
CONVERSATION = ("assistant", "user")


def claude_calls(lines):
    """[(name, args, result_text, success, cwd)] for the lines of a Claude transcript: each
    `tool_use` block paired with its `tool_result` by id, with the directory the shell was in."""
    out, order, seen = [], [], {}
    results = {}
    for line in lines:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type", "assistant") not in CONVERSATION:
            continue
        msg = ev.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                tid = b.get("id")
                if tid in seen:
                    continue
                seen[tid] = {"name": b.get("name"), "args": b.get("input", {}),
                             "cwd": ev.get("cwd") if isinstance(ev.get("cwd"), str) else None}
                order.append(tid)
            elif b.get("type") == "tool_result":
                tid = b.get("tool_use_id")
                c = b.get("content")
                if isinstance(c, list):
                    c = " ".join(x.get("text", "") for x in c
                                 if isinstance(x, dict))
                results[tid] = (c if isinstance(c, str) else "",
                                not b.get("is_error", False))
    for tid in order:
        info = seen[tid]
        res_text, ok = results.get(tid, ("", True))
        out.append((info["name"], info["args"], res_text, ok, info["cwd"]))
    return out


def _is_prompt(ev):
    """Is this entry a prompt the person gave, typed or as a slash command? Its own `user`
    entries, less tool results, a local command's output, an interrupt marker, a compaction
    summary, the meta text Claude Code adds, and a subagent's."""
    if (ev.get("type") != "user" or ev.get("isMeta") or ev.get("isSidechain")
            or ev.get("isCompactSummary")):
        return False
    content = (ev.get("message") or {}).get("content")
    if isinstance(content, list):
        blocks = [b for b in content if isinstance(b, dict)]
        if any(b.get("type") == "tool_result" for b in blocks):
            return False
        content = " ".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
    return isinstance(content, str) and not content.startswith(("<local-command",
                                                                "[Request interrupted"))


def claude_prompt_count(transcript_path):
    """How many prompts the person gave a Claude Code session (`_is_prompt`)."""
    n = 0
    for line in _lines(transcript_path):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        n += isinstance(ev, dict) and _is_prompt(ev)
    return n



def claude_subagent_files(transcript_path):
    """Subagent transcripts of a Claude Code session: `<dir>/<session-id>/subagents/*.jsonl`."""
    if not transcript_path:
        return []
    stem = os.path.splitext(transcript_path)[0]
    return sorted(glob.glob(os.path.join(stem, "subagents", "*.jsonl")))


def claude_session_usage(transcript_path, include_subagents=True):
    """Token breakdown of a Claude Code session, de-duplicated by assistant message id.

    Returns the four breakdown fields `pricing.effective_tokens` reads, plus `model` (the most
    used one), `requests` (distinct assistant messages carrying usage) and `files` read."""
    files = [transcript_path] + (claude_subagent_files(transcript_path)
                                 if include_subagents else [])
    by_id, models, anon = {}, {}, 0
    for path in files:
        for line in _lines(path):
            if '"usage"' not in line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("type") != "assistant":
                continue
            msg = ev.get("message") or {}
            usage = msg.get("usage")
            if not isinstance(usage, dict):
                continue
            mid = msg.get("id")
            if not mid:
                anon += 1
                mid = f"anon-{path}-{anon}"
            # Later entries of the same message carry the final counts; keep the last.
            by_id[mid] = {k: int(usage.get(k) or 0) for k in _CLAUDE_KEYS}
            model = msg.get("model")
            if model:
                models[model] = models.get(model, 0) + 1
    out = {k: sum(u[k] for u in by_id.values()) for k in _CLAUDE_KEYS}
    out["requests"] = len(by_id)
    out["model"] = max(models, key=models.get) if models else None
    out["files"] = len([f for f in files if os.path.exists(f)])
    return out


def claude_tool_usage(transcript_path, include_subagents=True):
    """{tool: calls} over a session and, by default, its subagents -- a tool a subagent used is
    in use, and denying it would take it from the subagent too."""
    from .calls import tool_usage
    counts = {}
    files = [transcript_path] + (claude_subagent_files(transcript_path)
                                 if include_subagents else [])
    for path in files:
        for name, n in tool_usage(path).items():
            counts[name] = counts.get(name, 0) + n
    return counts


# ---- the history Claude Code keeps of earlier sessions --------------------------------------
def claude_projects_dir(env=None):
    """`<config dir>/projects`, where Claude Code files every session's log by the directory it
    started in: `CLAUDE_CONFIG_DIR` when set (in `env`, else this process's environment), else
    `~/.claude`."""
    env = os.environ if env is None else env
    base = env.get("CLAUDE_CONFIG_DIR") or os.environ.get("CLAUDE_CONFIG_DIR")
    if not base:
        home = env.get("HOME") or os.path.expanduser("~")
        base = os.path.join(home, ".claude")
    return os.path.join(base, "projects")


def claude_history(env=None):
    """Every session log Claude Code still keeps: `projects/<folder>/<session id>.jsonl`. A
    session's subagents write under `<folder>/<session id>/subagents/` and stay with it."""
    root = claude_projects_dir(env)
    try:
        folders = sorted(os.listdir(root))
    except OSError:
        return []
    out = []
    for folder in folders:
        d = os.path.join(root, folder)
        try:
            names = sorted(os.listdir(d))
        except OSError:
            continue
        out.extend(os.path.join(d, n) for n in names
                   if n.endswith(".jsonl") and os.path.isfile(os.path.join(d, n)))
    return out


def project_folder_name(path):
    """The folder name Claude Code files a directory's sessions under: every character that is
    not a letter or a digit becomes `-` (`/Users/me/a.b` -> `-Users-me-a-b`)."""
    return re.sub(r"[^A-Za-z0-9]", "-", path or "")


def _epoch(stamp):
    if not isinstance(stamp, str) or not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def claude_session_meta(transcript_path, head_bytes=1 << 20):
    """{cwd, started, entrypoint} of a session log, from its first entries that name them:
    the directory it started in, when (epoch seconds) and from which front end (`cli`, or
    `sdk-*` for `claude -p` and the SDKs). Any of them may be None."""
    meta: dict = {"cwd": None, "started": None, "entrypoint": None}
    read = 0
    try:
        with open(transcript_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                read += len(line)
                if read > head_bytes:
                    break
                if '"cwd"' not in line and '"timestamp"' not in line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(ev, dict):
                    continue
                if meta["cwd"] is None and isinstance(ev.get("cwd"), str) and ev["cwd"]:
                    meta["cwd"] = ev["cwd"]
                if meta["started"] is None:
                    meta["started"] = _epoch(ev.get("timestamp"))
                if meta["entrypoint"] is None and isinstance(ev.get("entrypoint"), str):
                    meta["entrypoint"] = ev["entrypoint"]
                if all(v is not None for v in meta.values()):
                    break
    except OSError:
        pass
    return meta


_EXIT_RE = re.compile(r"(?im)^\s*(?:exit code|process exited with code)[:\s]+(-?\d+)")
_RESULT_CAP = 20000


def claude_turns(transcript_path):
    """The session's prompt-turns, as the loop guard's hooks would have seen them live:
    [(calls, ended_normally)], each call `(name, args, result_text, exit_code_or_None)`.

    A turn runs from one prompt the person gave (`_is_prompt`) to the next; a turn the person
    interrupted (`[Request interrupted ...`) did not end normally, nor did the session's last
    turn when its log stops on a tool call with no result. Subagents' calls are left out, as
    the live guard leaves them out. A failed call's exit code is read from its result text, the
    way the PostToolUseFailure hook reports it; a successful call has none."""
    turns = []
    cur = None                          # {"order": [ids], "calls": {id: ...}, "ok": bool}
    seen = set()

    def close():
        if cur is not None and cur["order"]:
            calls = []
            for tid in cur["order"]:
                c = cur["calls"][tid]
                if c.get("result") is None:
                    cur["ok"] = False   # a call that never returned: the turn was cut off
                    continue
                calls.append((c["name"], c["args"], c["result"], c.get("code")))
            turns.append((calls, cur["ok"]))

    for line in _lines(transcript_path):
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict) or ev.get("type") not in CONVERSATION \
                or ev.get("isSidechain"):
            continue
        if _is_prompt(ev):
            close()
            cur = {"order": [], "calls": {}, "ok": True}
            continue
        content = (ev.get("message") or {}).get("content")
        if ev.get("type") == "user":
            text = content if isinstance(content, str) else " ".join(
                b.get("text") or "" for b in content or () if isinstance(b, dict)
                and b.get("type") == "text")
            if isinstance(text, str) and text.startswith("[Request interrupted") \
                    and cur is not None:
                cur["ok"] = False
        if cur is None or not isinstance(content, list):
            continue
        for b in content:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_use":
                tid = b.get("id")
                if not tid or tid in seen:
                    continue
                seen.add(tid)
                cur["order"].append(tid)
                cur["calls"][tid] = {"name": str(b.get("name") or ""),
                                     "args": b.get("input") if isinstance(b.get("input"), dict)
                                     else {}, "result": None}
            elif b.get("type") == "tool_result":
                c = cur["calls"].get(b.get("tool_use_id"))
                if c is None or c["result"] is not None:
                    continue
                res = b.get("content")
                if isinstance(res, list):
                    res = " ".join(x.get("text", "") for x in res if isinstance(x, dict))
                res = (res if isinstance(res, str) else "")[:_RESULT_CAP]
                c["result"] = res
                if b.get("is_error"):
                    m = _EXIT_RE.search(res[:400])
                    c["code"] = int(m.group(1)) if m else None
    close()
    return turns


def claude_deferred_tools(transcript_path):
    """Tool names the session listed as deferred (`deferred_tools_delta` attachments).

    Deferred tools cost one listing line each, not a schema, so they matter little to the bill;
    they are still part of what the session could call, so the pruning catalog includes them."""
    names = set()
    for line in _lines(transcript_path):
        if '"deferred_tools_delta"' not in line:
            continue
        try:
            att = json.loads(line).get("attachment") or {}
        except ValueError:
            continue
        names.update(n for n in (att.get("addedNames") or []) if isinstance(n, str))
        names.difference_update(n for n in (att.get("removedNames") or []) if isinstance(n, str))
    return names
