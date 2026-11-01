"""Learning from the sessions Claude Code ran before the plugin was installed (`backfill`).

Each test builds a Claude Code history (`<CLAUDE_CONFIG_DIR>/projects/<folder>/<id>.jsonl`)
and pins one property of the import: it runs once, at a repository's first session start,
oldest first, and only before anything was folded live; a session that ends while it runs is
folded after it; imported sessions are scored only against the tools their own logs show, make
no model call, never let `auto` write a deny list by themselves, feed the loop guard's envelope,
and are counted apart.
"""
import io
import json
import os
import threading
import time
from datetime import datetime, timezone

import pytest

from tokeneconomizer import trace
from tokeneconomizer.plugin import accounting, backfill, cli, denylist, fold, hook, sidecall
from tokeneconomizer.plugin.context import LiveContext
from tokeneconomizer.plugin.hosts import ClaudeCode
from tokeneconomizer.plugin.paths import repo_identity, repo_state_dir
from tokeneconomizer.plugin.store import Store

DAY = 86400.0


@pytest.fixture
def cc(tmp_path):
    return tmp_path / "claude-config"


@pytest.fixture
def env(tmp_path, cc):
    return {"TOKENECONOMIZER_HOME": str(tmp_path / "home"), "TE_FOLD_SYNC": "1",
            "TE_TOOLPRUNE_QUIET_MINUTES": "0", "TE_MICROSKILLS": "envfacts",
            "CLAUDE_CONFIG_DIR": str(cc),
            "GIT_CEILING_DIRECTORIES": str(tmp_path), "PATH": os.environ.get("PATH", "")}


def _git(path, remote):
    (path / ".git" / "info").mkdir(parents=True)
    (path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (path / ".git" / "config").write_text(f'[remote "origin"]\n\turl = {remote}\n')
    (path / "pyproject.toml").write_text("[project]\nname='w'\n")
    return str(path)


@pytest.fixture
def repo(tmp_path):
    return _git(tmp_path / "repo", "https://github.com/acme/widget.git")


def call(env, payload):
    out = io.StringIO()
    hook.main([], io.StringIO(json.dumps(payload)), out, env)
    return json.loads(out.getvalue().splitlines()[-1])


def store_for(env, repo):
    return Store(repo_state_dir(repo_identity(repo, env), "claude", env))


def history_session(cc, cwd, sid, started, prompts, deferred=(), entrypoint="cli",
                    folder=None, interrupted=()):
    """A Claude Code log as Claude Code files it: under the folder named after `cwd`, every
    entry naming the directory, the time and the front end; per prompt a user entry, then per
    call an assistant `tool_use` (with usage) and a user `tool_result`. `deferred` lists tools
    the session loaded but did not call; `interrupted` the prompts the person interrupted."""
    d = cc / "projects" / (folder or trace.project_folder_name(cwd))
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{sid}.jsonl"
    stamp = datetime.fromtimestamp(started, timezone.utc).isoformat().replace("+00:00", "Z")
    base = {"cwd": cwd, "timestamp": stamp, "entrypoint": entrypoint, "sessionId": sid}
    lines = [dict(base, type="attachment", attachment={
        "type": "deferred_tools_delta", "addedNames": list(deferred), "removedNames": []})]
    n = 0
    for k, (text, calls) in enumerate(prompts):
        lines.append(dict(base, type="user", message={"role": "user", "content": text}))
        for name, inp, result, is_error in calls:
            n += 1
            lines.append(dict(base, type="assistant", message={
                "id": f"msg_{sid}_{n}", "model": "claude-opus-5-5", "role": "assistant",
                "usage": {"input_tokens": 10, "cache_read_input_tokens": 1000,
                          "cache_creation_input_tokens": 100, "output_tokens": 50},
                "content": [{"type": "tool_use", "id": f"tu_{sid}_{n}", "name": name,
                             "input": inp}]}))
            lines.append(dict(base, type="user", message={"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": f"tu_{sid}_{n}", "content": result,
                 "is_error": is_error}]}))
        if k in interrupted:
            lines.append(dict(base, type="user", message={"role": "user", "content": [
                {"type": "text", "text": "[Request interrupted by user]"}]}))
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    os.utime(path, (started + 600, started + 600))
    return str(path)


def trap(repo, i):
    """A session that falls into the same environment trap: bare `pytest` is missing."""
    return [(f"fix bug {i}", [
        ("Edit", {"file_path": f"{repo}/pkg/mod{i}.py", "old_string": "a", "new_string": "b"},
         "ok", False),
        ("Bash", {"command": "pytest -q tests/test_a.py"}, "bash: pytest: command not found", True),
        ("Bash", {"command": "python -m pytest -q tests/test_a.py"}, "1 passed in 0.1s", False),
    ])]


def start(env, repo, sid, transcript):
    return call(env, {"hook_event_name": "SessionStart", "session_id": sid, "source": "startup",
                      "transcript_path": transcript, "cwd": repo})


def end(env, repo, sid, transcript):
    return call(env, {"hook_event_name": "SessionEnd", "session_id": sid, "reason": "other",
                      "transcript_path": transcript, "cwd": repo})


def live_transcript(tmp_path, name, prompts, deferred=()):
    """A live session's log (where Claude Code writes it does not matter to the hooks)."""
    path = tmp_path / f"{name}.jsonl"
    lines = [{"type": "attachment", "attachment": {"type": "deferred_tools_delta",
                                                   "addedNames": list(deferred)}}]
    for text, calls in prompts:
        lines.append({"type": "user", "message": {"role": "user", "content": text}})
        for j, (tool, inp, result, err) in enumerate(calls):
            lines.append({"type": "assistant", "message": {"id": f"m{name}{j}", "content": [
                {"type": "tool_use", "id": f"t{name}{j}", "name": tool, "input": inp}]}})
            lines.append({"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": f"t{name}{j}", "content": result,
                 "is_error": err}]}})
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return str(path)


# ---- the import itself --------------------------------------------------------------------------
def test_the_first_session_start_imports_earlier_history_oldest_first_and_only_once(
        env, repo, cc, tmp_path):
    now = time.time()
    other = _git(tmp_path / "other", "https://github.com/acme/gadget.git")
    for i, age in enumerate((9, 3, 6, 7)):       # written out of order
        history_session(cc, repo, f"h{i}", now - age * DAY, trap(repo, i))
    history_session(cc, other, "elsewhere", now - 5 * DAY, trap(other, 9))
    fresh = history_session(cc, repo, "fresh", now - 60, trap(repo, 8))
    os.utime(fresh, (now, now))                 # still being written: maybe still running
    history_session(cc, repo, "empty", now - 4 * DAY, [])
    history_session(cc, repo, "sdk", now - 4 * DAY, trap(repo, 7), entrypoint="sdk-cli")

    first = live_transcript(tmp_path, "live1", [("hello", [])])
    out = start(env, repo, "live1", first)
    st = store_for(env, repo)
    folded = st.read("folded.json")
    assert sorted(folded) == ["h0", "h1", "h2", "h3"]
    assert all(v["imported"] for v in folded.values())
    ledger = st.read_lines("ledger.jsonl")
    assert [r["session_id"] for r in ledger] == ["h0", "h3", "h2", "h1"]   # 9, 7, 6, 3 days ago
    assert all(r["imported"] and r["untool_applied"] == [] for r in ledger)
    state = st.read("backfill.json")
    assert state["state"] == "done" and state["folded"] == 4 and "plan" not in state
    assert state["skipped"] == {"no prompt": 1, "one-prompt SDK run": 1,
                                "written in the last 10 minutes": 1}
    assert "learned from" not in out.get("systemMessage", "")

    # The next start has the fact the earlier sessions kept meeting, says where it came from,
    # once, and imports nothing more.
    nxt = start(env, repo, "live2", live_transcript(tmp_path, "live2", [("hi", [])]))
    assert ("`pytest …` fails here with command not found; `python -m pytest …` works."
            in nxt["hookSpecificOutput"]["additionalContext"])
    assert "learned from 4 earlier claude session(s)" in nxt["systemMessage"]
    again = start(env, repo, "live3", live_transcript(tmp_path, "live3", [("hi", [])]))
    assert "learned from" not in again.get("systemMessage", "")
    assert sorted(st.read("folded.json")) == ["h0", "h1", "h2", "h3"]


def test_a_repository_that_already_folded_a_live_session_is_not_imported(
        env, repo, cc, tmp_path, monkeypatch, capsys):
    """An upgrade: the plugin already learned from a live session here. Importing older ones
    after it would hand back facts newer sessions no longer support."""
    now = time.time()
    history_session(cc, repo, "old", now - 5 * DAY, trap(repo, 1))
    off = dict(env, TE_BACKFILL="off")
    t = live_transcript(tmp_path, "l1", [("hi", [])])
    start(off, repo, "l1", t)
    end(off, repo, "l1", t)
    start(env, repo, "l2", live_transcript(tmp_path, "l2", [("hi", [])]))
    st = store_for(env, repo)
    assert sorted(st.read("folded.json")) == ["l1"]
    state = st.read("backfill.json")
    assert state["state"] == "skipped" and "already folded live" in state["reason"]

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert cli.main(["--cwd", repo, "--host", "claude", "backfill"]) == 1
    assert "not run: 1 session(s) here were already folded live" in capsys.readouterr().out


def test_a_session_that_ends_during_an_import_is_folded_after_it(env, repo, cc, tmp_path,
                                                                  monkeypatch):
    """The import holds the fold lock for its whole run: a live fold gives up waiting, its
    session stays pending, and the next session start folds it, after the imported ones."""
    now = time.time()
    for i in range(3):
        history_session(cc, repo, f"h{i}", now - (5 - i) * DAY, trap(repo, i))
    ctx = LiveContext.build(ClaudeCode(), repo, env)
    started, release = threading.Event(), threading.Event()
    real = fold.fold_locked

    def slow(*a, **k):
        started.set()
        release.wait(5)
        return real(*a, **k)
    monkeypatch.setattr(fold, "fold_locked", slow)
    monkeypatch.setattr(fold, "FOLD_LOCK_WAIT_S", 0.2)
    th = threading.Thread(target=backfill.run, args=(ctx,))
    th.start()
    assert started.wait(5)
    t = live_transcript(tmp_path, "live", [("hi", [("Bash", {"command": "ls"}, "a", False)])])
    off = dict(env, TE_BACKFILL="off")        # this start must not run the import itself
    start(off, repo, "live", t)
    end(off, repo, "live", t)                  # its fold cannot take the lock
    st = store_for(env, repo)
    assert "live" not in (st.read("folded.json") or {})
    release.set()
    th.join(10)
    monkeypatch.setattr(fold, "fold_locked", real)
    assert fold.pending_sessions(ctx) == ["live"]
    start(env, repo, "next", live_transcript(tmp_path, "next", [("hi", [])]))
    assert [r["session_id"] for r in st.read_lines("ledger.jsonl")] == ["h0", "h1", "h2", "live"]


def test_a_run_that_died_resumes_its_saved_plan_in_order(env, repo, cc):
    now = time.time()
    paths = [history_session(cc, repo, f"h{i}", now - (5 - i) * DAY, trap(repo, i))
             for i in range(3)]
    ctx = LiveContext.build(ClaudeCode(), repo, env)
    items, _ = backfill.plan(ctx)
    assert [it["session_id"] for it in items] == ["h0", "h1", "h2"]
    fold.fold_locked(ctx, "h0", paths[0], imported={"started": items[0]["started"]})
    ctx.store.write("backfill.json", {"state": "running", "plan": items, "planned": 3,
                                      "folded": 1, "updated": now - 3600})
    # A session older than every planned one turns up meanwhile: not in this run's plan.
    history_session(cc, repo, "older", now - 9 * DAY, trap(repo, 9))
    assert backfill.start(ctx) is True
    st = ctx.store
    assert [r["session_id"] for r in st.read_lines("ledger.jsonl")] == ["h0", "h1", "h2"]
    assert st.read("backfill.json")["state"] == "done"
    assert st.read("backfill.json")["folded"] == 3


# ---- what imported sessions may change ---------------------------------------------------------
def _catalog(env, tools):
    d = os.path.join(env["TOKENECONOMIZER_HOME"], "hosts", "claude")
    os.makedirs(d, exist_ok=True)
    json.dump({"tools": tools}, open(os.path.join(d, "catalog.json"), "w"))


def _look(repo):
    return [("look", [("Read", {"file_path": f"{repo}/a.py"}, "x", False),
                      ("Bash", {"command": "ls"}, "a.py", False)])]


def test_imported_sessions_are_scored_only_against_the_tools_their_logs_show(
        env, repo, cc, tmp_path):
    """Today's probed catalog would count a tool installed last week as unused all through
    sessions that never had it."""
    _catalog(env, ["Bash", "Read", "Edit", "Workflow", "mcp__new__tool"])
    now = time.time()
    for i in range(6):
        history_session(cc, repo, f"h{i}", now - (9 - i) * DAY, _look(repo),
                        deferred=["mcp__old__tool"])
    start(env, repo, "live", live_transcript(tmp_path, "live", [("hi", [])]))
    unt = store_for(env, repo).read("untool.json")
    assert unt["imported_sessions"] == 6 and unt["sessions"] == 6
    assert unt["streaks"]["mcp__old__tool"] == 6
    assert "Workflow" not in unt["streaks"] and "mcp__new__tool" not in unt["streaks"]


def test_auto_never_writes_a_deny_list_from_history_alone(env, repo, cc, tmp_path):
    """Six imported sessions clear the evidence many times over; under the default `auto` the
    proposal still waits for `toolprune apply` until enough live sessions back it."""
    now = time.time()
    for i in range(6):
        history_session(cc, repo, f"h{i}", now - (9 - i) * DAY, _look(repo),
                        deferred=["mcp__old__tool"])

    def live(i):
        t = live_transcript(tmp_path, f"l{i}", _look(repo), deferred=["mcp__old__tool"])
        out = start(env, repo, f"l{i}", t)
        end(env, repo, f"l{i}", t)
        return out
    live(0)                                       # runs the import; ends as one live session
    st = store_for(env, repo)
    unt = st.read("untool.json")
    assert unt["proposal"] == ["mcp__old__tool"]
    assert unt["history_proposal"] == ["mcp__old__tool"] and not unt.get("auto_pending")
    assert not os.path.exists(denylist.settings_path(repo))
    out = live(1)
    assert "1 tool(s) went unused" in out["systemMessage"]
    assert "nothing changes until you run `/tokeneconomizer:toolprune apply`" in out["systemMessage"]
    for i in range(2, 4):
        live(i)
        assert not os.path.exists(denylist.settings_path(repo))
    live(4)              # five live sessions unused: the live evidence alone clears the bar
    deny = json.load(open(denylist.settings_path(repo)))["permissions"]["deny"]
    assert deny == ["mcp__old__tool"]


def test_a_tool_used_live_drops_its_imported_streak(env, repo, cc, tmp_path):
    now = time.time()
    for i in range(6):
        history_session(cc, repo, f"h{i}", now - (9 - i) * DAY, _look(repo),
                        deferred=["mcp__old__tool"])
    t = live_transcript(tmp_path, "l0", [("use it", [("mcp__old__tool", {}, "ok", False)])],
                        deferred=["mcp__old__tool"])
    start(env, repo, "l0", t)
    end(env, repo, "l0", t)
    unt = store_for(env, repo).read("untool.json")
    assert unt["streaks"]["mcp__old__tool"] == 0 and unt["proposal"] == []
    assert "mcp__old__tool" not in (unt.get("history_streaks") or {})


def test_an_import_makes_no_model_call(env, repo, cc, tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(sidecall, "runner_for", lambda *a, **k: asked.append(a) or None)
    env = dict(env, TE_MICROSKILLS="on", TE_MICROSKILLS_PROPOSE_EVERY="1")
    now = time.time()
    for i in range(4):
        history_session(cc, repo, f"h{i}", now - (9 - i) * DAY, trap(repo, i))
    start(env, repo, "live", live_transcript(tmp_path, "live", [("hi", [])]))
    assert len(store_for(env, repo).read("folded.json")) == 4
    assert asked == []


def test_imported_turns_feed_the_loop_guards_envelope(env, repo, cc, tmp_path):
    now = time.time()
    green = ("go", [("Edit", {"file_path": f"{repo}/a.py"}, "ok", False),
                    ("Bash", {"command": "pytest -q"}, "3 passed", False)])
    red = ("again", [("Edit", {"file_path": f"{repo}/a.py"}, "ok", False),
                     ("Bash", {"command": "pytest -q"},
                      "Exit code 1\nFAILED tests/test_a.py::t - AssertionError", True)])
    history_session(cc, repo, "h0", now - 3 * DAY, [green, red, green, green],
                    interrupted=(2,))
    start(env, repo, "live", live_transcript(tmp_path, "live", [("hi", [])]))
    st = store_for(env, repo)
    # the red turn did not end green, the interrupted one did not end at all
    assert st.read("envelope.json")["resolved_turns"] == [2, 2]
    assert st.read_lines("ledger.jsonl")[0]["turns"] == 4


def test_status_counts_imported_sessions_apart(env, repo, cc, tmp_path, monkeypatch, capsys):
    now = time.time()
    for i in range(2):
        history_session(cc, repo, f"h{i}", now - (5 - i) * DAY, trap(repo, i))
    t = live_transcript(tmp_path, "live", [("hi", [("Bash", {"command": "ls"}, "a", False)])])
    start(env, repo, "live", t)
    end(env, repo, "live", t)
    summary = accounting.summarize(store_for(env, repo).read_lines("ledger.jsonl"))
    assert summary["sessions"] == 1 and summary["imported_sessions"] == 2
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cli.main(["--cwd", repo, "--host", "claude", "status"])
    out = capsys.readouterr().out
    assert "  sessions     1 folded" in out
    assert "2 session(s) from before the plugin imported" in out


# ---- which logs belong to the repository -------------------------------------------------------
def test_logs_of_a_deleted_worktree_or_a_moved_checkout_still_belong(env, repo, cc, tmp_path):
    now = time.time()
    gone_wt = os.path.join(repo, ".claude", "worktrees", "feature")      # deleted since
    history_session(cc, gone_wt, "wt", now - 4 * DAY, trap(gone_wt, 1))
    # A checkout that moved: its log names the old path, filed under the current folder.
    history_session(cc, "/nowhere/old/repo", "moved", now - 5 * DAY, trap(repo, 2),
                    folder=trace.project_folder_name(repo))
    # A gone directory filed elsewhere is not this repository's.
    history_session(cc, "/nowhere/else", "stranger", now - 5 * DAY, trap(repo, 3))
    ctx = LiveContext.build(ClaudeCode(), repo, env)
    items, _ = backfill.plan(ctx)
    assert [it["session_id"] for it in items] == ["moved", "wt"]


def test_the_newest_sessions_are_kept_when_there_are_more_than_the_cap(env, repo, cc):
    now = time.time()
    for i in range(5):
        history_session(cc, repo, f"h{i}", now - (9 - i) * DAY, trap(repo, i))
    ctx = LiveContext.build(ClaudeCode(), repo, dict(env, TE_BACKFILL_MAX_SESSIONS="2"))
    items, skipped = backfill.plan(ctx)
    assert [it["session_id"] for it in items] == ["h3", "h4"]
    assert skipped == {"older than the newest 2": 3}


def test_claude_turns_reads_prompt_turns_as_the_hooks_saw_them(cc, repo):
    path = history_session(cc, repo, "t", time.time() - DAY, [
        ("one", [("Bash", {"command": "ls"}, "a", False),
                 ("Bash", {"command": "false"}, "Exit code 2\nboom", True)]),
        ("two", [("Read", {"file_path": "x"}, "File does not exist.", True)]),
    ], interrupted=(1,))
    turns = trace.claude_turns(path)
    assert [(len(c), ok) for c, ok in turns] == [(2, True), (1, False)]
    assert turns[0][0][1] == ("Bash", {"command": "false"}, "Exit code 2\nboom", 2)
    assert turns[1][0][0][3] is None
    assert trace.claude_session_meta(path)["cwd"] == repo
