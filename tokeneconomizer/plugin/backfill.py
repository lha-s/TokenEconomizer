"""Learning from the sessions an agent ran before the plugin was installed.

The plugin learns from each session it sees end, so a new install starts from nothing while
Claude Code already keeps a log of every earlier session under `~/.claude/projects/` (for as
long as its `cleanupPeriodDays` keeps them; 30 days by default). The first session start in a
repository hands that repository's earlier logs to a detached `te backfill`, which folds them
through the same `fold.fold_locked` a finished session takes, oldest first, and then live
learning carries on from there.

Folding treats each session as the newest so far -- the fact lifecycle confirms and retires on
recency, tool pruning counts unbroken streaks -- so the import keeps time order:

  * it runs only in a repository with nothing folded yet; one where the plugin already folded a
    session records that it skipped the import (`backfill.json`), and `te backfill` refuses
    there with the reason;
  * it holds the store's "fold" lock for the whole run, so a session that ends meanwhile is not
    folded between two imported ones: its fold gives up waiting, the session stays pending, and
    the next session start folds it, after the import;
  * the plan is saved before the first fold, so a run that dies resumes the same list in order
    at the next session start (and only if no live fold got in meanwhile).

What an imported session can change is bounded (see `fold.fold_locked`): no model call, tools
scored only against what its own log shows, no deny list held, written or restored -- under
`toolprune=auto` a proposal that rests on imported sessions waits for `toolprune apply` -- and
its ledger record, `folded.json` entry and `te status` line say `imported`.

Which logs belong here: those whose starting directory resolves to this repository (any clone
or worktree of it); for a directory that is gone, its nearest surviving parent when that is a
checkout of this repository (a deleted worktree inside it), or a project folder named exactly as
Claude Code names this checkout's. Left out: sessions the plugin recorded or folded, logs
written in the last `QUIET_S` (possibly still running), logs over `MAX_BYTES`, sessions with no
prompt, and one-prompt SDK runs (`claude -p`, scripts, probes). The newest
`backfill_max_sessions` remain (0: all).
"""
import os
import time
import traceback

from . import fold
from .paths import BY_PATH, BY_REMOTE, repo_identity, state_root

STATE = "backfill.json"
QUIET_S = fold.STALE_S          # a log written this recently may belong to a running session
MAX_BYTES = 256 * 1024 * 1024
RESPAWN_S = 600                 # a run that updated its state this recently is still going


def _record_error(env):
    try:
        root = state_root(env)
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "errors.log"), "a", encoding="utf-8") as fh:
            fh.write(f"--- {time.strftime('%Y-%m-%dT%H:%M:%S')} pid={os.getpid()} (backfill)\n")
            fh.write(traceback.format_exc())
    except Exception:
        pass


def live_folds(store):
    """Session ids this store folded from a live session (not imported)."""
    return [sid for sid, v in (store.read("folded.json") or {}).items()
            if not (isinstance(v, dict) and v.get("imported"))]


def status(store):
    return store.read(STATE) or {}


def _belongs(ctx, cwd, folder, cache):
    """Did a session that started in `cwd`, filed under project folder `folder`, run in this
    repository?"""
    from .. import trace
    if cwd:
        probe, gone = cwd, False
        while probe and not os.path.isdir(probe):
            parent = os.path.dirname(probe)
            probe, gone = (None if parent == probe else parent), True
        if probe:
            if probe not in cache:
                cache[probe] = repo_identity(probe, ctx.env)
            repo = cache[probe]
            if repo.key == ctx.repo.key and (not gone or repo.how in (BY_REMOTE, BY_PATH)):
                return True
            if not gone:
                return False
    return folder == trace.project_folder_name(ctx.repo.root)


def plan(ctx, now=None):
    """(items, skipped): this repository's earlier sessions to import, oldest first, each
    {session_id, transcript, started, ended, prompts}; and how many were left out, by reason.
    Reads logs, writes nothing."""
    now = now or time.time()
    store, host = ctx.store, ctx.host
    folded = store.read("folded.json") or {}
    skipped, items, cache = {}, [], {}

    def skip(why):
        skipped[why] = skipped.get(why, 0) + 1
    for path in host.history(ctx.env):
        sid = os.path.splitext(os.path.basename(path))[0]
        try:
            st = os.stat(path)
        except OSError:
            continue
        meta = host.session_meta(path)
        if not _belongs(ctx, meta.get("cwd"), os.path.basename(os.path.dirname(path)), cache):
            continue
        if sid in folded or os.path.exists(store.path("sessions", f"{sid}.json")):
            skip("seen by the plugin")
            continue
        if now - st.st_mtime < QUIET_S:
            skip("written in the last 10 minutes")
            continue
        if st.st_size > MAX_BYTES:
            skip("larger than 256 MB")
            continue
        try:
            prompts = host.prompt_count(path)
        except (OSError, ValueError):
            prompts = None
        if not prompts:
            skip("no prompt")
            continue
        if str(meta.get("entrypoint") or "").startswith("sdk") and prompts <= 1:
            skip("one-prompt SDK run")
            continue
        items.append({"session_id": sid, "transcript": path, "prompts": prompts,
                      "started": meta.get("started") or st.st_mtime, "ended": st.st_mtime})
    items.sort(key=lambda it: (float(it["started"]), it["session_id"]))
    cap = int(ctx.config.backfill_max_sessions or 0)
    if cap and len(items) > cap:
        skipped[f"older than the newest {cap}"] = len(items) - cap
        items = items[-cap:]
    return items, skipped


def _refusal(n):
    return (f"{n} session(s) here were already folded live, and folding older ones after them "
            f"would undo what the newer ones taught; earlier history is imported only before "
            f"the first fold")


def run(ctx, now=None):
    """Import this repository's earlier sessions, holding the fold lock throughout. Returns the
    state document (`backfill.json`); its `state` is `done`, `skipped` (a live fold came first),
    or `busy` (another fold or import holds the lock: nothing was written)."""
    store = ctx.store
    with store.locked("fold", timeout=fold.FOLD_LOCK_WAIT_S) as got:
        if not got:
            return {"state": "busy"}
        st = status(store)
        live = live_folds(store)
        if live:
            st = dict(st, state="skipped", reason=_refusal(len(live)), updated=time.time())
            st.pop("plan", None)
            store.write(STATE, st)
            return st
        if st.get("state") in ("done", "skipped"):
            return st
        if st.get("state") == "running" and st.get("plan") is not None:
            items = st["plan"]                       # a run that died: the same list, in order
        else:
            items, skipped = plan(ctx, now=now)
            st = {"state": "running", "plan": items, "planned": len(items), "skipped": skipped,
                  "started_at": time.time(), "folded": 0, "errors": 0}
        st["updated"] = time.time()
        store.write(STATE, st)
        done = set(store.read("folded.json") or {})
        for it in items:
            if it["session_id"] in done:
                continue
            try:
                rec = fold.fold_locked(ctx, it["session_id"], it["transcript"],
                                       imported={"started": it.get("started"),
                                                 "ended": it.get("ended")})
                st["folded"] = int(st.get("folded") or 0) + (rec is not None)
            except Exception:
                st["errors"] = int(st.get("errors") or 0) + 1
                _record_error(ctx.env)
            st["updated"] = time.time()
            store.write(STATE, st)
        st.pop("plan", None)
        st.update(state="done", finished_at=time.time(), updated=time.time(),
                  first=items[0]["started"] if items else None,
                  last=items[-1]["started"] if items else None)
        store.write(STATE, st)
        return st


def start(ctx):
    """At a session start: launch the import once per repository and agent, detached (inline
    under `TE_FOLD_SYNC=1`). Cheap when there is nothing to do: one read of `backfill.json`."""
    if not ctx.config.backfill or not ctx.host.has_history:
        return False
    st = status(ctx.store)
    state = st.get("state")
    if state in ("done", "skipped"):
        return False
    if state in ("queued", "running") and time.time() - float(st.get("updated") or 0) < RESPAWN_S:
        return False
    if state is None and live_folds(ctx.store):
        # A repository the plugin already learned from: say why nothing is imported, once.
        ctx.store.write(STATE, {"state": "skipped", "reason": _refusal(len(live_folds(ctx.store))),
                                "updated": time.time()})
        return False
    ctx.store.write(STATE, dict(st, state=state if state == "running" else "queued",
                                updated=time.time()))
    if ctx.env.get("TE_FOLD_SYNC") == "1":
        run(ctx)
        return True
    return ctx.spawn(["backfill", "--quiet"], "backfill.log")


def _day(ts):
    return time.strftime("%Y-%m-%d", time.localtime(float(ts))) if ts else "?"


def notice(ctx, command):
    """Once, after an import that folded anything: a sentence saying so (else None)."""
    st = status(ctx.store)
    if st.get("state") != "done" or st.get("noticed") or not st.get("folded"):
        return None
    st["noticed"] = True
    ctx.store.write(STATE, st)
    return (f"TokenEconomizer learned from {st['folded']} earlier {ctx.host.name} session(s) in "
            f"{ctx.repo.label} that ran before it was installed ({_day(st.get('first'))} to "
            f"{_day(st.get('last'))}); `{command}` counts them apart.")
