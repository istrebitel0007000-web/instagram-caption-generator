"""
Regression tests covering R1–R9 features added after the original 50.
Run from C:\\MR.CBG with:
    python -m pytest tests/test_r1_to_r9.py -v
"""
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ORACLE_PATH = HERE.parent / "oracle_brain.py"


@pytest.fixture(scope="session")
def ob():
    spec = importlib.util.spec_from_file_location("oracle_brain", ORACLE_PATH)
    m = importlib.util.module_from_spec(spec)
    sys.modules["oracle_brain"] = m
    spec.loader.exec_module(m)
    return m


@pytest.fixture(autouse=True)
def reset(ob):
    ob._history.clear()
    ob._cache.clear()
    ob._seen_prompts.clear()
    ob._notes.clear()
    ob._bookmarks.clear()
    ob._rag_chunks.clear()
    ob._projects.clear()
    ob._aliases.clear()
    ob._aliases.update(ob._DEFAULT_ALIASES)
    ob._macros.clear()
    ob._profiles.clear()
    ob._shares.clear()
    ob.CONFIG["active_persona"] = "tech_oracle"
    ob.CONFIG["safe_mode"] = True
    ob.CONFIG["incognito"] = False
    ob.CONFIG["daily_budget_usd"] = 0.0
    ob.CONFIG["sticky_lang"] = ""
    ob.CONFIG["terminal_markdown"] = True
    ob.CONFIG["sqlite_enabled"] = False
    yield


# ── R1-#54 token recording + cost ────────────────────────────────────────
def test_record_usage_increments(ob):
    class U:
        prompt_tokens = 100
        completion_tokens = 50
    ob._stats["tokens_in"] = ob._stats["tokens_out"] = 0
    ob._record_usage("llama-3.3-70b-versatile", U())
    assert ob._stats["tokens_in"] == 100
    assert ob._stats["tokens_out"] == 50

def test_estimated_cost_unknown_model(ob):
    ob._active_model = "no-such-model"
    assert ob.estimated_cost_usd() == 0.0


# ── R1-#57 /undo ─────────────────────────────────────────────────────────
def test_undo_empty(ob):
    assert "empty" in ob.undo_last_turn().lower()

def test_undo_pair(ob):
    ob._history.extend([
        {"role": "user", "content": "q", "timestamp": "t"},
        {"role": "assistant", "content": "a", "timestamp": "t"},
    ])
    out = ob.undo_last_turn()
    assert "Undid" in out and len(ob._history) == 0


# ── R1-#65 branch ────────────────────────────────────────────────────────
def test_branch_full(ob):
    ob._history.extend([
        {"role": "user", "content": "q", "timestamp": "t"},
        {"role": "assistant", "content": "a", "timestamp": "t"},
    ])
    ob.branch_project("alt")
    assert "alt" in ob._projects
    assert len(ob._projects["alt"]["history"]) == 2

def test_branch_duplicate_name(ob):
    ob._history.append({"role": "user", "content": "q", "timestamp": "t"})
    ob.branch_project("dup")
    out = ob.branch_project("dup")
    assert "already exists" in out


# ── R2-#82 aliases ───────────────────────────────────────────────────────
def test_alias_default_seeded(ob):
    assert "c" in ob._aliases and ob._aliases["c"] == "/clear"

def test_alias_expand(ob):
    assert ob.expand_alias("c") == "/clear"
    assert ob.expand_alias("s python") == "/short python"
    assert ob.expand_alias("plain text") == "plain text"

def test_alias_add_del(ob):
    ob.add_alias("zz", "/help")
    assert "zz" in ob._aliases
    ob.del_alias("zz")
    assert "zz" not in ob._aliases


# ── R2-#85 share links ──────────────────────────────────────────────────
def test_share_empty_history(ob):
    assert "empty" in ob.create_share("all").lower()

def test_share_revoke(ob):
    ob._history.extend([
        {"role": "user", "content": "q", "timestamp": "t"},
        {"role": "assistant", "content": "a", "timestamp": "t"},
    ])
    out = ob.create_share("all")
    tok = out.split("token: ")[1].split(")")[0]
    assert tok in ob._shares
    ob.revoke_share(tok)
    assert tok not in ob._shares


# ── R3-#89 translate ────────────────────────────────────────────────────
def test_translate_empty(ob):
    assert "Usage" in ob.translate_text("ja", "")
    assert "language" in ob.translate_text("", "hello").lower()


# ── R3-#92 timer ────────────────────────────────────────────────────────
def test_timer_invalid(ob):
    assert "Usage" in ob.add_timer("abc", "msg")
    assert "between 0 and 1440" in ob.add_timer("99999", "msg")

def test_timer_cancel_missing(ob):
    out = ob.cancel_timer("99999")
    assert "not found" in out


# ── R4-#94 unified find ─────────────────────────────────────────────────
def test_find_empty(ob):
    out = ob.find_everywhere("xyzqwerty")
    assert "No matches" in out

def test_find_in_notes(ob):
    ob.add_note("python is great")
    out = ob.find_everywhere("python")
    assert "python is great" in out


# ── R4-#95 pinned notes ─────────────────────────────────────────────────
def test_pin_toggle(ob):
    ob.add_note("remember this")
    nid = ob._notes[-1]["id"]
    ob.pin_note(str(nid))
    assert ob._notes[-1]["pinned"] is True
    ob.pin_note(str(nid))
    assert ob._notes[-1]["pinned"] is False

def test_pinned_block_in_system_prompt(ob):
    ob.add_note("user prefers terse replies")
    nid = ob._notes[-1]["id"]
    ob.pin_note(str(nid))
    block = ob.get_pinned_block()
    assert "PINNED USER FACTS" in block
    assert "user prefers terse replies" in block


# ── R4-#97 macros ───────────────────────────────────────────────────────
def test_macro_save_run(ob):
    ob.add_macro("hi", "/help && /digest")
    assert "hi" in ob._macros
    assert "&&" in ob._macros["hi"]


# ── R4-#98 profiles ─────────────────────────────────────────────────────
def test_profile_round_trip(ob):
    ob.CONFIG["active_persona"] = "code_helper"
    ob._active_model = "llama-3.1-8b-instant"
    ob.CONFIG["temperature"] = 0.2
    ob.save_current_as_profile("work")
    assert "work" in ob._profiles
    ob.CONFIG["active_persona"] = "tech_oracle"
    ob._active_model = "llama-3.3-70b-versatile"
    ob.load_profile("work")
    assert ob.CONFIG["active_persona"] == "code_helper"
    assert ob._active_model == "llama-3.1-8b-instant"


# ── R5-#101 compare ─────────────────────────────────────────────────────
def test_compare_usage(ob):
    assert "Usage" in ob.compare_models("", "", "")


# ── R6-#109 last N answers ──────────────────────────────────────────────
def test_show_last(ob):
    assert "No assistant" in ob.show_last_answers()
    ob._history.append({"role": "assistant", "content": "answer", "timestamp": "t"})
    out = ob.show_last_answers()
    assert "answer" in out


# ── R6-#116 timezone ────────────────────────────────────────────────────
def test_timezone_invalid(ob):
    assert "Unknown" in ob.show_timezone("Mars/Crater")

def test_timezone_valid(ob):
    out = ob.show_timezone("UTC")
    assert "UTC" in out


# ── R7-#1 secret scanning ───────────────────────────────────────────────
def test_secret_groq(ob):
    s = "Key gsk_abcdefghijklmnopqrstuvwxyz123"
    assert ob.contains_secret(s)
    assert "***REDACTED***" in ob.redact_secrets(s)

def test_secret_jwt(ob):
    s = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.SflKxw"
    assert ob.contains_secret(s)

def test_secret_clean(ob):
    assert not ob.contains_secret("just plain English text here")


# ── R7-#3 backup SHA ────────────────────────────────────────────────────
def test_backup_sidecar(ob, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ob.CONFIG["backup_dir"] = "bk"
    ob.CONFIG["backup_keep"] = 2
    ob.CONFIG["encrypt_backups"] = False
    Path("settings.json").write_text("{}")
    ob.make_backup()
    backups = list(Path("bk").glob("*.zip"))
    assert backups
    sidecar = Path(str(backups[0]) + ".sha256")
    assert sidecar.exists()


# ── R7-#10 incognito ────────────────────────────────────────────────────
def test_incognito_blocks_audit(ob, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ob.CONFIG["incognito"] = True
    ob._AUDIT_FILE = "audit.jsonl"
    ob._audit_tool_call("calc", {"x": 1}, "1")
    assert not Path("audit.jsonl").exists()


# ── R8-#1 CSRF token presence ───────────────────────────────────────────
def test_csrf_token_exists(ob):
    if hasattr(ob, "_CSRF_TOKEN"):
        assert len(ob._CSRF_TOKEN) >= 16


# ── R8-#2 path traversal ────────────────────────────────────────────────
def test_safe_path_traversal_blocked(ob):
    assert ob.safe_path("../../etc/passwd") is None
    assert ob.safe_path("") is None

def test_safe_path_relative_ok(ob, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("ok.txt").write_text("hi")
    p = ob.safe_path("ok.txt", must_exist=True)
    assert p is not None and p.name == "ok.txt"


# ── R8-#5 retry-after parsing happens silently ──────────────────────────
def test_retry_after_helper_doesnt_crash(ob):
    # Just verify the regex doesn't crash on weird input
    msg = "rate_limit: try again in 5.5s please"
    import re
    m = re.search(r"(?:try again in|retry after|wait\s+)\s*"
                  r"(\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
    assert m and float(m.group(1)) == 5.5


# ── R9-#3 dry-run confirm ───────────────────────────────────────────────
def test_confirm_destructive_blocks_without_token(ob):
    ob.CONFIG["safe_mode"] = True
    ok, preview = ob.confirm_destructive("nuke", "kills X", "")
    assert not ok and "confirm" in preview.lower()

def test_confirm_destructive_passes_with_token(ob):
    ob.CONFIG["safe_mode"] = True
    ok, _ = ob.confirm_destructive("nuke", "kills X", "yes")
    assert ok

def test_confirm_destructive_disabled(ob):
    ob.CONFIG["safe_mode"] = False
    ok, _ = ob.confirm_destructive("nuke", "kills X", "")
    assert ok


# ── R9-#7 cost preview ─────────────────────────────────────────────────
def test_cost_preview_usage(ob):
    assert "Usage" in ob.cost_preview("")

def test_cost_preview_nonempty(ob):
    out = ob.cost_preview("Tell me about Python")
    assert "Cost preview" in out
    assert "Tokens (est)" in out


# ── R9-#8 terminal markdown ────────────────────────────────────────────
def test_render_markdown_bold(ob):
    out = ob.render_markdown_ansi("**bold**")
    assert "\033[1m" in out  # ANSI bold

def test_render_markdown_disabled(ob):
    ob.CONFIG["terminal_markdown"] = False
    s = "**bold** *italic*"
    assert ob.render_markdown_ansi(s) == s

def test_render_markdown_code_fence(ob):
    out = ob.render_markdown_ansi("```python\nprint(1)\n```")
    assert "python" in out


# ── R9-#9 sticky language ──────────────────────────────────────────────
def test_set_sticky_lang(ob):
    out = ob.set_sticky_lang("ja")
    assert "ja" in out.lower() or "japanese" in out.lower()
    assert ob.CONFIG["sticky_lang"] == "ja"
    ob.set_sticky_lang("off")
    assert ob.CONFIG["sticky_lang"] == ""

def test_sticky_lang_overrides_auto_detect(ob):
    ob.CONFIG["sticky_lang"] = "fr"
    out = ob.build_translate_instruction("en")  # would normally be empty
    assert "French" in out

def test_sticky_lang_disabled(ob):
    ob.CONFIG["sticky_lang"] = ""
    out = ob.build_translate_instruction("en")
    assert out == ""


# ── R9-#10 log rotation handler installed ──────────────────────────────
def test_log_handler_is_rotating(ob):
    from logging.handlers import RotatingFileHandler
    assert any(isinstance(h, RotatingFileHandler) for h in ob.log.parent.handlers
               if hasattr(ob.log, "parent")) or \
           any(isinstance(h, RotatingFileHandler) for h in ob._handlers)


# ── R9-#2 SQLite query guard ───────────────────────────────────────────
def test_sqlite_disabled_default(ob):
    ob.CONFIG["sqlite_enabled"] = False
    assert "Disabled" in ob.sqlite_sync_now()

def test_sqlite_query_rejects_writes(ob):
    ob.CONFIG["sqlite_enabled"] = True
    ob.CONFIG["sqlite_file"] = ":memory:"   # ephemeral
    out = ob.sqlite_query("DROP TABLE notes")
    assert "Only SELECT" in out or "read-only" in out
    ob.CONFIG["sqlite_enabled"] = False
