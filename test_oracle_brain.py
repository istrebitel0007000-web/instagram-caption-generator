"""
pytest suite for Oracle Brain v5.

Run from C:\\MR.CBG with:
    pip install pytest
    python -m pytest tests/ -v

These tests use a mocked Groq client so no real API key is needed.
Tests that require a microphone, network, or pyttsx3 are skipped if
those subsystems aren't available.
"""
import importlib.util
import os
import shutil
import sys
import tempfile
import json
from pathlib import Path

import pytest


# ── Module loading ────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
ORACLE_PATH = PROJECT_ROOT / "oracle_brain.py"


@pytest.fixture(scope="session")
def ob():
    """Load the oracle_brain module once per test session."""
    spec = importlib.util.spec_from_file_location("oracle_brain", ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["oracle_brain"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_groq(ob):
    """Replace Groq client with a fake that records calls and returns canned data."""
    class FakeChunk:
        def __init__(self, t):
            class D: content = t
            class C: delta = D()
            self.choices = [C()]
            self.usage = None

    class FinalChunk:
        def __init__(self):
            class U:
                prompt_tokens = 10
                completion_tokens = 5
            self.choices = []
            self.usage = U()

    class FakeStream:
        def __init__(self, words=None):
            self.words = words or ["mock", " stream", " reply"]
        def __iter__(self):
            for w in self.words:
                yield FakeChunk(w)
            yield FinalChunk()

    class FakeNonStream:
        class Choice:
            class M:
                content = "mock non-stream reply"
            message = M()
        class U:
            prompt_tokens = 50
            completion_tokens = 10
        choices = [Choice()]
        usage = U()

    captured = {"calls": []}

    class Completions:
        def create(self, **kw):
            captured["calls"].append(kw)
            if kw.get("stream"):
                return FakeStream()
            return FakeNonStream

    class Chat:
        completions = Completions()

    class FakeGroq:
        def __init__(self, api_key=None):
            pass
        chat = Chat()

    orig_groq = ob.Groq
    orig_keys = list(ob.API_KEYS)
    ob.Groq = FakeGroq
    ob.API_KEYS = ["fake_key_xxxxxx"]
    yield captured
    ob.Groq = orig_groq
    ob.API_KEYS = orig_keys


@pytest.fixture(autouse=True)
def reset_state(ob):
    """Reset mutable globals between tests."""
    ob._history.clear()
    ob._cache.clear()
    ob._seen_prompts.clear()
    ob._answer_history.clear()
    ob._notes.clear()
    ob._bookmarks.clear()
    ob._projects.clear()
    ob._rag_chunks.clear()
    ob._daily_log.clear()
    ob._session_topics.clear()
    ob._stats["total_requests"] = 0
    ob._stats["cache_hits"] = 0
    ob._stats["duplicate_hits"] = 0
    ob._stats["tokens_in"] = 0
    ob._stats["tokens_out"] = 0
    ob.CONFIG["active_persona"] = "tech_oracle"
    ob.CONFIG["active_project"] = ""
    ob.CONFIG["user_set_length"] = False
    ob.CONFIG["rag_enabled"] = False
    yield


# ── Pure helpers ──────────────────────────────────────────────────────────────
class TestPureHelpers:
    def test_validate_input_empty(self, ob):
        assert "empty" in ob.validate_input("").lower()

    def test_validate_input_too_long(self, ob):
        assert "too long" in ob.validate_input("x" * 10000).lower()

    def test_validate_input_ok(self, ob):
        assert ob.validate_input("hello world") is None

    def test_detect_language_russian(self, ob):
        assert ob.detect_language("Привет, как дела?") == "ru"

    def test_detect_language_english(self, ob):
        assert ob.detect_language("Hello world") == "en"

    def test_detect_language_chinese(self, ob):
        assert ob.detect_language("你好世界") == "zh"

    def test_detect_mood_frustrated(self, ob):
        assert ob.detect_mood("ugh, this doesn't work") == "frustrated"

    def test_detect_mood_neutral_question(self, ob):
        # Regression: "What is X?" used to wrongly trigger 'confused'
        assert ob.detect_mood("What is the time?") == "neutral"

    def test_tag_topics(self, ob):
        topics = ob.tag_topics("How do I write a Python pandas script?")
        assert "python" in topics

    def test_optimize_prompt_strips_filler(self, ob):
        out = ob.optimize_prompt("Can you please explain how Python works?")
        assert "please" not in out.lower()
        assert "can you" not in out.lower()

    def test_validate_response_too_short(self, ob):
        assert ob.validate_response("q", "ok") is False

    def test_validate_response_refusal(self, ob):
        assert ob.validate_response("q", "I'm sorry, I cannot help.") is False

    def test_score_confidence_high(self, ob):
        score, label = ob.score_confidence(
            "The answer is 42. Run this command: ls -la"
        )
        assert score >= 8

    def test_score_confidence_low(self, ob):
        score, _ = ob.score_confidence(
            "Maybe it could be that, I am not sure"
        )
        assert score <= 4

    def test_needs_clarification_short_vague(self, ob):
        assert ob.needs_clarification("what?")
        assert ob.needs_clarification("hi")

    def test_needs_clarification_specific_short(self, ob):
        # Regression: "ls -la" used to trigger clarify; shouldn't
        assert not ob.needs_clarification("ls -la")
        assert not ob.needs_clarification("what is TCP?")


# ── Cache & dedup ─────────────────────────────────────────────────────────────
class TestCacheAndDedup:
    def test_cache_set_eviction(self, ob):
        ob._CACHE_MAX = 3
        for i in range(10):
            ob._cache_set(f"k{i}", f"v{i}")
        assert len(ob._cache) == 3
        assert "k9" in ob._cache
        assert "k0" not in ob._cache

    def test_register_seen_eviction(self, ob):
        ob._SEEN_MAX = 5
        for i in range(20):
            ob.register_seen_prompt(f"prompt {i}", f"answer {i}")
        assert len(ob._seen_prompts) <= 5

    def test_fuzzy_duplicate(self, ob):
        ob.register_seen_prompt("what is dns?", "DNS = Domain Name System")
        # Apostrophe + missing question mark — fuzzy should still match
        result = ob.check_duplicate("what's dns")
        assert result == "DNS = Domain Name System"

    def test_fuzzy_duplicate_no_false_positive(self, ob):
        ob.register_seen_prompt("what is dns?", "DNS = Domain Name System")
        assert ob.check_duplicate("how do I cook pasta?") is None


# ── Plugins ───────────────────────────────────────────────────────────────────
class TestPlugins:
    def test_plugin_calc_basic(self, ob):
        assert ob.run_plugin("calc", "2 + 3 * 4") == "14"

    def test_plugin_calc_blocks_unsafe(self, ob):
        result = ob.run_plugin("calc", "__import__('os').system('rm -rf /')")
        assert "Invalid" in result or "Calc error" in result

    def test_plugin_unknown(self, ob):
        result = ob.run_plugin("nonexistent", "")
        assert "not found" in result.lower()

    def test_register_with_schema_appears_in_tool_list(self, ob):
        ob.register_plugin(
            "test_tool", lambda x: f"got {x}",
            "test plugin",
            schema={"type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"]},
        )
        names = [t["function"]["name"] for t in ob._build_tool_definitions()]
        assert "plugin_test_tool" in names
        del ob._plugins["test_tool"]


# ── Code extractor ────────────────────────────────────────────────────────────
class TestCodeExtractor:
    def test_python_extension(self, ob, tmp_path):
        ob.CONFIG["code_output_dir"] = str(tmp_path)
        saved = ob.extract_and_save_code(
            "Here:\n```python\nprint(42)\n```", "test"
        )
        assert len(saved) == 1
        assert saved[0].endswith(".py")

    def test_typescript_extension(self, ob, tmp_path):
        ob.CONFIG["code_output_dir"] = str(tmp_path)
        saved = ob.extract_and_save_code(
            "```ts\nconst x: number = 1;\n```", "test"
        )
        assert len(saved) == 1
        assert saved[0].endswith(".ts")

    def test_no_blocks(self, ob, tmp_path):
        ob.CONFIG["code_output_dir"] = str(tmp_path)
        saved = ob.extract_and_save_code("plain text only", "x")
        assert saved == []


# ── ask_oracle (mocked Groq) ──────────────────────────────────────────────────
class TestAskOracle:
    def test_streaming_calls_on_token(self, ob, fake_groq):
        tokens = []
        result = ob.ask_oracle("Tell me about TCP networking",
                               silent=True,
                               on_token=lambda t: tokens.append(t))
        assert result.strip()
        assert len(tokens) >= 2

    def test_cache_hit_returns_same(self, ob, fake_groq):
        ob.ask_oracle("Explain DNS resolution in depth", silent=True)
        first_calls = len(fake_groq["calls"])
        ob.ask_oracle("Explain DNS resolution in depth", silent=True)
        # Either duplicate or cache short-circuited the second call
        assert len(fake_groq["calls"]) == first_calls
        assert (ob._stats["cache_hits"] + ob._stats["duplicate_hits"]) >= 1

    def test_input_validation_rejects_empty(self, ob, fake_groq):
        result = ob.ask_oracle("", silent=True)
        assert "INPUT ERROR" in result

    def test_smart_clarify_short_vague(self, ob, fake_groq):
        result = ob.ask_oracle("what?", silent=True)
        assert "brief" in result.lower() or "detail" in result.lower()

    def test_per_persona_temperature(self, ob, fake_groq):
        ob.CONFIG["active_persona"] = "code_helper"
        ob.ask_oracle("write a function", silent=True)
        last_call = fake_groq["calls"][-1]
        assert last_call.get("temperature") == ob.PERSONAS["code_helper"]["temperature"]

    def test_token_usage_recorded(self, ob, fake_groq):
        ob.ask_oracle("Tell me about Python pandas library", silent=True)
        assert ob._stats["tokens_in"] >= 1
        assert ob._stats["tokens_out"] >= 1

    def test_slash_help(self, ob, fake_groq):
        result = ob.ask_oracle("/help", silent=True)
        assert "Core Commands" in result

    def test_slash_stats(self, ob, fake_groq):
        result = ob.ask_oracle("/stats", silent=True)
        assert "Usage Stats" in result

    def test_slash_undo(self, ob, fake_groq):
        ob._history.extend([
            {"role": "user", "content": "q", "timestamp": "t"},
            {"role": "assistant", "content": "a", "timestamp": "t"},
        ])
        result = ob.ask_oracle("/undo", silent=True)
        assert "Undid" in result
        assert len(ob._history) == 0


# ── Notes / Bookmarks / Personas / Templates ──────────────────────────────────
class TestPersistedFeatures:
    def test_note_lifecycle(self, ob):
        msg = ob.add_note("buy milk")
        assert "saved" in msg.lower()
        assert "buy milk" in ob.list_notes()
        ob.clear_notes()
        assert "no notes" in ob.list_notes().lower()

    def test_bookmark_lifecycle(self, ob):
        ob._history.append({"role": "assistant", "content": "answer", "timestamp": "t"})
        msg = ob.bookmark_last("first")
        assert "first" in msg
        assert "first" in ob.list_bookmarks()

    def test_persona_add_delete(self, ob):
        ob.add_persona("test_p", "Test Persona", "instr")
        assert "test_p" in ob.PERSONAS
        ob.del_persona("test_p")
        assert "test_p" not in ob.PERSONAS

    def test_persona_cannot_delete_builtin(self, ob):
        result = ob.del_persona("tech_oracle")
        assert "cannot delete" in result.lower()


# ── RAG ───────────────────────────────────────────────────────────────────────
class TestRAG:
    def test_add_and_search(self, ob):
        ob.rag_add("Python is a programming language.", source="note")
        ob.rag_add("Tokyo is the capital of Japan.", source="note")
        hits = ob.rag_search("python language")
        assert len(hits) >= 1
        assert "Python" in hits[0][1]["text"]

    def test_search_empty_returns_nothing(self, ob):
        assert ob.rag_search("anything") == []

    def test_min_chunk_size(self, ob):
        result = ob.rag_add("hi")
        assert "too short" in result.lower()


# ── Diff viewer ───────────────────────────────────────────────────────────────
class TestDiff:
    def test_needs_two_answers(self, ob):
        result = ob.diff_answers("nonexistent")
        assert "at least 2" in result.lower()

    def test_unified_diff(self, ob):
        ob.record_answer("q", "Line A\nLine B\nLine C")
        ob.record_answer("q", "Line A\nLine B changed\nLine C")
        result = ob.diff_answers("q")
        assert "+Line B changed" in result
        assert "-Line B" in result
        assert "Similarity" in result


# ── Web search (mocked) ───────────────────────────────────────────────────────
class TestWebSearch:
    def test_empty_query(self, ob):
        assert "empty" in ob.web_search("").lower()


# ── Branching ─────────────────────────────────────────────────────────────────
class TestBranching:
    def test_branch_full_history(self, ob):
        ob._history.extend([
            {"role": "user", "content": "q1", "timestamp": "t"},
            {"role": "assistant", "content": "a1", "timestamp": "t"},
        ])
        result = ob.branch_project("alt")
        assert "alt" in result
        assert "alt" in ob._projects
        assert len(ob._projects["alt"]["history"]) == 2

    def test_branch_partial(self, ob):
        ob._history.extend([
            {"role": "user", "content": "q1", "timestamp": "t"},
            {"role": "assistant", "content": "a1", "timestamp": "t"},
            {"role": "user", "content": "q2", "timestamp": "t"},
            {"role": "assistant", "content": "a2", "timestamp": "t"},
        ])
        ob.branch_project("partial", from_index=3)
        assert len(ob._projects["partial"]["history"]) == 2
        assert ob._projects["partial"]["history"][0]["content"] == "q2"


# ── Exports ───────────────────────────────────────────────────────────────────
class TestExports:
    def test_anki_no_bookmarks(self, ob, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = ob.export_to_anki()
        assert "no bookmarks" in result.lower()

    def test_anki_with_bookmarks(self, ob, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ob._history.append({"role": "assistant",
                            "content": "TCP basics", "timestamp": "t"})
        ob.bookmark_last("TCP")
        result = ob.export_to_anki("cards.tsv")
        assert "exported" in result.lower()
        assert (tmp_path / "cards.tsv").exists()
        content = (tmp_path / "cards.tsv").read_text(encoding="utf-8")
        assert "TCP" in content

    def test_obsidian_export(self, ob, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ob._history.extend([
            {"role": "user", "content": "what is python?", "timestamp": "2026-04-24T10:00"},
            {"role": "assistant", "content": "high-level lang", "timestamp": "2026-04-24T10:00"},
        ])
        result = ob.export_to_obsidian("vault")
        assert "exported" in result.lower()
        vault = tmp_path / "vault"
        assert vault.exists()
        assert (vault / "_index.md").exists()
        notes = list(vault.glob("*.md"))
        # At least the Q/A note + index
        assert len(notes) >= 2
