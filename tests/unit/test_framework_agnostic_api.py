"""Tests for framework-agnostic helpers (openai_tools, inject_context, tool_names)."""
from memoryos import MemoryOS, MemoryConfig


def _build_offline_memory():
    return MemoryOS(config=MemoryConfig(
        graph_password="x",
        extraction_fn=lambda s, u: "{}",
    ))


def test_tool_names():
    m = _build_offline_memory()
    names = m.tool_names()
    assert "remember_fact" in names
    assert "recall_facts" in names
    assert "forget_fact" in names
    assert "who_do_i_know" in names


def test_openai_tools_format():
    m = _build_offline_memory()
    schemas = m.openai_tools()
    assert len(schemas) == 4
    for schema in schemas:
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert schema["function"]["parameters"]["type"] == "object"


def test_openai_tools_no_user_id_exposed():
    """The LLM must never see user_id as a parameter."""
    m = _build_offline_memory()
    for schema in m.openai_tools():
        params = schema["function"]["parameters"].get("properties", {})
        assert "user_id" not in params, f"user_id leaked into {schema['function']['name']}"


def test_inject_context_empty_user(monkeypatch):
    """When the user has no graph data, inject_context returns the messages unchanged."""
    m = _build_offline_memory()
    monkeypatch.setattr(m.reader, "get_user_context", lambda uid: "")
    msgs = [{"role": "user", "content": "hi"}]
    out = m.inject_context(msgs, user_id="anyone")
    assert out == msgs


def test_inject_context_appends_to_existing_system(monkeypatch):
    m = _build_offline_memory()
    monkeypatch.setattr(m.reader, "get_user_context", lambda uid: "USER LIKES PYTHON")
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
    ]
    out = m.inject_context(msgs, user_id="u1")
    assert len(out) == 2  # didn't add a second system message
    assert out[0]["role"] == "system"
    assert "You are helpful." in out[0]["content"]
    assert "USER LIKES PYTHON" in out[0]["content"]


def test_inject_context_prepends_when_no_system(monkeypatch):
    m = _build_offline_memory()
    monkeypatch.setattr(m.reader, "get_user_context", lambda uid: "USER LIKES PYTHON")
    msgs = [{"role": "user", "content": "hi"}]
    out = m.inject_context(msgs, user_id="u1")
    assert len(out) == 2
    assert out[0]["role"] == "system"
    assert "USER LIKES PYTHON" in out[0]["content"]


def test_handle_tool_call_unknown_tool():
    import pytest
    m = _build_offline_memory()
    with pytest.raises(ValueError, match="Unknown memory tool"):
        m.handle_tool_call("nonexistent_tool", {}, user_id="u1")


def test_public_properties_exposed():
    """Make sure public properties on MemoryOS are accessible (no leading underscore)."""
    m = _build_offline_memory()
    assert m.connection is not None
    assert m.store is not None
    assert m.reader is not None
    assert m.writer is not None
    assert m.extractor is not None
    assert m.config is not None
