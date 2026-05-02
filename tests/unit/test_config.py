"""Unit tests for MemoryConfig.

These tests neutralize load_dotenv() so the user's local .env doesn't bleed
into the test env.
"""
import pytest

from memoryos import MemoryConfig


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    """Stop from_env() from picking up the developer's local .env."""
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)


def test_defaults():
    cfg = MemoryConfig()
    assert cfg.graph_backend == "neo4j"
    assert cfg.llm_provider == "openai"
    assert cfg.extraction_model == "gpt-4o-mini"
    assert cfg.max_stm_messages == 200


def test_from_env_falls_back_to_defaults(monkeypatch):
    for k in (
        "LLM_PROVIDER", "GRAPH_BACKEND",
        "OPENAI_API_KEY", "GROQ_API_KEY",
        "REDIS_URL", "REDIS_HOST", "REDIS_PORT",
        "MEMORY_EXTRACTION_MODEL",
        "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD",
    ):
        monkeypatch.delenv(k, raising=False)
    cfg = MemoryConfig.from_env()
    assert cfg.llm_provider == "openai"
    assert cfg.graph_backend == "neo4j"
    assert cfg.extraction_model == "gpt-4o-mini"


def test_from_env_groq_provider(monkeypatch):
    monkeypatch.delenv("MEMORY_EXTRACTION_MODEL", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    cfg = MemoryConfig.from_env()
    assert cfg.llm_provider == "groq"
    assert cfg.groq_api_key == "gsk_test"
    assert "llama" in cfg.extraction_model


def test_from_env_memgraph(monkeypatch):
    monkeypatch.setenv("GRAPH_BACKEND", "memgraph")
    monkeypatch.setenv("MEMGRAPH_URI", "bolt://localhost:7688")
    cfg = MemoryConfig.from_env()
    assert cfg.graph_backend == "memgraph"
    assert cfg.graph_url == "bolt://localhost:7688"


def test_from_env_anthropic_provider(monkeypatch):
    monkeypatch.delenv("MEMORY_EXTRACTION_MODEL", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    cfg = MemoryConfig.from_env()
    assert cfg.llm_provider == "anthropic"
    assert cfg.anthropic_api_key == "sk-ant-test"
    assert "claude" in cfg.extraction_model


def test_custom_extraction_fn():
    fn = lambda s, u: "{}"
    cfg = MemoryConfig(extraction_fn=fn)
    assert cfg.extraction_fn is fn
