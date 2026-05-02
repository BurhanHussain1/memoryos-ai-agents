"""Extractor unit tests — uses custom_fn so no live LLM call is made."""
from memoryos import MemoryConfig
from memoryos.graph.extractor import MemoryExtractor


def test_custom_fn_provider_label():
    cfg = MemoryConfig(extraction_fn=lambda s, u: "{}")
    ex = MemoryExtractor(cfg)
    assert ex._provider == "custom"


def test_extract_empty_payload():
    cfg = MemoryConfig(extraction_fn=lambda s, u: '{"nodes": [], "edges": []}')
    ex = MemoryExtractor(cfg)
    payload = ex.extract("u1", [{"role": "user", "content": "hello"}])
    assert payload.user_id == "u1"
    assert payload.nodes == []
    assert payload.edges == []


def test_extract_strips_json_fences():
    cfg = MemoryConfig(extraction_fn=lambda s, u: '```json\n{"nodes":[],"edges":[]}\n```')
    ex = MemoryExtractor(cfg)
    payload = ex.extract("u1", [{"role": "user", "content": "hi"}])
    assert payload.user_id == "u1"


def test_extract_handles_invalid_json():
    cfg = MemoryConfig(extraction_fn=lambda s, u: "not json at all")
    ex = MemoryExtractor(cfg)
    payload = ex.extract("u1", [{"role": "user", "content": "hi"}])
    assert payload.nodes == []
    assert payload.edges == []


def test_extract_real_payload():
    json_resp = """{
        "nodes": [{"type": "Fact", "key": {"fact_id": "f1"}, "properties": {"content": "loves Python"}}],
        "edges": [{
            "type": "KNOWS",
            "from_type": "User", "from_key": {"user_id": "u1"},
            "to_type": "Fact", "to_key": {"fact_id": "f1"}
        }]
    }"""
    cfg = MemoryConfig(extraction_fn=lambda s, u: json_resp)
    ex = MemoryExtractor(cfg)
    payload = ex.extract("u1", [{"role": "user", "content": "I love Python"}])
    assert len(payload.nodes) == 1
    assert payload.nodes[0].type == "Fact"
    assert len(payload.edges) == 1


def test_unknown_provider_raises():
    import pytest
    cfg = MemoryConfig(llm_provider="unknown")
    with pytest.raises(ValueError, match="Unknown llm_provider"):
        MemoryExtractor(cfg)
