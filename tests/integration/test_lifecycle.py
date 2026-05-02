"""Integration test — requires Neo4j + Redis running.

Run: pytest tests/integration/ -m integration
Skip in regular CI: pytest tests/unit/   (default behaviour)
"""
import asyncio
import os
import uuid

import pytest

# Skip the whole file if integration deps aren't available
pytest.importorskip("neo4j")
pytest.importorskip("redis")

INTEGRATION = os.getenv("MEMORYOS_RUN_INTEGRATION") == "1"


pytestmark = pytest.mark.skipif(
    not INTEGRATION,
    reason="Set MEMORYOS_RUN_INTEGRATION=1 + run Neo4j+Redis to enable",
)


@pytest.mark.asyncio
async def test_full_lifecycle():
    """End-to-end: write fact → extract on session end → read back across sessions."""
    from memoryos import MemoryOS, MemoryConfig

    user_id = f"test_user_{uuid.uuid4().hex[:6]}"
    session_id_1 = f"sess_{uuid.uuid4().hex[:8]}"

    cfg = MemoryConfig.from_env()
    cfg.extraction_fn = lambda system, user: (
        '{"nodes": [{"type":"Fact","key":{"fact_id":"f1"},"properties":{"content":"likes test","category":"general"}}],'
        ' "edges": [{"type":"KNOWS","from_type":"User","from_key":{"user_id":"' + user_id + '"},'
        '"to_type":"Fact","to_key":{"fact_id":"f1"}}]}'
    )

    memory = MemoryOS(config=cfg)
    await memory.init()

    try:
        # Write through the tool API
        memory.handle_tool_call("remember_fact", {"content": "I like pasta", "category": "personal"}, user_id=user_id)

        # Save messages and trigger extraction
        memory.save_message(session_id_1, "user", "I like pasta")
        memory.save_message(session_id_1, "assistant", "Got it.")
        status = memory.end_session(user_id, session_id_1)
        assert status == "queued"

        # Wait for extraction
        for _ in range(30):
            if memory.get_extraction_status(user_id) == "ready":
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("extraction did not finish in time")

        # Read back from a fresh "session"
        ctx = memory.get_context(user_id)
        assert "pasta" in ctx.lower() or "likes test" in ctx.lower()

    finally:
        memory.delete_user_memory(user_id)
        await memory.close()
