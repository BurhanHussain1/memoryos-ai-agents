"""
memoryos — drop-in persistent memory for AI agents.

Quick start:
    from memoryos import MemoryOS, MemoryConfig

    memory = MemoryOS(
        graph_url="bolt://localhost:7687",
        graph_password="password",
        redis_url="redis://localhost:6379",
        openai_api_key="sk-...",
    )
    await memory.init()

    # Attach to your agent:
    agent = create_deep_agent(
        model=...,
        tools=[...] + memory.tools,
        middleware=[memory.middleware],
        checkpointer=memory.checkpointer,
    )

    # Per request:
    result = await agent.ainvoke(
        memory.wrap_input(message, user_id),
        config=memory.make_config(user_id, session_id),
    )
    memory.save_message(session_id, "user", message)
    memory.save_message(session_id, "assistant", result["messages"][-1].content)

    # On session end:
    memory.end_session(user_id, session_id)
"""

from memoryos.config import MemoryConfig
from memoryos.core import MemoryOS
from memoryos.graph.connection import GraphConnection
from memoryos.graph.reader import GraphReader
from memoryos.graph.schema import EdgeData, MemoryPayload, NodeData
from memoryos.graph.writer import GraphWriter
from memoryos.middleware import MemoryInjectorMiddleware
from memoryos.store import MemoryStore

__version__ = "0.1.0"

__all__ = [
    "EdgeData",
    "GraphConnection",
    "GraphReader",
    "GraphWriter",
    "MemoryConfig",
    "MemoryInjectorMiddleware",
    "MemoryOS",
    "MemoryPayload",
    "MemoryStore",
    "NodeData",
]
