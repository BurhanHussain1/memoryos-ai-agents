from __future__ import annotations

import logging

from memoryos.config import MemoryConfig
from memoryos.graph.connection import GraphConnection
from memoryos.graph.extractor import MemoryExtractor
from memoryos.graph.reader import GraphReader
from memoryos.graph.writer import GraphWriter
from memoryos.middleware import MemoryInjectorMiddleware
from memoryos.store import MemoryStore
from memoryos.tools.memory import make_memory_tools

logger = logging.getLogger(__name__)


class MemoryOS:
    """
    Drop-in persistent memory layer for any AI agent.

    Usage — minimal:
        memory = MemoryOS(
            graph_url="bolt://localhost:7687",
            graph_password="password",
            redis_url="redis://localhost:6379",
            openai_api_key="sk-...",
        )
        await memory.init()

        agent = create_deep_agent(
            model=...,
            tools=[...] + memory.tools,
            middleware=[memory.middleware],
            checkpointer=memory.checkpointer,
        )

        # Per request:
        result = await agent.ainvoke(
            memory.wrap_input(user_message, user_id),
            config=memory.make_config(user_id, session_id),
        )
        memory.save_message(session_id, "user", user_message)
        memory.save_message(session_id, "assistant", result["messages"][-1].content)

        # On session end:
        memory.end_session(user_id, session_id)

    Usage — from env:
        memory = MemoryOS.from_env()
        await memory.init()

    Usage — explicit config:
        cfg = MemoryConfig(graph_url=..., graph_password=..., ...)
        memory = MemoryOS(config=cfg)
        await memory.init()
    """

    def __init__(self, config: MemoryConfig | None = None, **kwargs):
        if config is None:
            config = MemoryConfig(**kwargs)
        self._config = config

        # Core components
        self._connection = GraphConnection(config)
        self._store = MemoryStore(config)
        self._reader = GraphReader(self._connection)
        self._writer = GraphWriter(self._connection)
        self._extractor = MemoryExtractor(config)

        # Wire extractor + writer into store so it can fire-and-forget
        self._store._extractor = self._extractor
        self._store._writer = self._writer

        # Middleware and tools
        self._middleware = MemoryInjectorMiddleware(
            reader=self._reader,
            store=self._store,
        )
        self._tools = make_memory_tools(self._connection.get_driver)

        self._initialized = False
        logger.info(f"[MemoryOS] Created — backend={config.graph_backend} | extraction_model={config.extraction_model}")

    @classmethod
    def from_env(cls) -> MemoryOS:
        """Build MemoryOS from environment variables (reads .env if present)."""
        return cls(config=MemoryConfig.from_env())

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def init(self) -> MemoryOS:
        """
        Initialize async components: Redis checkpointer + graph indexes.
        Must be awaited once at application startup before the first request.
        """
        await self._store.init_checkpointer()
        try:
            self._writer.ensure_indexes()
        except Exception as e:
            logger.warning(f"[MemoryOS] Index setup failed (non-fatal): {e}")
        self._initialized = True
        logger.info(f"[MemoryOS] Ready — graph={self._config.graph_url} | redis={self._config.redis_url}")
        return self

    async def close(self) -> None:
        """Graceful shutdown — close Redis checkpointer + graph driver."""
        await self._store.close_checkpointer()
        self._connection.close()
        logger.info("[MemoryOS] Shutdown complete")

    # ── Properties exposed to the host agent ────────────────────────────────

    @property
    def checkpointer(self):
        """LangGraph checkpointer (AsyncRedisSaver or MemorySaver fallback)."""
        return self._store.get_checkpointer()

    @property
    def middleware(self) -> MemoryInjectorMiddleware:
        """AgentMiddleware that injects graph context into every LLM call."""
        return self._middleware

    @property
    def tools(self) -> list:
        """LangChain tools: remember_fact, recall_facts, forget_fact, who_do_i_know."""
        return self._tools

    @property
    def connection(self) -> GraphConnection:
        """GraphConnection — exposes the graph driver to custom tools."""
        return self._connection

    @property
    def store(self) -> MemoryStore:
        """MemoryStore — Redis STM + checkpointer manager."""
        return self._store

    @property
    def reader(self) -> GraphReader:
        """GraphReader — read-side queries over the user's KG."""
        return self._reader

    @property
    def writer(self) -> GraphWriter:
        """GraphWriter — write-side merge ops over the user's KG."""
        return self._writer

    @property
    def extractor(self) -> MemoryExtractor:
        """MemoryExtractor — converts a session's transcript into KG nodes/edges."""
        return self._extractor

    @property
    def config(self) -> MemoryConfig:
        """The MemoryConfig instance backing this MemoryOS."""
        return self._config

    # ── Per-request helpers ──────────────────────────────────────────────────

    def wrap_input(self, message: str, user_id: str) -> dict:
        """
        Build the state dict for agent.ainvoke().

        result = await agent.ainvoke(
            memory.wrap_input(user_message, user_id),
            config=memory.make_config(user_id, session_id),
        )
        """
        return {
            "messages": [{"role": "user", "content": message}],
            "user_id": user_id,
        }

    def make_config(self, user_id: str, session_id: str) -> dict:
        """Build the LangGraph config dict for agent.ainvoke()."""
        return {
            "configurable": {
                "thread_id": session_id,
                "user_id": user_id,
            }
        }

    def save_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to Redis STM for this session."""
        self._store.save_message(session_id, role, content)

    def end_session(self, user_id: str, session_id: str) -> str:
        """
        Trigger async LTM extraction for the session.
        Returns 'queued' if extraction started, 'skipped' if no messages.
        Non-blocking — extraction runs in a background thread.
        """
        return self._store.end_session(user_id, session_id)

    def get_context(self, user_id: str) -> str:
        """Fetch formatted memory context string from the graph (bypasses cache)."""
        return self._reader.get_user_context(user_id)

    def get_extraction_status(self, user_id: str) -> str:
        """Returns 'processing', 'ready', or 'unknown'."""
        return self._store.get_extraction_status(user_id)

    def invalidate_cache(self, user_id: str) -> None:
        """Force the middleware to re-fetch context on the next request."""
        self._middleware.invalidate_cache(user_id)

    def delete_user_memory(self, user_id: str) -> int:
        """Delete all graph nodes for a user. Returns number of user nodes deleted."""
        self.invalidate_cache(user_id)
        return self._writer.delete_user(user_id)

    def verify_connectivity(self) -> dict[str, bool]:
        """Health check — returns {'graph': bool, 'redis': bool}."""
        graph_ok = False
        redis_ok = False
        try:
            self._connection.get_driver().verify_connectivity()
            graph_ok = True
        except Exception:
            pass
        try:
            r = self._store._get_redis()
            if r:
                r.ping()
                redis_ok = True
        except Exception:
            pass
        return {"graph": graph_ok, "redis": redis_ok}

    # ── Framework-agnostic helpers ──────────────────────────────────────────
    # Use these when integrating with non-LangGraph frameworks:
    # OpenAI SDK / Assistants API / LlamaIndex / AutoGen / custom agents.

    def openai_tools(self) -> list[dict]:
        """Memory tools as OpenAI tool schemas.

        Use with the OpenAI SDK, Assistants API, or any framework that
        accepts OpenAI's `{"type": "function", "function": {...}}` format.

        Tool names match what `handle_tool_call()` dispatches on.
        """
        from langchain_core.utils.function_calling import convert_to_openai_tool
        return [convert_to_openai_tool(t) for t in self._tools]

    def handle_tool_call(self, name: str, args: dict, user_id: str) -> str:
        """Execute a memory tool by name — for non-LangChain frameworks.

        The LLM-supplied `args` dict should contain the visible parameters
        (e.g. content, category for remember_fact). user_id is supplied
        explicitly here instead of via LangGraph runtime config.

        Example with raw OpenAI SDK:

            tool_call = response.choices[0].message.tool_calls[0]
            result = memory.handle_tool_call(
                name=tool_call.function.name,
                args=json.loads(tool_call.function.arguments),
                user_id=current_user_id,
            )
        """
        config = {"configurable": {"user_id": user_id}}
        for tool in self._tools:
            if tool.name == name:
                return tool.invoke(args, config=config)
        available = [t.name for t in self._tools]
        raise ValueError(f"Unknown memory tool: {name!r}. Available: {available}")

    def inject_context(self, messages: list[dict], user_id: str) -> list[dict]:
        """Prepend the user's memory context to a plain message list.

        Returns a new list — does not mutate input. If a system message
        already exists at index 0, the context is appended to it instead
        of being added as a second system message.

        Works with any framework that uses OpenAI-style messages
        ({"role": "system" | "user" | "assistant", "content": "..."}).

        Example:

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input},
            ]
            messages = memory.inject_context(messages, user_id)
            response = openai_client.chat.completions.create(model=..., messages=messages)
        """
        ctx = self.get_context(user_id)
        if not ctx or not ctx.strip():
            return list(messages)
        block = f"{ctx}\n\nUse the above context to personalize your responses."
        if messages and messages[0].get("role") == "system":
            merged = {
                "role": "system",
                "content": messages[0]["content"] + "\n\n" + block,
            }
            return [merged, *messages[1:]]
        return [{"role": "system", "content": block}, *list(messages)]

    def tool_names(self) -> list[str]:
        """Names of the bundled memory tools — useful for filtering tool-call dispatch."""
        return [t.name for t in self._tools]
