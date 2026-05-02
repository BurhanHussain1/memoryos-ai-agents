from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import SystemMessage

from memoryos.graph.reader import GraphReader
from memoryos.store import MemoryStore

logger = logging.getLogger(__name__)


class MemoryInjectorMiddleware(AgentMiddleware):
    """
    Injects graph memory context into every LLM call.

    before_agent() — fetches context once per agent turn.
    wrap_model_call() — prepends it as a SystemMessage block on every LLM call.
    """

    def __init__(self, reader: GraphReader, store: MemoryStore):
        super().__init__()
        self._reader = reader
        self._store = store
        self._cache: dict[str, tuple[str, float]] = {}
        self._cache_ttl = store._config.context_cache_ttl
        self._current_context: str = ""
        self._current_user_id: str = "default"

    def _get_context(self, user_id: str) -> str:
        if user_id in self._cache:
            ctx, ts = self._cache[user_id]
            if time.time() - ts < self._cache_ttl:
                logger.debug(f"[MemoryInjector] Cache hit user={user_id}")
                return ctx
            del self._cache[user_id]

        logger.info(f"[MemoryInjector] Querying graph for user={user_id}")
        try:
            ctx = self._reader.get_user_context(user_id)
        except Exception as e:
            logger.warning(f"[MemoryInjector] Graph read failed: {e}")
            ctx = ""

        # Bridge period — append raw STM from previous session during extraction
        bridge = self._store.get_bridge_context(user_id)
        if bridge:
            ctx = (ctx or "") + bridge
            logger.info(f"[MemoryInjector] Bridge context appended for user={user_id}")

        if ctx:
            self._cache[user_id] = (ctx, time.time())
        return ctx

    def invalidate_cache(self, user_id: str) -> None:
        self._cache.pop(user_id, None)

    # ── before_agent — runs once per turn ───────────────────────────────────

    def _resolve_user_id(self, state: Any, runtime: Any) -> str:
        """Walk every plausible location LangGraph might stash user_id."""
        # 1. State as dict (raw input)
        try:
            if isinstance(state, dict):
                uid = state.get("user_id")
                if uid:
                    return uid
        except Exception:
            pass

        # 2. State as Pydantic model / dataclass / object with attr
        uid = getattr(state, "user_id", None)
        if uid:
            return uid

        # 3. runtime.config["configurable"]["user_id"] — official path
        try:
            cfg = getattr(runtime, "config", None)
            if callable(cfg):
                cfg = cfg()
            if isinstance(cfg, dict):
                uid = cfg.get("configurable", {}).get("user_id")
                if uid:
                    return uid
        except Exception:
            pass

        # 4. runtime["configurable"]["user_id"] — runtime is a dict in some versions
        try:
            if isinstance(runtime, dict):
                uid = runtime.get("configurable", {}).get("user_id")
                if uid:
                    return uid
        except Exception:
            pass

        # 5. runtime.context — typed context object (newer LangGraph)
        try:
            ctx = getattr(runtime, "context", None)
            if ctx is not None:
                if isinstance(ctx, dict):
                    uid = ctx.get("user_id")
                else:
                    uid = getattr(ctx, "user_id", None)
                if uid:
                    return uid
        except Exception:
            pass

        # 6. langgraph.config.get_config() — last-resort global lookup
        try:
            from langgraph.config import get_config
            cfg = get_config() or {}
            uid = cfg.get("configurable", {}).get("user_id")
            if uid:
                return uid
        except Exception:
            pass

        return "default"

    def before_agent(self, state: Any, runtime: Any) -> None:
        user_id = self._resolve_user_id(state, runtime)
        if user_id == "default":
            # Surface what's available so we can debug if propagation breaks again
            logger.warning(
                f"[MemoryInjector] user_id not found — state_type={type(state).__name__} "
                f"runtime_type={type(runtime).__name__} runtime_attrs={dir(runtime)[:15]}"
            )
        self._current_user_id = user_id
        self._current_context = self._get_context(user_id)
        logger.info(f"[MemoryInjector] Context loaded for user={user_id} ({len(self._current_context)} chars)")

    async def abefore_agent(self, state: Any, runtime: Any) -> None:
        self.before_agent(state, runtime)

    # ── wrap_model_call — runs before every LLM call ────────────────────────

    def _build_memory_block(self) -> str | None:
        ctx = self._current_context
        if not ctx or not ctx.strip():
            return None
        return f"\n\n{ctx}\n\nUse the above context to personalize your responses."

    def _apply_injection(self, request: ModelRequest, block: str) -> ModelRequest:
        if request.system_message is not None:
            new_system = SystemMessage(content=cast(
                "list[str | dict[str, str]]",
                [*request.system_message.content_blocks, {"type": "text", "text": block}],
            ))
        else:
            new_system = SystemMessage(content=block)
        logger.info(
            f"[MemoryInjector] Injected {len(block)} chars for user={self._current_user_id}"
        )
        return request.override(system_message=new_system)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        block = self._build_memory_block()
        if block:
            return handler(self._apply_injection(request, block))
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        block = self._build_memory_block()
        if block:
            return await handler(self._apply_injection(request, block))
        return await handler(request)
