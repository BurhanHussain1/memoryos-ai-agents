from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor

import redis as redis_lib
from langgraph.checkpoint.memory import MemorySaver

from memoryos.config import MemoryConfig

logger = logging.getLogger(__name__)


class MemoryStore:
    """Redis-backed STM + LangGraph checkpointer. One instance per MemoryOS."""

    def __init__(self, config: MemoryConfig):
        self._config = config
        self._redis_client: redis_lib.Redis | None = None
        self._checkpointer = MemorySaver()
        self._redis_cm = None
        self._executor = ThreadPoolExecutor(max_workers=3)

        # injected after init() — avoid circular import at construction time
        self._extractor = None
        self._writer = None

    def _get_redis(self) -> redis_lib.Redis | None:
        if self._redis_client is not None:
            return self._redis_client
        cfg = self._config
        try:
            r = redis_lib.Redis(
                host=cfg.redis_host,
                port=cfg.redis_port,
                password=cfg.redis_password or None,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            r.ping()
            self._redis_client = r
            logger.info("[MemoryStore] Redis connected")
            logger.info(f"[STM] Redis connected at {cfg.redis_host}:{cfg.redis_port}")
        except Exception as e:
            logger.warning(f"[MemoryStore] Redis unavailable — STM disabled: {e}")
        return self._redis_client

    async def init_checkpointer(self):
        """Set up AsyncRedisSaver; falls back to in-memory MemorySaver."""
        try:
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver
            self._redis_cm = AsyncRedisSaver.from_conn_string(self._config.redis_url)
            saver = await self._redis_cm.__aenter__()
            await saver.asetup()
            self._checkpointer = saver
            logger.info("[MemoryStore] Redis checkpointer ready")
            logger.info("[Checkpointer] Redis checkpointer ready")
        except Exception as e:
            logger.warning(f"[MemoryStore] Redis checkpointer unavailable, using in-memory: {e}")
            logger.info(f"[Checkpointer] Falling back to in-memory: {e}")
        return self._checkpointer

    async def close_checkpointer(self) -> None:
        try:
            if self._redis_cm is not None:
                await self._redis_cm.__aexit__(None, None, None)
        except Exception:
            pass

    def get_checkpointer(self):
        return self._checkpointer

    # ── STM helpers ─────────────────────────────────────────────────────────

    def _stm_key(self, session_id: str) -> str:
        return f"memoryos:stm:{session_id}"

    def save_message(self, session_id: str, role: str, content: str) -> None:
        r = self._get_redis()
        if not r:
            return
        try:
            msg = json.dumps({"role": role, "content": content})
            key = self._stm_key(session_id)
            r.rpush(key, msg)
            r.ltrim(key, -self._config.max_stm_messages, -1)
            r.expire(key, self._config.stm_ttl_seconds)
            logger.info(f"[STM] Saved message role={role} session={session_id}")
        except Exception as e:
            logger.warning(f"[MemoryStore] save_message failed: {e}")

    def get_stm_messages(self, session_id: str) -> list[dict]:
        r = self._get_redis()
        if not r:
            return []
        try:
            raw = r.lrange(self._stm_key(session_id), 0, -1)
            return [json.loads(m) for m in raw]
        except Exception as e:
            logger.warning(f"[MemoryStore] get_stm_messages failed: {e}")
            return []

    def delete_stm(self, session_id: str) -> None:
        r = self._get_redis()
        if not r:
            return
        try:
            r.delete(self._stm_key(session_id))
        except Exception:
            pass

    # ── Session end / extraction ─────────────────────────────────────────────

    def end_session(self, user_id: str, session_id: str) -> str:
        messages = self.get_stm_messages(session_id)
        if not messages:
            logger.info(f"[Session] No messages to extract for session={session_id}")
            logger.info(f"[MemoryStore] No messages for session={session_id}")
            return "skipped"
        r = self._get_redis()
        if r:
            try:
                r.set(f"memoryos:status:{user_id}", "processing", ex=3600)
                r.set(f"memoryos:prev_session:{user_id}", session_id, ex=3600)
            except Exception:
                pass
        logger.info(f"[Session] Session ended user={user_id} | {len(messages)} messages → queuing extraction")
        self._executor.submit(self._extract_and_write, user_id, session_id, messages)
        return "queued"

    def _extract_and_write(self, user_id: str, session_id: str, messages: list[dict]) -> None:
        logger.info(f"[Memory] Extracting memory for user={user_id} ({len(messages)} messages)...")
        try:
            payload = self._extractor.extract(user_id, messages)
            if payload.nodes:
                logger.info(f"[Memory] Writing {len(payload.nodes)} nodes, {len(payload.edges)} edges to graph...")
                stats = self._writer.write(payload)
                logger.info(f"[Memory] KG write complete: {stats}")
                logger.info(f"[MemoryStore] KG write done: {stats}")
            else:
                logger.info(f"[Memory] Nothing to extract from session={session_id}")
                logger.info("[MemoryStore] Nothing extracted — skipping KG write")
        except Exception as e:
            logger.info(f"[Memory] Extraction failed for user={user_id}: {e}")
            logger.error(f"[MemoryStore] KG extraction error: {e}")
        finally:
            self.delete_stm(session_id)
            r = self._get_redis()
            if r:
                try:
                    r.set(f"memoryos:status:{user_id}", "ready", ex=86400)
                    r.delete(f"memoryos:prev_session:{user_id}")
                    logger.info(f"[Memory] Extraction complete — status=ready for user={user_id}")
                except Exception:
                    pass

    # ── Status helpers ───────────────────────────────────────────────────────

    def get_extraction_status(self, user_id: str) -> str:
        r = self._get_redis()
        if not r:
            return "unknown"
        try:
            return r.get(f"memoryos:status:{user_id}") or "ready"
        except Exception:
            return "unknown"

    def get_bridge_context(self, user_id: str) -> str | None:
        """Return raw STM from previous session during extraction bridge period."""
        r = self._get_redis()
        if not r:
            return None
        try:
            status = r.get(f"memoryos:status:{user_id}")
            if status != "processing":
                return None
            prev_session_id = r.get(f"memoryos:prev_session:{user_id}")
            if not prev_session_id:
                return None
            raw = r.lrange(f"memoryos:stm:{prev_session_id}", 0, -1)
            if not raw:
                return None
            lines = ["\n### Recent Conversation (being saved to memory)"]
            for m in raw[-10:]:
                try:
                    msg = json.loads(m)
                    role = msg.get("role", "unknown").upper()
                    content = msg.get("content", "")[:300]
                    lines.append(f"- {role}: {content}")
                except Exception:
                    pass
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"[MemoryStore] Bridge context failed: {e}")
            return None
