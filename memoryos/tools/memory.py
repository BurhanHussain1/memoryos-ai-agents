from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _user_id_from_config(config: RunnableConfig | None) -> str:
    """Pull user_id out of the LangGraph runtime config (auto-injected)."""
    if not config:
        return "default"
    try:
        return config.get("configurable", {}).get("user_id") or "default"
    except Exception:
        return "default"


def make_memory_tools(driver_factory) -> list:
    """
    Returns memory tools bound to the provided driver_factory callable.
    driver_factory() → neo4j driver (GraphConnection.get_driver).

    user_id is auto-injected from the LangGraph runtime config — the LLM
    never sees or supplies it, so it cannot pick the wrong user.
    """

    def _driver():
        return driver_factory()

    @tool
    def remember_fact(
        content: str,
        category: str = "general",
        config: RunnableConfig = None,
    ) -> str:
        """Store a fact about the user in long-term memory.

        Args:
            content: The fact to remember (e.g. "I prefer dark mode").
            category: Category tag — general, work, personal, preference.
        """
        user_id = _user_id_from_config(config)
        fact_id = f"fact_{uuid.uuid4().hex[:8]}"
        now = datetime.now(UTC).isoformat()
        with _driver().session() as s:
            s.run(
                """
                MERGE (u:User {user_id: $uid})
                MERGE (f:Fact {fact_id: $fid})
                SET f.content = $content, f.category = $cat,
                    f.confidence = 1.0, f.created_at = $now
                MERGE (u)-[:KNOWS]->(f)
                """,
                uid=user_id, fid=fact_id,
                content=content, cat=category, now=now,
            )
        logger.info(f"[Tool:remember_fact] Stored fact for user={user_id}: {content[:80]}")
        return f"Got it! I've stored: '{content}'"

    @tool
    def recall_facts(
        category: str = "",
        config: RunnableConfig = None,
    ) -> str:
        """Recall facts stored about the user, optionally filtered by category.

        Args:
            category: Optional filter — general, work, personal, preference.
                      Leave empty to return all facts.
        """
        user_id = _user_id_from_config(config)
        where = ""
        params: dict[str, Any] = {"uid": user_id}
        if category:
            where = "AND toLower(f.category) = toLower($cat)"
            params["cat"] = category
        with _driver().session() as s:
            results = s.run(
                f"""
                MATCH (:User {{user_id: $uid}})-[:KNOWS]->(f:Fact)
                WHERE 1=1 {where}
                RETURN f.content AS content, f.category AS cat
                ORDER BY f.created_at DESC LIMIT 20
                """,
                **params,
            ).data()
        logger.info(f"[Tool:recall_facts] user={user_id} category={category!r} → {len(results)} facts")
        if not results:
            label = f" in category '{category}'" if category else ""
            return f"No facts stored{label} yet."
        lines = [f"What I know about you ({len(results)} facts):"]
        for f in results:
            lines.append(f"- [{f['cat']}] {f['content']}")
        return "\n".join(lines)

    @tool
    def forget_fact(
        fact_id: str,
        config: RunnableConfig = None,
    ) -> str:
        """Remove a specific fact from memory.

        Args:
            fact_id: The fact ID to remove (visible in recall_facts output).
        """
        user_id = _user_id_from_config(config)
        with _driver().session() as s:
            result = s.run(
                """
                MATCH (:User {user_id: $uid})-[r:KNOWS]->(f:Fact {fact_id: $fid})
                DELETE r, f
                RETURN count(f) AS deleted
                """,
                uid=user_id, fid=fact_id,
            ).data()
        deleted = result[0]["deleted"] if result else 0
        logger.info(f"[Tool:forget_fact] user={user_id} fact_id={fact_id} deleted={deleted}")
        return f"Fact {fact_id} removed." if deleted else f"Fact {fact_id} not found."

    @tool
    def who_do_i_know(config: RunnableConfig = None) -> str:
        """List all people stored in the user's Knowledge Graph."""
        user_id = _user_id_from_config(config)
        with _driver().session() as s:
            results = s.run(
                """
                MATCH (:User {user_id: $uid})-[:KNOWS_PERSON]->(p:Person)
                RETURN p.name AS name, p.relation AS relation, p.context AS ctx
                ORDER BY p.name
                """,
                uid=user_id,
            ).data()
        logger.info(f"[Tool:who_do_i_know] user={user_id} → {len(results)} people")
        if not results:
            return "No people stored in your memory yet."
        lines = [f"People you know ({len(results)}):"]
        for p in results:
            rel = f" ({p['relation']})" if p.get("relation") else ""
            ctx = f" — {p['ctx']}" if p.get("ctx") else ""
            lines.append(f"- {p['name']}{rel}{ctx}")
        return "\n".join(lines)

    return [remember_fact, recall_facts, forget_fact, who_do_i_know]
