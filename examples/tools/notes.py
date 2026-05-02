from __future__ import annotations

import uuid
import logging
from datetime import datetime, timezone

from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

logger = logging.getLogger(__name__)


def _user_id_from_config(config: RunnableConfig | None) -> str:
    if not config:
        return "default"
    return config.get("configurable", {}).get("user_id") or "default"


def make_note_tools(driver_factory) -> list:
    """Returns note tools bound to a Neo4j driver factory."""

    @tool
    def create_note(title: str, content: str, tags: str = "", config: RunnableConfig = None) -> str:
        """Create and save a personal note for the user.

        Args:
            title: Short descriptive title for the note.
            content: Full note content.
            tags: Optional comma-separated tags.
        """
        user_id = _user_id_from_config(config)
        note_id = f"note_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        with driver_factory().session() as s:
            s.run(
                """
                MERGE (u:User {user_id: $uid})
                MERGE (n:Note {note_id: $nid})
                SET n.title = $title, n.content = $content,
                    n.tags = $tags, n.created_at = $now
                MERGE (u)-[:HAS_NOTE]->(n)
                """,
                uid=user_id, nid=note_id,
                title=title, content=content, tags=tags, now=now,
            )
        return f"Note saved: '{title}' (id: {note_id})"

    @tool
    def search_notes(query: str, config: RunnableConfig = None) -> str:
        """Search the user's notes by keyword in title or content.

        Args:
            query: Search term to match against note titles and content.
        """
        user_id = _user_id_from_config(config)
        with driver_factory().session() as s:
            results = s.run(
                """
                MATCH (:User {user_id: $uid})-[:HAS_NOTE]->(n:Note)
                WHERE toLower(n.title) CONTAINS toLower($q)
                   OR toLower(n.content) CONTAINS toLower($q)
                RETURN n.note_id AS id, n.title AS title,
                       n.content AS content, n.created_at AS ts
                ORDER BY n.created_at DESC LIMIT 10
                """,
                uid=user_id, q=query,
            ).data()
        if not results:
            return f"No notes found matching '{query}'."
        lines = [f"Found {len(results)} note(s) for '{query}':"]
        for r in results:
            date = r["ts"][:10] if r.get("ts") else "?"
            snippet = r["content"][:150] + ("..." if len(r["content"]) > 150 else "")
            lines.append(f"- [{r['id']}] {r['title']} ({date})")
            lines.append(f"  {snippet}")
        return "\n".join(lines)

    @tool
    def list_notes(limit: int = 10, config: RunnableConfig = None) -> str:
        """List the user's most recent notes.

        Args:
            limit: Maximum number of notes to return (default 10).
        """
        user_id = _user_id_from_config(config)
        with driver_factory().session() as s:
            results = s.run(
                """
                MATCH (:User {user_id: $uid})-[:HAS_NOTE]->(n:Note)
                RETURN n.note_id AS id, n.title AS title, n.created_at AS ts
                ORDER BY n.created_at DESC LIMIT $lim
                """,
                uid=user_id, lim=limit,
            ).data()
        if not results:
            return "You have no saved notes yet."
        lines = [f"Your {len(results)} most recent notes:"]
        for r in results:
            date = r["ts"][:10] if r.get("ts") else "?"
            lines.append(f"- [{r['id']}] {r['title']} ({date})")
        return "\n".join(lines)

    @tool
    def delete_note(note_id: str, config: RunnableConfig = None) -> str:
        """Delete a note by its ID.

        Args:
            note_id: The note ID to delete.
        """
        user_id = _user_id_from_config(config)
        with driver_factory().session() as s:
            result = s.run(
                """
                MATCH (:User {user_id: $uid})-[r:HAS_NOTE]->(n:Note {note_id: $nid})
                DELETE r, n
                RETURN count(n) AS deleted
                """,
                uid=user_id, nid=note_id,
            ).data()
        deleted = result[0]["deleted"] if result else 0
        return f"Note {note_id} deleted." if deleted else f"Note {note_id} not found."

    return [create_note, search_notes, list_notes, delete_note]
