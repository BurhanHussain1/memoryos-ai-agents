from __future__ import annotations

import logging
from typing import Any

from memoryos.graph.connection import GraphConnection

logger = logging.getLogger(__name__)


class GraphReader:
    def __init__(self, connection: GraphConnection):
        self._conn = connection

    def _run(self, query: str, **params) -> list[dict[str, Any]]:
        with self._conn.get_driver().session() as s:
            return [dict(r) for r in s.run(query, **params)]

    def get_user_context(self, user_id: str) -> str:
        lines = ["## What I know about you"]
        has_data = False

        projects = self._run(
            "MATCH (:User {user_id: $uid})-[:WORKING_ON]->(p:Project) "
            "RETURN p.name AS name, p.status AS status, p.description AS desc",
            uid=user_id,
        )
        if projects:
            has_data = True
            lines.append("\n### Projects you're working on")
            for p in projects:
                status = f" [{p['status']}]" if p.get("status") else ""
                desc = f" — {p['desc']}" if p.get("desc") else ""
                lines.append(f"- {p['name']}{status}{desc}")

        prefs = self._run(
            "MATCH (:User {user_id: $uid})-[:PREFERS]->(p:Preference) "
            "RETURN p.key AS key, p.value AS value ORDER BY p.confidence DESC",
            uid=user_id,
        )
        if prefs:
            has_data = True
            lines.append("\n### Your preferences")
            for p in prefs:
                lines.append(f"- {p['key']}: {p['value']}")

        people = self._run(
            "MATCH (:User {user_id: $uid})-[:KNOWS_PERSON]->(p:Person) "
            "RETURN p.name AS name, p.relation AS relation, p.context AS ctx",
            uid=user_id,
        )
        if people:
            has_data = True
            lines.append("\n### People you know")
            for p in people:
                rel = f" ({p['relation']})" if p.get("relation") else ""
                ctx = f": {p['ctx']}" if p.get("ctx") else ""
                lines.append(f"- {p['name']}{rel}{ctx}")

        facts = self._run(
            "MATCH (:User {user_id: $uid})-[:KNOWS]->(f:Fact) "
            "RETURN f.content AS content, f.category AS cat "
            "ORDER BY f.created_at DESC LIMIT 10",
            uid=user_id,
        )
        if facts:
            has_data = True
            lines.append("\n### Facts about you")
            for f in facts:
                cat = f" [{f['cat']}]" if f.get("cat") else ""
                lines.append(f"- {f['content']}{cat}")

        reminders = self._run(
            "MATCH (:User {user_id: $uid})-[:HAS_REMINDER]->(r:Reminder) "
            "WHERE r.completed = false "
            "RETURN r.title AS title, r.due_date AS due ORDER BY r.due_date ASC LIMIT 5",
            uid=user_id,
        )
        if reminders:
            has_data = True
            lines.append("\n### Upcoming reminders")
            for r in reminders:
                due = f" (due: {r['due']})" if r.get("due") else ""
                lines.append(f"- {r['title']}{due}")

        if not has_data:
            return ""

        return "\n".join(lines)
