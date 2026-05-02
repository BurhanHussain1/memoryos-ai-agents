from __future__ import annotations

import uuid
import logging

from langchain_core.tools import tool
from langchain_core.runnables.config import RunnableConfig

logger = logging.getLogger(__name__)


def _user_id_from_config(config: RunnableConfig | None) -> str:
    if not config:
        return "default"
    return config.get("configurable", {}).get("user_id") or "default"


def make_reminder_tools(driver_factory) -> list:
    """Returns reminder tools bound to a Neo4j driver factory."""

    @tool
    def set_reminder(title: str, due_date: str, config: RunnableConfig = None) -> str:
        """Set a reminder for the user.

        Args:
            title: What to remind the user about.
            due_date: ISO 8601 date/time string (e.g. "2026-05-01T09:00:00").
        """
        user_id = _user_id_from_config(config)
        reminder_id = f"rem_{uuid.uuid4().hex[:8]}"
        with driver_factory().session() as s:
            s.run(
                """
                MERGE (u:User {user_id: $uid})
                MERGE (r:Reminder {reminder_id: $rid})
                SET r.title = $title, r.due_date = $due, r.completed = false
                MERGE (u)-[:HAS_REMINDER]->(r)
                """,
                uid=user_id, rid=reminder_id, title=title, due=due_date,
            )
        return f"Reminder set: '{title}' for {due_date} (id: {reminder_id})"

    @tool
    def list_reminders(include_completed: bool = False, config: RunnableConfig = None) -> str:
        """List the user's reminders.

        Args:
            include_completed: If True, also show completed reminders.
        """
        user_id = _user_id_from_config(config)
        where = "" if include_completed else "WHERE r.completed = false"
        with driver_factory().session() as s:
            results = s.run(
                f"""
                MATCH (:User {{user_id: $uid}})-[:HAS_REMINDER]->(r:Reminder)
                {where}
                RETURN r.reminder_id AS id, r.title AS title,
                       r.due_date AS due, r.completed AS done
                ORDER BY r.due_date ASC
                """,
                uid=user_id,
            ).data()
        if not results:
            return "No reminders found."
        lines = [f"Your reminders ({len(results)}):"]
        for r in results:
            status = "[DONE]" if r.get("done") else "[PENDING]"
            due = r.get("due") or "no date"
            lines.append(f"- {status} {r['title']} — {due} (id: {r['id']})")
        return "\n".join(lines)

    @tool
    def complete_reminder(reminder_id: str, config: RunnableConfig = None) -> str:
        """Mark a reminder as completed.

        Args:
            reminder_id: The reminder ID to mark as complete.
        """
        user_id = _user_id_from_config(config)
        with driver_factory().session() as s:
            result = s.run(
                """
                MATCH (:User {user_id: $uid})-[:HAS_REMINDER]->(r:Reminder {reminder_id: $rid})
                SET r.completed = true
                RETURN r.title AS title
                """,
                uid=user_id, rid=reminder_id,
            ).data()
        if not result:
            return f"Reminder {reminder_id} not found."
        return f"Reminder '{result[0]['title']}' marked as complete."

    return [set_reminder, list_reminders, complete_reminder]
