from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

VALID_NODE_TYPES = frozenset({
    "User", "Fact", "Project", "Person", "Preference", "Note", "Reminder",
})

VALID_EDGE_TYPES = frozenset({
    "KNOWS", "WORKING_ON", "KNOWS_PERSON", "PREFERS", "HAS_NOTE", "HAS_REMINDER",
})


class NodeData(BaseModel):
    type: str
    key: dict[str, Any]
    properties: dict[str, Any] = Field(default_factory=dict)


class EdgeData(BaseModel):
    type: str
    from_type: str
    from_key: dict[str, Any]
    to_type: str
    to_key: dict[str, Any]
    properties: dict[str, Any] = Field(default_factory=dict)


class MemoryPayload(BaseModel):
    user_id: str
    nodes: list[NodeData] = Field(default_factory=list)
    edges: list[EdgeData] = Field(default_factory=list)
