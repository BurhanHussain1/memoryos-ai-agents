from __future__ import annotations

import json
import logging

from memoryos.graph.connection import GraphConnection
from memoryos.graph.schema import EdgeData, MemoryPayload, NodeData

logger = logging.getLogger(__name__)

_INDEXES_NEO4J = [
    "CREATE INDEX user_id_idx IF NOT EXISTS FOR (n:User) ON (n.user_id)",
    "CREATE INDEX fact_id_idx IF NOT EXISTS FOR (n:Fact) ON (n.fact_id)",
    "CREATE INDEX note_id_idx IF NOT EXISTS FOR (n:Note) ON (n.note_id)",
    "CREATE INDEX reminder_id_idx IF NOT EXISTS FOR (n:Reminder) ON (n.reminder_id)",
    "CREATE INDEX person_name_idx IF NOT EXISTS FOR (n:Person) ON (n.name)",
    "CREATE INDEX project_name_idx IF NOT EXISTS FOR (n:Project) ON (n.name)",
    "CREATE INDEX preference_key_idx IF NOT EXISTS FOR (n:Preference) ON (n.key)",
]

_INDEXES_MEMGRAPH = [
    "CREATE INDEX ON :User(user_id)",
    "CREATE INDEX ON :Fact(fact_id)",
    "CREATE INDEX ON :Note(note_id)",
    "CREATE INDEX ON :Reminder(reminder_id)",
    "CREATE INDEX ON :Person(name)",
    "CREATE INDEX ON :Project(name)",
    "CREATE INDEX ON :Preference(key)",
]


class GraphWriter:
    def __init__(self, connection: GraphConnection):
        self._conn = connection
        self._indexes_ready = False

    def ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        indexes = (
            _INDEXES_MEMGRAPH
            if self._conn.backend_name == "memgraph"
            else _INDEXES_NEO4J
        )
        with self._conn.get_driver().session() as s:
            for idx in indexes:
                try:
                    s.run(idx)
                except Exception:
                    pass
        self._indexes_ready = True
        logger.info(f"[GraphWriter] Indexes ensured ({self._conn.backend_name})")

    def ensure_user_node(self, user_id: str) -> None:
        with self._conn.get_driver().session() as s:
            s.run("MERGE (u:User {user_id: $uid})", uid=user_id)

    def write(self, payload: MemoryPayload) -> dict:
        self.ensure_indexes()
        self.ensure_user_node(payload.user_id)

        nodes_merged = 0
        edges_merged = 0

        with self._conn.get_driver().session() as s:
            for node in payload.nodes:
                if node.type == "User":
                    continue
                try:
                    self._merge_node(s, node)
                    nodes_merged += 1
                except Exception as e:
                    logger.warning(f"[GraphWriter] Node merge failed: {e}")

            for edge in payload.edges:
                try:
                    self._merge_edge(s, edge)
                    edges_merged += 1
                except Exception as e:
                    logger.warning(f"[GraphWriter] Edge merge failed: {e}")

        logger.info(
            f"[GraphWriter] user={payload.user_id} nodes={nodes_merged} edges={edges_merged}"
        )
        return {"nodes_merged": nodes_merged, "edges_merged": edges_merged}

    def delete_user(self, user_id: str) -> int:
        with self._conn.get_driver().session() as s:
            result = s.run(
                "MATCH (u:User {user_id: $uid}) "
                "OPTIONAL MATCH (u)-[r]->(n) "
                "DETACH DELETE n, u "
                "RETURN count(u) AS deleted",
                uid=user_id,
            ).data()
        return result[0]["deleted"] if result else 0

    def _merge_node(self, session, node: NodeData) -> None:
        key_clauses = ", ".join(f"{k}: ${k}" for k in node.key)
        query = f"MERGE (n:{node.type} {{{key_clauses}}})"
        set_props = {k: v for k, v in node.properties.items() if k not in node.key}
        if set_props:
            set_clauses = ", ".join(f"n.{k} = $prop_{k}" for k in set_props)
            query += f" SET {set_clauses}"
        params = dict(node.key)
        for k, v in set_props.items():
            params[f"prop_{k}"] = json.dumps(v) if isinstance(v, list | dict) else v
        session.run(query, **params)

    def _merge_edge(self, session, edge: EdgeData) -> None:
        from_clauses = ", ".join(f"{k}: $from_{k}" for k in edge.from_key)
        to_clauses = ", ".join(f"{k}: $to_{k}" for k in edge.to_key)
        query = (
            f"MATCH (a:{edge.from_type} {{{from_clauses}}})"
            f" MATCH (b:{edge.to_type} {{{to_clauses}}})"
            f" MERGE (a)-[r:{edge.type}]->(b)"
        )
        params = {}
        for k, v in edge.from_key.items():
            params[f"from_{k}"] = v
        for k, v in edge.to_key.items():
            params[f"to_{k}"] = v
        session.run(query, **params)
