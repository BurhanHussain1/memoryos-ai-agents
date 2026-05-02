from __future__ import annotations

import logging

from neo4j import GraphDatabase

from memoryos.config import MemoryConfig

logger = logging.getLogger(__name__)


class GraphConnection:
    """Per-instance Neo4j/Memgraph driver — no global state."""

    def __init__(self, config: MemoryConfig):
        self._config = config
        self._driver = None

    def get_driver(self):
        if self._driver is None:
            cfg = self._config
            if cfg.graph_password:
                auth = (cfg.graph_username, cfg.graph_password)
            else:
                auth = None
            self._driver = GraphDatabase.driver(cfg.graph_url, auth=auth)
            logger.info(f"[GraphConnection] Connected to {cfg.graph_backend} at {cfg.graph_url}")
        return self._driver

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("[GraphConnection] Driver closed")

    @property
    def backend_name(self) -> str:
        return self._config.graph_backend
