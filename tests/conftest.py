"""Shared pytest configuration."""
import os
import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests in tests/integration/ as integration."""
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def offline_config():
    """A MemoryConfig that doesn't try to connect to anything live."""
    from memoryos import MemoryConfig
    return MemoryConfig(
        graph_url="bolt://localhost:7687",
        graph_password="x",
        openai_api_key="test-key",
        extraction_fn=lambda system, user: '{"nodes": [], "edges": []}',
    )
