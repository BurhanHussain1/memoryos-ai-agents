from memoryos import MemoryPayload, NodeData, EdgeData


def test_node_data_minimal():
    n = NodeData(type="Fact", key={"fact_id": "fact_abc"})
    assert n.type == "Fact"
    assert n.properties == {}


def test_edge_data():
    e = EdgeData(
        type="KNOWS",
        from_type="User", from_key={"user_id": "u1"},
        to_type="Fact",  to_key={"fact_id": "f1"},
    )
    assert e.type == "KNOWS"


def test_memory_payload():
    payload = MemoryPayload(
        user_id="u1",
        nodes=[NodeData(type="Fact", key={"fact_id": "f1"}, properties={"content": "hi"})],
        edges=[EdgeData(
            type="KNOWS",
            from_type="User", from_key={"user_id": "u1"},
            to_type="Fact", to_key={"fact_id": "f1"},
        )],
    )
    assert payload.user_id == "u1"
    assert len(payload.nodes) == 1
    assert len(payload.edges) == 1
