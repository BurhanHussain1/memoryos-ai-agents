# MemoryOS — Demo App

A reference FastAPI + Streamlit chat app built on the `memoryos` package, featuring:

- Personal AI agent built with **deepagents**
- 12 tools across 4 skill domains (notes, reminders, research, memory)
- Multi-session UI with live memory inspection
- Both OpenAI and Groq supported via `LLM_PROVIDER` env var
- Both Neo4j and Memgraph supported via `GRAPH_BACKEND` env var

## Setup

```bash
# 1. Install package + extras
pip install -e ".[all]"
pip install -r examples/requirements.txt

# 2. Configure
cp .env.example .env       # edit and add your keys

# 3. Start Neo4j + Redis
docker compose -f examples/docker-compose.yml up -d

# 4. Run the API server (from project root)
python -m examples.run

# 5. Run the Streamlit chat UI (in a second terminal)
streamlit run examples/streamlit_app.py
```

Open the chat at http://localhost:8501.

## Layout

```
examples/
├── api/                 FastAPI app
│   ├── main.py
│   ├── models.py
│   └── session_manager.py
├── tools/               Demo tools (notes, reminders, research)
├── skills/              deepagents skill files
├── agent.py             Agent factory (combines memory + demo tools)
├── streamlit_app.py     Chat UI
├── run.py               API runner with preflight checks
├── docker-compose.yml   Neo4j + Memgraph + Redis
└── requirements.txt
```

## What this demo shows

- **`agent.py`** — how to wire `memory.tools`, `memory.middleware`, and `memory.checkpointer` into a deepagents agent
- **`api/main.py`** — how to use `memory.wrap_input()`, `memory.make_config()`, and `memory.end_session()` per request
- **`tools/notes.py`** and **`tools/reminders.py`** — how to build custom tools that pull `user_id` from the LangGraph runtime config (so the LLM can't pick the wrong user) using `memory._connection.get_driver` for graph access
- **`streamlit_app.py`** — a self-contained UI that exercises every API endpoint

## Switching backends

Edit `.env`:

```bash
GRAPH_BACKEND=neo4j      # or memgraph
LLM_PROVIDER=openai      # or groq
```

Restart the API. Everything else is automatic.

## Viewing the graph

- **Neo4j Browser:** http://localhost:7474 (user: `neo4j`, password from `.env`)
- **Memgraph Lab:**  http://localhost:3000 (no auth)

Try this Cypher query after a session:

```cypher
MATCH (u:User {user_id: 'your_user_id'})-[r]->(n)
RETURN u, r, n
```
