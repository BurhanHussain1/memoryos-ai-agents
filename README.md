# memoryos

Drop-in **persistent memory layer** for any AI agent — pluggable Knowledge Graph (Neo4j or Memgraph) for long-term memory + Redis for short-term memory and LangGraph checkpointing.

Works with OpenAI or Groq for memory extraction. Works with any LangGraph-compatible agent framework (deepagents, LangChain agents, raw LangGraph).

## Install

```bash
pip install -e .[openai]      # OpenAI backend
pip install -e .[groq]        # Groq backend
pip install -e .[all]         # Both
```

## Quick start

```python
from memoryos import MemoryOS

memory = MemoryOS(
    graph_url="bolt://localhost:7687",
    graph_password="your-password",
    redis_url="redis://localhost:6379",
    openai_api_key="sk-...",
)
await memory.init()

# Plug into any LangGraph agent:
agent = create_deep_agent(
    model=...,
    tools=[*your_tools, *memory.tools],
    middleware=[memory.middleware],
    checkpointer=memory.checkpointer,
)

# Per request:
result = await agent.ainvoke(
    memory.wrap_input(message, user_id),
    config=memory.make_config(user_id, session_id),
)
memory.save_message(session_id, "user", message)
memory.save_message(session_id, "assistant", result["messages"][-1].content)

# When the user closes the session:
memory.end_session(user_id, session_id)   # background extraction → KG
```

Or build from environment variables:

```python
memory = MemoryOS.from_env()
await memory.init()
```

## What you get

### LangGraph / deepagents — drop-in API

| Property / method        | Purpose                                                   |
|--------------------------|-----------------------------------------------------------|
| `memory.checkpointer`    | LangGraph AsyncRedisSaver (falls back to in-memory)      |
| `memory.middleware`      | Auto-injects user context into every LLM call            |
| `memory.tools`           | `remember_fact`, `recall_facts`, `forget_fact`, `who_do_i_know` |
| `memory.wrap_input()`    | Build the state dict for `agent.ainvoke()`               |
| `memory.make_config()`   | Build the LangGraph config (`thread_id` + `user_id`)     |

### Framework-agnostic API (OpenAI SDK / Assistants / LlamaIndex / AutoGen / custom)

| Method                       | Purpose                                                |
|------------------------------|--------------------------------------------------------|
| `memory.openai_tools()`      | Memory tools as OpenAI tool schemas                    |
| `memory.handle_tool_call()`  | Dispatch a tool call by name with explicit user_id     |
| `memory.inject_context()`    | Prepend memory to a plain `[{"role":..., "content":...}]` list |
| `memory.tool_names()`        | List of tool names for filtering dispatch              |

### Always available

| Method                   | Purpose                                                  |
|--------------------------|----------------------------------------------------------|
| `memory.save_message()`  | Append a turn to STM                                     |
| `memory.end_session()`   | Trigger async KG extraction                              |
| `memory.get_context()`   | Read formatted user context from the graph              |
| `memory.get_extraction_status()` | `'processing'`, `'ready'`, or `'unknown'`        |
| `memory.delete_user_memory()` | Wipe all graph nodes for a user                     |
| `memory.verify_connectivity()` | Health check for graph + Redis                     |

## Use with raw OpenAI SDK (no LangChain)

```python
from openai import OpenAI
from memoryos import MemoryOS

memory = MemoryOS.from_env()
await memory.init()
client = OpenAI()

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": user_input},
]
messages = memory.inject_context(messages, user_id=user_id)   # ← prepend memory

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=memory.openai_tools(),                              # ← OpenAI tool schemas
)
msg = response.choices[0].message

# Dispatch tool calls back through MemoryOS
for tc in msg.tool_calls or []:
    result = memory.handle_tool_call(
        name=tc.function.name,
        args=json.loads(tc.function.arguments),
        user_id=user_id,
    )

memory.save_message(session_id, "user", user_input)
memory.save_message(session_id, "assistant", msg.content or "")
memory.end_session(user_id, session_id)   # on close
```

A complete runnable version is at [`examples/openai_sdk_example.py`](examples/openai_sdk_example.py).

## Configuration

All settings live on `MemoryConfig`:

```python
from memoryos import MemoryConfig, MemoryOS

cfg = MemoryConfig(
    # Graph backend
    graph_backend="neo4j",        # or "memgraph"
    graph_url="bolt://localhost:7687",
    graph_username="neo4j",
    graph_password="...",

    # Redis
    redis_url="redis://localhost:6379",

    # LLM provider for extraction
    llm_provider="openai",        # or "groq"
    openai_api_key="sk-...",
    groq_api_key="gsk_...",
    extraction_model="gpt-4o-mini",

    # Tuning
    max_stm_messages=200,
    stm_ttl_seconds=86400,
    context_cache_ttl=300,
)
memory = MemoryOS(config=cfg)
```

## Architecture

```
 ┌───────────────────────────────────────────────────────┐
 │  Your Agent (deepagents / LangGraph)                  │
 │   ├─ tools = [...your tools, *memory.tools]            │
 │   ├─ middleware = [memory.middleware]                  │
 │   └─ checkpointer = memory.checkpointer                │
 └───────────────────────────────────────────────────────┘
                  │                    │
   ┌──────────────▼────┐   ┌──────────▼─────────────┐
   │  Redis (STM)      │   │  Knowledge Graph (LTM) │
   │  conversation     │   │  Neo4j or Memgraph     │
   │  + checkpoints    │   │  facts, projects,      │
   │                   │   │  people, prefs, etc.   │
   └───────────────────┘   └────────────────────────┘
                                       ▲
                       ┌───────────────┴────────────┐
                       │  Extractor (OpenAI/Groq)   │
                       │  Triggered on session end  │
                       └────────────────────────────┘
```

## Demo

A complete FastAPI + Streamlit reference app lives in `examples/`. See [examples/README.md](examples/README.md) to run it.

## License

MIT
