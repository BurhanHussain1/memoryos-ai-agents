# Changelog

All notable changes to this project will be documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-02

Initial public release.

### Added
- `MemoryOS` core class with checkpointer, middleware, tools, and lifecycle management
- Neo4j and Memgraph backends via `GRAPH_BACKEND` env var
- LLM providers: OpenAI, Groq, Anthropic, Gemini, Ollama, LiteLLM (100+ models), and a `custom` callable for any provider
- Framework-agnostic helpers: `openai_tools()`, `handle_tool_call()`, `inject_context()`, `tool_names()`
- LangGraph-native helpers: `wrap_input()`, `make_config()`, `MemoryInjectorMiddleware`
- Memory tools auto-inject `user_id` from runtime config — LLM cannot pass wrong user
- Bridge period for cross-session continuity while extraction runs in background
- Public properties for all core components: `connection`, `store`, `reader`, `writer`, `extractor`, `config`
- Demo app under `examples/` (FastAPI + Streamlit + deepagents)
- Reference example using raw OpenAI SDK (no LangChain)
- CI on Python 3.11 and 3.12
- Optional pip extras for each provider: `[openai]`, `[groq]`, `[anthropic]`, `[gemini]`, `[ollama]`, `[litellm]`, `[all]`
