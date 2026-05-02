# Contributing to memoryos

Thanks for considering a contribution. This project is in early days and
the API is still evolving — small, focused PRs are easiest to review.

## Quick start

```bash
git clone https://github.com/<your-fork>/memoryos
cd memoryos
python -m venv venv && source venv/bin/activate    # or venv\Scripts\activate on Windows
pip install -e ".[all]"
pip install ruff pytest pytest-asyncio
```

## Running tests

```bash
# Unit tests — no external services needed
pytest tests/unit/

# Integration tests — requires Neo4j + Redis running locally
docker compose -f examples/docker-compose.yml up -d
MEMORYOS_RUN_INTEGRATION=1 pytest tests/integration/
```

## Linting

```bash
ruff check memoryos/
ruff format memoryos/
```

CI runs `ruff check` on every push.

## Branches and commits

- Branch from `main`
- Use clear, imperative commit messages: `add Anthropic extraction provider`
- One PR ≠ one commit, but please squash trivial fixup commits before opening the PR
- Update `CHANGELOG.md` under the "Unreleased" section in the same PR

## What we welcome

- New LLM providers (the pattern in `memoryos/graph/extractor.py` is straightforward)
- Better error messages and recovery
- Tests for paths that aren't covered yet
- Documentation improvements
- Support for more graph backends (e.g. ArangoDB, JanusGraph)
- Adapters for more agent frameworks (LlamaIndex, AutoGen, CrewAI)

## What needs discussion first (open an issue)

- Changes to `MemoryOS` public API
- Changes to the graph schema (`memoryos/graph/schema.py`)
- Anything that breaks existing users

## Releasing (maintainers)

1. Bump version in `pyproject.toml` and `memoryos/__init__.py`
2. Move "Unreleased" section in `CHANGELOG.md` under the new version + date
3. Tag: `git tag v0.x.y && git push --tags`
4. GitHub Releases → publish a release on that tag → `publish.yml` ships to PyPI via OIDC

## Code of conduct

Be kind. Assume good faith. Disagree with ideas, not people.
