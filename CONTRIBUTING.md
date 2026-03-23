# Contributing to hopfield-memory

## Development Setup

```bash
git clone https://github.com/shahzebqazi/mhn-ai-agent-memory.git
cd mhn-ai-agent-memory
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Branch Conventions

- `feature/description` -- new features
- `fix/description` -- bug fixes
- `docs/description` -- documentation
- `refactor/description` -- restructuring

## Commit Messages

Use conventional commits:

```text
feat(encoders): add nomic-embed-text encoder
fix(tiered): correct eviction order in hot store
docs(readme): update installation instructions
test(network): add capacity scaling benchmarks
```

## Pull Requests

1. Fork and create a branch
2. Write tests for new functionality
3. Ensure `pytest` passes
4. Open a PR with a clear description

## Code Style

- Type hints on all public functions
- Docstrings on all public classes and methods
- Run `ruff check` before committing
