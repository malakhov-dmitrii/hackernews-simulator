# Contributing to hackernews-simulator

## Quick Start

1. Fork the repo
2. Clone: `git clone https://github.com/YOUR_USERNAME/hackernews-simulator`
3. Install: `uv sync --all-extras`
4. Test: `uv run pytest -k "not slow"`
5. Create branch: `git checkout -b feat/my-feature`
6. Make changes, add tests
7. Submit PR

## Code Style
- Python 3.11+
- Keep it simple and readable
- Add tests for new functionality
- Use existing patterns (check conftest.py for fixtures)

## Running Tests
```bash
uv run pytest                    # all tests
uv run pytest -k "not slow"     # skip slow tests (no network/real models)
uv run pytest tests/test_X.py   # specific test file
```
