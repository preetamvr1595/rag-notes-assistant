# Contributing to RAG Notes Assistant

Thank you for your interest in contributing!

## Development Workflow
1. Fork and clone the repository.
2. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows).
3. Install dependencies: `pip install -r requirements-dev.txt`.
4. Run tests: `pytest tests/`.
5. Ensure code passes linting before submitting a Pull Request.

## Coding Standards
- Follow PEP 8 guidelines.
- Include docstrings for new public functions.
- Add tests under `tests/` for any new retrieval or chunking logic.
