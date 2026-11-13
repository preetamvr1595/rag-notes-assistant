# Changelog

All notable changes to this project will be documented in this file.

## [0.3.0] - 2026-11-13
### Added
- FastAPI REST API endpoints (`/ask`, `/search`, `/health`, `/metrics`).
- Docker containerization with multi-service `docker-compose.yml`.
- PyTorch deep learning and Git/Docker study notes corpus.
- CI workflows for Ruff linting and Docker build verification.
- Automated benchmark runner and refusal gate confusion matrix scripts.
- Extended evaluation dataset `eval/golden_extended.json`.

## [0.2.0] - 2026-11-12
### Added
- Domain models in `rag_notes/types.py` (`Passage`, `SearchResult`, `Citation`, `AnswerResult`).
- Configuration dataclass `rag_notes/config.py`.
- Custom exceptions in `rag_notes/exceptions.py`.
- Granular test suites for tokenization, chunker, loaders, and evaluation metrics.
- CSS stylesheet and UI components for Streamlit app.
- SQL and Advanced Python study notes corpus.
- Latency and index profiling scripts.

## [0.1.0] - 2026-11-11
### Added
- Initial release of RAG Notes Assistant.
- Heading-aware chunker and sparse BM25 + character n-gram fusion.
- Out-of-scope refusal gate.
- Baseline evaluation benchmark with 62 in-scope and 24 out-of-scope questions.
