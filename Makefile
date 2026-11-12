.PHONY: help install test lint run index clean

help:
	@echo "Available commands:"
	@echo "  install   Install runtime dependencies"
	@echo "  test      Run test suite with pytest"
	@echo "  lint      Check formatting with ruff"
	@echo "  index     Build inverted index from data/notes"
	@echo "  run       Start Streamlit web application"

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v

lint:
	ruff check .

index:
	python -m rag_notes.cli build

run:
	streamlit run app/streamlit_app.py
