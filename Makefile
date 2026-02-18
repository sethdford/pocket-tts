# pocket-tts — Top-level Makefile

.PHONY: test lint

test:
	uv run pytest -n 3 -v

lint:
	uvx pre-commit run --all-files
