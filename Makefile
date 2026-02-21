.PHONY: test test-all lint format bench clean

test:
	uv run pytest tests/ -m "not slow and not requires_torch and not requires_cuda" -v

test-all:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/

format:
	uv run ruff format src/ tests/

bench:
	uv run pytest tests/test_performance/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
