.PHONY: help build run dev dashboard install setup logs stop restart shell lint clean

help:
	@echo ""
	@echo "XAUUSD Trading Intelligence System"
	@echo "==================================="
	@echo ""
	@echo "  make setup      First-time setup: copy .env.example -> .env"
	@echo "  make build      Build the Docker image"
	@echo "  make run        Run dashboard in Docker (http://localhost:8501)"
	@echo "  make dev        Run dashboard locally without Docker"
	@echo "  make dashboard  Alias for dev"
	@echo "  make install    Install Python dependencies locally"
	@echo "  make logs       Tail container logs"
	@echo "  make stop       Stop the running container"
	@echo "  make restart    Rebuild image and restart dashboard"
	@echo "  make shell      Open a shell inside the container"
	@echo "  make lint       Syntax-check all .py files"
	@echo "  make clean      Remove cache, .pyc, and build artefacts"
	@echo ""

# ── First-time setup ──────────────────────────────────────────────────────────

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env — open it and add your GEMINI_API_KEY before running"; \
	else \
		echo ".env already exists"; \
	fi
	@mkdir -p data

# ── Docker targets ─────────────────────────────────────────────────────────────

build: setup
	docker compose build

run: setup
	docker compose up trade-agent

logs:
	docker compose logs -f trade-agent

stop:
	docker compose down

restart: stop
	docker compose build
	docker compose up trade-agent

shell: setup
	docker compose run --rm --entrypoint bash trade-agent

# ── Local dev targets ──────────────────────────────────────────────────────────

install: setup
	pip3 install --no-cache-dir --prefer-binary -r requirements.txt

dev: setup install
	python3 -m streamlit run dashboard.py --server.port=8501

dashboard: dev

# ── Utility targets ────────────────────────────────────────────────────────────

lint:
	@echo "Checking Python syntax..."
	@for f in *.py; do python -m py_compile "$$f" && echo "  OK: $$f"; done
	@echo "All files passed"

clean:
	rm -rf .cache __pycache__ data
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -f *.db latest_analysis.json
	@echo "Clean complete"
