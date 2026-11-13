# SoongTunes v2 — one-line entry points.  `make help` lists them.
SHELL := /bin/bash
COMPOSE := docker compose
BACKEND := cd backend && set -a && source ../.env && set +a && uv run
EVAL_DATASET ?=
EVAL_ARGS ?=
STRATEGIES ?= hybrid,v1_contextual,v1_stateless,bm25,vector
TAG_VERSION ?= tag-harrier-v2
SONG_VERSION ?= song-harrier-v2-pop5
MIN_POPULARITY ?= 5

.PHONY: help env restore infra up down monitoring quality logs index embed embed-local api web eval test lint

help: ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

env: ## create .env with random passwords (keeps an existing one)
	@test -f .env && echo ".env exists, leaving it" || { \
	  cp .env.example .env; \
	  for k in NEO4J_PASSWORD POSTGRES_PASSWORD LLM_API_KEY EMBED_API_KEY; do \
	    v=$$(openssl rand -base64 24 | tr -d '/+=' | cut -c1-24); sed -i.bak "s|^$$k=.*|$$k=$$v|" .env; done; \
	  rm -f .env.bak; echo "wrote .env — set LLM_BASE_URL/EMBED_BASE_URL"; }

restore: ## restore the v1 dump: make restore DUMP=/path/neo4j.dump
	scripts/restore_neo4j.sh $(DUMP)

infra: ## start Neo4j, Postgres, OpenSearch, Redpanda, Redis
	$(COMPOSE) up -d --build neo4j postgres opensearch redpanda redis

up: ## start everything (infra + api + web) on localhost:3000
	$(COMPOSE) --profile app up -d --build

down: ## stop everything (volumes are kept)
	$(COMPOSE) --profile app down

monitoring: ## Prometheus :9090 + Grafana :3001 + quality checker exporter :9109
	$(COMPOSE) --profile monitoring up -d prometheus grafana
	$(BACKEND) stunes quality serve

quality: ## run data-quality checks once (exit 1 on failure)
	$(BACKEND) stunes quality check

logs: ## follow api/web logs
	$(COMPOSE) --profile app logs -f api web

index: ## derived stores: export graph -> OpenSearch, Neo4j indexes + features
	$(BACKEND) stunes graph-indexes
	$(BACKEND) stunes graph-features
	$(BACKEND) stunes export
	$(BACKEND) stunes index-search

embed: ## embed tags + songs into pgvector and activate (needs an embedder)
	$(BACKEND) stunes backfill --entity tag --version $(TAG_VERSION) --batch 256
	$(BACKEND) stunes activate --version $(TAG_VERSION)
	$(BACKEND) stunes backfill --entity song --version $(SONG_VERSION) --min-popularity $(MIN_POPULARITY)
	$(BACKEND) stunes activate --version $(SONG_VERSION)

embed-local: ## run the local (MPS/CPU) embedding server on :8201
	cd infra/embed-local && set -a && source ../../.env && set +a && \
	  uv run uvicorn server:app --host 127.0.0.1 --port 8201

api: ## run the API on the host with reload (:8000)
	$(BACKEND) uvicorn stunes.api:app --reload --port 8000

web: ## run the web app in dev mode (:3000)
	cd web && API_INTERNAL_URL=http://localhost:8000 npm run dev

eval: ## replay the eval set: make eval EVAL_DATASET=... [STRATEGIES=...] [EVAL_ARGS=...]
	@test -n "$(EVAL_DATASET)" || { echo "set EVAL_DATASET=/path/to/evaluation_dataset.csv"; exit 1; }
	$(BACKEND) stunes eval --dataset $(EVAL_DATASET) --strategies $(STRATEGIES) $(EVAL_ARGS)

test: ## unit tests
	cd backend && uv run pytest -q

lint: ## ruff + tsc + eslint
	cd backend && uv run ruff check src tests && uv run ruff format --check src tests
	cd web && npx tsc --noEmit && npx eslint .

