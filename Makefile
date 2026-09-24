# Shortcuts for the things you run more than twice.
# Windows: these work from Git Bash, or just copy the command out.

.PHONY: install install-train install-quant dev api web test lint fmt run plan clean

install:          ## API + dry-run pipeline + dev tools
	pip install -e ".[dev]"
	npm install --prefix web

install-train:    ## add torch, transformers, peft (multi-GB)
	pip install -e ".[train]"

install-quant:    ## add bitsandbytes for the 4-bit QLoRA path (needs CUDA)
	pip install -e ".[quant]"

api:              ## backend on :8000
	uvicorn finetune_lab.api.app:app --reload --port 8000

web:              ## frontend on :5173
	npm run dev --prefix web

test:
	pytest

lint:
	ruff check src tests
	ruff format --check src tests

fmt:
	ruff check src tests --fix
	ruff format src tests

run:              ## dry run from the terminal, no browser needed
	finetune-lab run

plan:             ## what a real run would do on this machine
	finetune-lab plan

clean:
	rm -rf artifacts .pytest_cache .ruff_cache .coverage htmlcov
