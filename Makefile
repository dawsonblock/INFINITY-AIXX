PY=python

.PHONY: install test lint format typecheck smoke eval

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

format:
	black .

typecheck:
	mypy src

smoke:
	$(PY) -m infinity_dual_hybrid.train_cli --env CartPole-v1 --total-timesteps 2000 --seed 123 --run-id smoke

eval:
	$(PY) -m infinity_dual_hybrid.eval --env CartPole-v1 --episodes 10 --checkpoint checkpoints/smoke/latest.pt
