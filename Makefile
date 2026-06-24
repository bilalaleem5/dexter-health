.PHONY: run test tick eval api

run:
	python -m src.run --letters letters --data data --out proposals.json

test:
	pytest -q

tick:
	python -m src.tick --data data --advance-days 3

eval:
	python eval/run_eval.py

api:
	uvicorn src.api.app:app --reload
