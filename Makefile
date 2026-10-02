.PHONY: setup data train evaluate test api ui demo run docker-up docker-down clean lint format

PYTHON = ./.venv/Scripts/python.exe
PYTEST = ./.venv/Scripts/pytest.exe
STREAMLIT = ./.venv/Scripts/streamlit.exe
UVICORN = ./.venv/Scripts/uvicorn.exe
RUFF = ./.venv/Scripts/ruff.exe
BLACK = ./.venv/Scripts/black.exe

run:
	$(PYTHON) run_all.py

setup:
	py -3.11 -m venv .venv
	./.venv/Scripts/pip.exe install -r requirements.txt

data:
	$(PYTHON) scripts/seed_market_data.py
	$(PYTHON) scripts/generate_catalog_and_report.py

train:
	$(PYTHON) -c "from finsight.tools.financial_tools import FinSightToolRegistry; reg = FinSightToolRegistry(); print(reg.run_quant_models())"

test:
	$(PYTEST) tests/ -v

lint:
	$(RUFF) check src/ tests/

format:
	$(BLACK) src/ tests/ ui/

api:
	$(UVICORN) finsight.api.server:app --host 127.0.0.1 --port 8000 --reload

ui:
	$(STREAMLIT) run ui/app.py --server.port 8501

demo:
	$(PYTHON) -c "from finsight.orchestration.graph import FinSightOrchestrator; orch = FinSightOrchestrator(); state = orch.execute('Analyze Tata Motors for the next 12 months'); print(state.final_memo[:500])"

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

clean:
	rm -rf __pycache__ .pytest_cache runs/ mlruns/
