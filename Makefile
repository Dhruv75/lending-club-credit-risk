PY ?= python
DATA ?= data/raw/accepted_2007_to_2018Q4.csv.gz

.PHONY: install demo train test lint app clean

install:            ## install the package with app + dev extras
	$(PY) -m pip install -e ".[app,dev]"

demo:               ## synthetic end-to-end run (writes to runs/demo/)
	$(PY) -m creditrisk.pipeline --demo

train:              ## real run on the Kaggle file (writes to models/ and reports/)
	$(PY) -m creditrisk.pipeline --data $(DATA)

test:
	$(PY) -m pytest

lint:
	ruff check src tests app scripts
	ruff format --check src tests app scripts

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf runs data/processed/*.parquet data/processed/*.json
