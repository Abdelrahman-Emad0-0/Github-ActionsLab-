# TODO: Add common commands for students.
# Suggested targets:
# - setup: install dependencies
# - test: run tests
# - lint: run lint checks
# - clean: remove generated files

#C:/venvs/bigdata/Scripts/python.exe

PYTHON = python
CONFIG = configs/config.toml

.PHONY: help setup validate clean_data features train classify report pipeline test format check lint isort clean

setup:
	@echo "TODO: Install dependencies."
	pip install -e ".[dev]"

validate:
	$(PYTHON) src/data/validate.py --config $(CONFIG) \
	    --input data/raw/Teen_Mental_Health_Dataset.csv \
	    --output reports/validation_raw.json

clean_data:
	$(PYTHON) src/data/preprocess.py --config $(CONFIG)

features:
	$(PYTHON) src/features/engineer.py --config $(CONFIG)

train:
	$(PYTHON) src/models/train.py --config $(CONFIG)

classify:
	$(PYTHON) src/models/classify.py --config $(CONFIG)

report:
	$(PYTHON) src/reports/generate_report.py --config $(CONFIG)

pipeline:
	$(PYTHON) src/data/validate.py --config $(CONFIG) \
	    --input data/raw/Teen_Mental_Health_Dataset.csv \
	    --output reports/validation_raw.json
	$(PYTHON) src/data/preprocess.py --config $(CONFIG)
	$(PYTHON) src/data/validate.py --config $(CONFIG) \
	    --input data/processed/cleaned.csv \
	    --output reports/validation_cleaned.json
	$(PYTHON) src/features/engineer.py --config $(CONFIG)
	$(PYTHON) src/models/train.py --config $(CONFIG)
	$(PYTHON) src/models/classify.py --config $(CONFIG)
	$(PYTHON) src/reports/generate_report.py --config $(CONFIG)

test:
	@echo "TODO: Run tests."
	$(PYTHON) -m pytest tests/ -v

lint:
	@echo "TODO: Run linting/format checks."
	ruff check src/ tests/

clean:
	@echo "TODO: Remove generated files."
	rm -rf data/processed/ models/ reports/ __pycache__ .pytest_cache

format:
	black src/ tests/

check:
	black --check src/ tests/

isort:
	isort src/ tests/
