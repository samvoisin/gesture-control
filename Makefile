init-venv:
	python3 -m venv venv

init: init-venv  # create virtual env and install deps
	./venv/bin/python3 -m pip install -U pip
	./venv/bin/python3 -m pip install -r requirements/requirements-dev.txt
	./venv/bin/python3 -m pip install -r requirements/requirements.txt
	./venv/bin/python3 -m pre_commit install --install-hooks --overwrite
	./venv/bin/python3 -m pip install -e .

lint:  # format all source code
	./venv/vin/isort gestrol/ tests/ setup.py
 	.venv/bin/black --config=pyproject.toml --check .
	.venv/bin/flake8 --config=.flake8 --per-file-ignores='tests/*'

test:  # run all tests in project
	./venv/vin/pytest -vvv tests/
