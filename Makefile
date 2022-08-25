init-venv:
	python3 -m venv venv

init: init-venv  # create virtual env and install deps
	./venv/bin/python3 -m pip install -U pip
	./venv/bin/python3 -m pip install -r requirements/requirements-dev.txt
	./venv/bin/python3 -m pip install -r requirements/requirements.txt
	./venv/bin/python3 -m pip install -e .
	./venv/bin/python3 -m pre_commit install --install-hooks --overwrite
	./venv/vin/python3 -m pip check

lint:  # format all source code
	./venv/bin/isort gestrol/ tests/ setup.py
	.venv/bin/black --config=pyproject.toml --check .
	.venv/bin/flake8 --config=.flake8 --per-file-ignores='tests/*'

test:  # run all tests in project
	./venv/bin/pytest -vvv tests/

clean:  # remove development files
	rm -rf venv
	find . -name __pycache__ | xargs rm -rf
	find . -name *.egg-info | xargs rm -rf
	find . -name .pytest_cache | xargs rm -rf
