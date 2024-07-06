update-requirements:
	@./venv/bin/pip-compile setup.py --resolver=backtracking --output-file=./requirements/requirements.txt
	@./venv/bin/pip-compile ./requirements/requirements-dev.in --output-file=./requirements/requirements-dev.txt

upgrade-requirements:
	@./venv/bin/pip-compile --upgrade setup.py --resolver=backtracking --output-file=./requirements/requirements.txt
	@./venv/bin/pip-compile --upgrade ./requirements/requirements-dev.in --output-file=./requirements/requirements-dev.txt

sync-venv: update-requirements
	@./venv/bin/pip-sync ./requirements/requirements.txt ./requirements/requirements-dev.txt
	@./venv/bin/pip install -e .

init:  # create virtual env and install deps
	@python3 -m venv venv
	@./venv/bin/python3 -m pip install -U pip

	@./venv/bin/python3 -m pip install -r requirements/requirements-dev.txt
	@./venv/bin/python3 -m pip install -r requirements/requirements.txt

	@./venv/bin/python3 -m pip install -e .
	@./venv/bin/python3 -m pre_commit install --install-hooks --overwrite

lint:  # format all source code
	@./venv/bin/ruff format --config=pyproject.toml .

test:  # run all tests in project
	@./venv/bin/pytest -vv --cov-fail-under=80 --cov=./gesturemote --cov-report=term --cov-report=xml tests/

clean:  # remove development files
	rm -rf venv
	find . -name __pycache__ | xargs rm -rf
	find . -name *.egg-info | xargs rm -rf
	find . -name .pytest_cache | xargs rm -rf
