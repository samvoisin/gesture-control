update-requirements:
	@./venv/bin/uv pip compile setup.py -o requirements/requirements.txt
	@./venv/bin/uv pip compile requirements/requirements-dev.in -o requirements/requirements-dev.txt

upgrade-requirements:
	@./venv/bin/uv pip compile --upgrade setup.py -o requirements/requirements.txt
	@./venv/bin/uv pip compile --upgrade requirements/requirements-dev.in -o requirements/requirements-dev.txt

sync-venv: update-requirements
	@./venv/bin/uv pip sync requirements/requirements.txt requirements/requirements-dev.txt
	@./venv/bin/uv pip install -e .

# create virtual env and install deps
init:
	@python3 -m venv venv
	@./venv/bin/python3 -m pip install -U pip

	@./venv/bin/python3 -m pip install uv

	@./venv/bin/uv pip install -r requirements/requirements-dev.txt
	@./venv/bin/uv pip install -r requirements/requirements.txt

	@./venv/bin/uv pip install -e .
	@./venv/bin/python3 -m pre_commit install --install-hooks --overwrite

lint:  # lint all source code
	@./venv/bin/ruff check --config=pyproject.toml

test:  # run all tests in project
	@./venv/bin/pytest -vv --cov-fail-under=80 --cov=./gesturemote --cov-report=term --cov-report=xml tests/

clean:  # remove development files
	rm -rf ./venv
	find . -name __pycache__ | xargs rm -rf
	find . -name *.egg-info | xargs rm -rf
	find . -name .pytest_cache | xargs rm -rf
