update-requirements:
	@uv pip compile setup.py -o requirements/requirements.txt
	@uv pip compile setup.py -o requirements/requirements-dev.txt

upgrade-requirements:
	@uv pip compile --upgrade setup.py -o requirements/requirements.txt
	@uv pip compile --upgrade requirements/requirements-dev.in -o requirements/requirements-dev.txt

sync-venv: update-requirements
	@uv pip sync requirements/requirements.txt requirements/requirements-dev.txt
	@uv pip install -e .

# create virtual env and install deps
# assumes uv installed on local machine
init:
	@uv venv

	@uv pip install -r requirements/requirements-dev.txt
	@uv pip install -r requirements/requirements.txt

	@uv pip install -e .
	@.venv/bin/python3 -m pre_commit install --install-hooks --overwrite

lint:  # lint all source code
	@.venv/bin/ruff check --config=pyproject.toml

test:  # run all tests in project
	@.venv/bin/pytest -vv --cov-fail-under=80 --cov=./gesturemote --cov-report=term --cov-report=xml tests/

clean:  # remove development files
	rm -rf .venv
	find . -name __pycache__ | xargs rm -rf
	find . -name *.egg-info | xargs rm -rf
	find . -name .pytest_cache | xargs rm -rf
