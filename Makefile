update-requirements:
	uv lock

sync-venv: update-requirements
	uv sync

init:
	uv venv

	uv pip install -r requirements/requirements-dev.txt
	uv pip install -r requirements/requirements.txt

	uv pip install -e .
	./venv/bin/python3 -m pre_commit install --install-hooks --overwrite

lint:  # lint all source code
	@./venv/bin/ruff check --config=pyproject.toml

test:  # run all tests in project
	@./venv/bin/pytest -vv tests/

clean:  # remove development files
	rm -rf ./venv
	find . -name __pycache__ | xargs rm -rf
	find . -name *.egg-info | xargs rm -rf
	find . -name .pytest_cache | xargs rm -rf
