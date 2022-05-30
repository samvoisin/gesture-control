init-venv:
	python3 -m venv venv

init: init-venv  # create virtual env and install deps
	./venv/bin/python3 -m pip install -U pip
	./venv/bin/python3 -m pip install -r requirements/requirements-dev.txt
	./venv/bin/python3 -m pre_commit install --install-hooks --overwrite
