# Makefile for the project

# Python Environment

.PHONY : clean_env install_env reinstall_env install_precommit_hooks
reinstall_env :
	clean_env install_env

clean_env :
	pip freeze | xargs pip uninstall -y

install_env :
	python -m pip install --upgrade pip
	pip install pip-tools
	pip-compile requirements.in 
	pip install -r requirements.txt

install_precommit_hooks :
	@echo "Installing pre-commit hooks ..."
	@pip install pre-commit
	@pre-commit install
	@echo "✅ Pre-commit hooks installed"

# Run the application 

.PHONY : train run_app

train :
	python lib/model.py

run_app :
	python app.py
