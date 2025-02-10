# Makefile for the project

# Python Environment

.PHONY : clean_env install_env reinstall_env
reinstall_env :
	clean_env install_env

clean_env :
	pip freeze | xargs pip uninstall -y

install_env :
	python -m pip install --upgrade pip
	pip install pip-tools
	pip-compile requirements.in 
	pip install -r requirements.txt


# Run the application 

.PHONY : train run_app

train :
	python lib/model.py

run_app :
	python app.py
