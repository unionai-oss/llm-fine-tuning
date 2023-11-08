install-piptools:
	pip install pip-tools

requirements.txt: install-piptools
	pip-compile requirements.in --upgrade --verbose --resolver=backtracking --annotation-style=line
