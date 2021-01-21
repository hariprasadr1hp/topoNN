.PHONY: all test clean clean-pyc clean-build clean-jpg clean-svg

all: main.py
	python main.py

clean: clean-pyc clean-build clean-svg clean-mp4

clean-pyc:
	find . -name '*.pyc' -exec rm --force {} + 
	find . -name '*.pyo' -exec rm --force {} + 

clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

clean-svg:
	rm -f ./data/*.svg

clean-mp4:
	rm -f ./data/*.mp4

test:
	pytest