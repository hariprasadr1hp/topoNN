.PHONY: all test clean clean-pyc clean-build clean-jpg clean-svg

all: main.py
	python main.py

test: test-meshGen test-neuralNet test-solveFE2D

test-meshGen:
	pytest tests/test_meshGen.py -v

test-neuralNet:
	pytest tests/test_neuralNet.py -v

test-solveFE2D:
	pytest tests/test_solveFE2D.py -v

test-solveFE3D:
	pytest tests/test_solveFE3D.py -v


clean: clean-pyc clean-build clean-svg clean-mp4

clean-pyc:
	find . -name '*.pyc' -exec rm --force {} + 
	find . -name '*.pyo' -exec rm --force {} + 

clean-build:
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

clean-jpg:
	rm -f ./data/*.jpg

clean-svg:
	rm -f ./data/*.svg

clean-mp4:
	rm -f ./data/*.mp4


