.PHONY: all
all: main.py
	python main.py

.PHONY: test
test:
	pytest .

.PHONY: clean
clean:
	find . -name '*.pyc' -exec rm --force {} + 
	find . -name '*.pyo' -exec rm --force {} + 
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -f ./data/*.jpg
	rm -f ./data/*.svg
	rm -f ./data/*.mp4


