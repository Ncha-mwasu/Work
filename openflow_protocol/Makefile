build: clean prepare
	python3 setup.py sdist
	ls -l dist/

clean:
	rm -rf build/ dist/ *.egg-info/

prepare:
	pip3 install --upgrade pip setuptools wheel twine

testupload: build
	twine upload -r pypitest dist/*

upload: build
	twine upload dist/*
