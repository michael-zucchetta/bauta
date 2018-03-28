.PHONY: test
.PHONY: install

help:
	@echo "make test    -- runs tests"
	@echo "make install -- installs bauta"
	@echo

install:
	./install.sh

dataset:
	./setup_dataset.py

test:
	 coverage run --source bauta/  -m unittest discover -s .  -p 'Test*.py' && \
	 coverage report
