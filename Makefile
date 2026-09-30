PYTHON ?= .venv/bin/python

.PHONY: test m0 m1 m2 m3 m4 m5 m6 m7 m8

test:
	$(PYTHON) -m pytest -q

m0 m1 m2 m3 m4 m5 m6 m7 m8:
	$(PYTHON) scripts/run_$@.py
