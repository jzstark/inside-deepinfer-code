.PHONY: labs check test

DEEPINFER_SOURCE ?= ../DeepInfer

labs:
	python3 labs/run_all.py

check:
	python3 tools/check_sources.py "$(DEEPINFER_SOURCE)"

test:
	python3 -m unittest tests.test_run_all
