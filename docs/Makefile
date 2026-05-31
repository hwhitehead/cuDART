# Makefile for Sphinx documentation

SPHINX_BUILD = sphinx-build
SOURCE_DIR = source
BUILD_DIR = build

.PHONY: help Makefile

%: Makefile
	@$(SPHINX_BUILD) -M $@ "$(SOURCE_DIR)" "$(BUILD_DIR)"

clean:
	rm -r doctrees/ html/