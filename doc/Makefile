# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    = -v
SPHINXBUILD   = sphinx-build
BUILDDIR      = _build

# Internal variables
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(SPHINXOPTS) .

.PHONY: help html dirhtml htmlhelp

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"
	@echo "  dirhtml   to make HTML files named index.html in directories"
	@echo "  htmlhelp  to make HTML files and a HTML help project"


force_html: force html

force:
	find . -name \*.rst -exec touch {} \;

html:
	# These two lines make the build a bit more lengthy, and the
	# the embedding of images more robust
	rm -rf $(BUILDDIR)/html/_images
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	touch $(BUILDDIR)/html/.nojekyll
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

html-strict:
	# Build html documentation using a strict mode: Warnings are
	# considered as errors.
	make check
	touch $(BUILDDIR)/html/.nojekyll
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml:
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	touch $(BUILDDIR)/dirhtml .nojekyll
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

htmlhelp:
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

install:
	git clone --no-checkout --depth 1 https://github.com/Parietal-INRIA/fmralign.github.io.git build/fmralign.github.io
	touch build/fmralign.github.io/.nojekyll
	make html
	cd build/ && \
	cp -r html/* fmralign.github.io && \
	cd fmralign.github.io && \
	git add * && \
	git add .nojekyll && \
	git commit -a -m 'Make install' && \
	git push
