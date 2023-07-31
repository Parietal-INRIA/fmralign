# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


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
