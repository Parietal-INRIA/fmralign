# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

import os
import sys
from pathlib import Path

import fmralign

# If extensions (or modules to document with autodoc) are in another
# directory, add these directories to sys.path here. If the directory
# is relative to the documentation root, use os.path.abspath to make it
# absolute, like shown here.
sys.path.insert(0, os.path.abspath("sphinxext"))
from github_link import make_linkcode_resolve

# We also add the directory just above to enable local imports of fmralign
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

# General information about the project.
project = "fmralign"
copyright = "fmralign developers"

# The full current version, including alpha/beta/rc tags.
current_version = fmralign.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.imgmath",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinxext.opengraph",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
    "sphinx_copybutton",
    "sphinx_design",
    "gh_substitutions",
    "myst_parser",
    "numpydoc",
]


autosummary_generate = True

autodoc_typehints = "none"

autodoc_default_options = {
    "imported-members": True,
    "inherited-members": True,
    "undoc-members": True,
    "member-order": "bysource",
    #  We cannot have __init__: it causes duplicated entries
    #  'special-members': '__init__',
}

# Get rid of spurious warnings due to some interaction between
# autosummary and numpydoc. See
# https://github.com/phn/pytpm/issues/3#issuecomment-12133978 for more
# details
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".md"]

# The encoding of source files.
# source_encoding = 'utf-8'

# Generate the plots for the gallery
plot_gallery = "True"

# The master toctree document.
master_doc = "index"

# sphinxcontrib-bibtex
bibtex_bibfiles = ["./references.bib", "./soft_references.bib"]
bibtex_style = "unsrt"
bibtex_reference_style = "author_year"
bibtex_footbibliography_header = ""

# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "tune_toc.rst",
    "includes/big_toc_css.rst",
    "includes/bigger_toc_css.rst",
]

exclude_trees = ["_build", "templates", "includes"]

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'friendly'
# pygments_style = 'manni'
pygments_style = "sas"
pygments_dark_style = "stata-dark"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  Major themes that come with
# Sphinx are currently 'default' and 'sphinxdoc'.
html_theme = "furo"

# Add custom css instructions from themes/custom.css
font_awesome = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/"
html_css_files = [
    "custom.css",
    (
        "https://cdnjs.cloudflare.com/ajax/libs/"
        "font-awesome/5.15.4/css/all.min.css"
    ),
    f"{font_awesome}fontawesome.min.css",
    f"{font_awesome}solid.min.css",
    f"{font_awesome}brands.min.css",
]

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
# html_style = "nature.css"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "dark_css_variables": {
        "color-announcement-background": "#935610",
        "color-announcement-text": "#FFFFFF",
    },
    "light_css_variables": {
        "admonition-font-size": "100%",
        "admonition-title-font-size": "100%",
        "color-announcement-background": "#FBB360",
        "color-announcement-text": "#111418",
        "color-admonition-title--note": "#448aff",
        "color-admonition-title-background--note": "#448aff10",
    },
    "source_repository": "https://github.com/nilearn/nilearn/",
    "source_branch": "main",
    "source_directory": "doc/",
}

# Add any paths that contain custom themes here, relative to this directory.
# html_theme_path = ["themes"]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "fmralign"

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "fmralign"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = ""

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "logos/favicon.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["images", "themes"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# template names.
# html_additional_pages = {}

# If false, no module index is generated.
html_use_modindex = False

# If false, no index is generated.
html_use_index = False

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = ''

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "PythonScientific"

# Sphinx copybutton config
copybutton_prompt_text = ">>> "

# -- Extension configuration -------------------------------------------------
numpydoc_show_class_members = False

_python_doc_base = "http://docs.python.org/3.9"

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": (_python_doc_base, None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://scipy.github.io/devdocs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "nibabel": ("https://nipy.org/nibabel", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
}

extlinks = {
    "sklearn": ("https://scikit-learn.org/stable/%s", None),
    "inria": ("https://team.inria.fr/%s", None),
    "nilearn": ("https://nilearn.github.io/stable/%s", None),
    "nipy": ("https://nipy.org/%s", None),
}

# Check intersphinx reference targets exist
nitpicky = True
# Temporary solution to nilearn/nilearn#3997
nitpick_ignore = [
    ("py:class", "sklearn.utils.metadata_routing.MetadataRequest"),
]

binder_branch = "main" if "dev" in current_version else current_version

sphinx_gallery_conf = {
    "doc_module": "nilearn",
    "backreferences_dir": Path("modules", "generated"),
    "reference_url": {"nilearn": None},
    "junit": "../test-results/sphinx-gallery/junit.xml",
    "examples_dirs": "../examples/",
    "gallery_dirs": "auto_examples",
    # Ignore the function signature leftover by joblib
    "ignore_pattern": r"func_code\.py",
    # "show_memory": not sys.platform.startswith("win"),
    "remove_config_comments": True,
    "matplotlib_animations": True,
    "nested_sections": True,
    "binder": {
        "org": "nilearn",
        "repo": "nilearn",
        "binderhub_url": "https://mybinder.org",
        "branch": binder_branch,
        "dependencies": "./binder/requirements.txt",
        "use_jupyter_lab": True,
    },
    "default_thumb_file": "logos/nilearn-desaturate-100.png",
}


def touch_example_backreferences(
    app,
    what,  # noqa: ARG001
    name,
    obj,  # noqa: ARG001
    options,  # noqa: ARG001
    lines,  # noqa: ARG001
):
    # generate empty examples files, so that we don't get
    # inclusion errors if there are no examples for a class / module
    examples_path = Path(
        app.srcdir, "modules", "generated", f"{name}.examples"
    )
    if not examples_path.exists():
        examples_path.touch()


def setup(app):
    app.connect("autodoc-process-docstring", touch_example_backreferences)


# The following is used by sphinx.ext.linkcode to provide links to github
linkcode_resolve = make_linkcode_resolve(
    "nilearn",
    "https://github.com/nilearn/"
    "nilearn/blob/{revision}/"
    "{package}/{path}#L{lineno}",
)

# -- sphinxext.opengraph configuration -------------------------------------
ogp_site_url = "https://nilearn.github.io/"
ogp_image = "https://nilearn.github.io/_static/nilearn-logo.png"
ogp_use_first_image = True
ogp_site_name = "Nilearn"
