# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cuDART'
copyright = '2026, Henry Whitehead'
author = 'Henry Whitehead'
release = '0.9'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc"
]

templates_path = ['_templates']
exclude_patterns = []
add_module_names = False




# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_favicon = '_static/favicon.svg'

# auto-doc
import sys, os
from pathlib import Path
BASE_DIR = Path(__file__).parents[2]
sys.path.append(os.path.join(BASE_DIR, "pysrc"))