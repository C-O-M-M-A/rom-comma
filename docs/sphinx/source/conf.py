# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'rom-comma'
copyright = '2023, Robert A. Milton'
author = 'Robert A. Milton'
release = '1.0'

# -- General configuration ---------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.coverage', 'sphinx.ext.napoleon', 'sphinx.ext.autosummary']
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_permalinks_icon = '§'
# html_theme = 'insipid'
html_theme = 'cloud'
html_theme_options = {"max_width": "17.5in"}
html_static_path = ['_static']
# import sphinx_theme
# html_theme = 'stanford_theme'
# html_theme_path = [sphinx_theme.get_html_theme_path('stanford-theme')]
