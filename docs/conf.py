# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'DW-NCA'
copyright = '2024, UBANDIYA NAJIB YUSUF'
author = 'UBANDIYA NAJIB YUSUF'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']
