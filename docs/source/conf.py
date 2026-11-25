import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Mixing Rotation'
copyright = '2026, Neal P. Avis Kozar, Christopher Cappiello, Melissa D. Diamond, Aaron C. Vincent'
author = 'Neal P. Avis Kozar, Christopher Cappiello, Melissa D. Diamond, Aaron C. Vincent'
release = '0.9.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]
autosummary_generate = True
autosummary_generate_overwrite = True
intersphinx_mapping = {
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
# # Can swap to imgmath to make use of a local latex extension instead of mathjax on the web
# extensions.append("sphinx.ext.imgmath"), extensions.remove("sphinx.ext.mathjax")
# imgmath_image_format = "svg"
# imgmath_latex = "latex"

templates_path = ['_templates']
exclude_patterns = ['Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'classic'
