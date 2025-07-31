# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from cifutils import __version__

project = "cifutils"
copyright = "2025, bakerlab"
author = "bakerlab"

# Clean the version for documentation display (remove +dev and -dirty)
raw_version = str(__version__)
# Extract just the base version (e.g., "2.29.0" from "2.29.0+dev26.ad450d1-dirty")
clean_version = raw_version.split("+")[0].split("-")[0]
version = clean_version.replace("v", "")


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate docs from docstrings
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    # "nbsphinx",  # Jupyter notebook support
    "sphinx_gallery.gen_gallery",  # Generates auto_examples/ from examples/
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "show_nav_level": 1,
    "collapse_navigation": False,
    "navigation_depth": -1,  # Unlimited depth
    "globaltoc_collapse": False,
    "globaltoc_includehidden": True,
    "globaltoc_maxdepth": -1,  # Unlimited depth
    "header_links_before_dropdown": 8,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "logo": {
        "text": "cifutils",
        "image_light": "_static/ipd_logo_small.svg",
        "image_dark": "_static/ipd_logo_small.svg",
    },
    "navbar_start": ["navbar-logo", "version-switcher"],
    "switcher": {
        "json_url": "https://baker-laboratory.github.io/cifutils/latest/_static/switcher.json",
        "version_match": version,
    },
}

sphinx_gallery_conf = {
    "examples_dirs": "examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # where to put the generated gallery
    "image_scrapers": ("matplotlib",),
    "thumbnail_size": (350, 350),
    "default_thumb_file": "_static/default_thumbnail.png",
}
