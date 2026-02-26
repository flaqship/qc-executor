import os
import sys

# Make the package importable during docs build
sys.path.insert(0, os.path.abspath("../src"))

import executor  # noqa: E402

# -- Project information -----------------------------------------------------
project = "executor"
copyright = "2024, Fraunhofer IPA"
author = "David Kreplin, Moritz Willmann, Jan Schnabel, Manuel Hagelüken, Marco Roth"
release = executor.__version__
version = executor.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "alabaster"
html_static_path = ["_static"]
html_theme_options = {
    "description": "Abstraction layer for quantum circuits and operators across multiple backends.",
    "github_user": "flaqship",
    "github_repo": "Executor",
    "github_banner": True,
    "fixed_sidebar": True,
}
