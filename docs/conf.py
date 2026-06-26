import os
import sys

# Make the package importable during docs build
sys.path.insert(0, os.path.abspath("../src"))

import qc_executor  # noqa: E402

# -- Project information -----------------------------------------------------
project = "QC Executor"
copyright = "2026, Fraunhofer IPA"
author = "David Kreplin, Moritz Willmann, Marco Roth, Dennis Kleinhans, Florian Wieland"
release = qc_executor.__version__
version = qc_executor.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autosummary_imported_members = False
autosummary_ignore_module_all = True
autodoc_default_options = {
    "members": False,
    "undoc-members": False,
    "show-inheritance": False,
    "imported-members": False,
}
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "alabaster"
# Avoid warnings when local static assets are not present.
_static_dir = os.path.join(os.path.dirname(__file__), "_static")
html_static_path = ["_static"] if os.path.isdir(_static_dir) else []
html_theme_options = {
    "description": "Abstraction layer for quantum circuits and operators across multiple backends.",
    "github_user": "flaqship",
    "github_repo": "qc-executor",
    "github_banner": True,
    "fixed_sidebar": True,
}
