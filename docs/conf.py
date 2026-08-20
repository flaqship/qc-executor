import os
import sys

# Make the package importable during docs build
sys.path.insert(0, os.path.abspath("../src"))

import qc_executor  # noqa: E402


def _build_plugin_dependencies():
    """Map each plugin package to the dependencies it needs, read from pyproject.

    Keyed by the plugin's import path (e.g. ``qc_executor.qiskit``) so the
    shared autosummary ``plugin`` template can list them per plugin. Each value
    is ``{"extra": <pip extra>, "requires": [<requirement strings>]}``.
    """
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - older docs builders
        return {}

    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path, "rb") as fh:
        meta = tomllib.load(fh)

    project = meta.get("project", {})
    core_deps = project.get("dependencies", [])
    optional_deps = project.get("optional-dependencies", {})

    # Backend name -> pip extra that provides its full feature set.
    backend_extra = {
        "qiskit": "qiskit-full",
        "pennylane": "pennylane",
        "qulacs": "qulacs",
        "pauli_propagation": "pauli_propagation",
    }

    mapping = {}
    for backend, extra in backend_extra.items():
        keyword = backend.split("_")[0]
        # Core dependencies that belong to this backend (e.g. qiskit ships in core).
        requires = [dep for dep in core_deps if keyword in dep.lower()]
        requires += optional_deps.get(extra, [])
        mapping[f"qc_executor.{backend}"] = {"extra": extra, "requires": requires}
    return mapping

# -- Project information -----------------------------------------------------
project = "QC Executor"
copyright = "2026, Fraunhofer IPA"
author = "Moritz Willmann, Dennis Kleinhans, Florian Wieland, David Kreplin"
release = qc_executor.__version__
version = qc_executor.__version__

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# -- MyST (Markdown) ---------------------------------------------------------
# Allows the README and other Markdown files to be rendered by Sphinx.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

autosummary_generate = True
autosummary_imported_members = False
# Respect each plugin package's ``__all__`` so the shared plugin template
# documents exactly the public classes the package exports.
autosummary_ignore_module_all = False
# Extra variables made available to the autosummary templates (used by the
# shared ``plugin`` template to list each plugin's dependencies).
autosummary_context = {"plugin_dependencies": _build_plugin_dependencies()}
# Documented members are rendered for the curated ``autoclass`` directives in
# the API reference. Individual directives may narrow this with an explicit
# ``:members:`` list. Undocumented and imported members stay hidden to keep the
# generated pages focused on the public API.
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "imported-members": False,
}
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autoclass_content = "class"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "flaqship"
# Avoid warnings when local static assets are not present.
_static_dir = os.path.join(os.path.dirname(__file__), "_static")
html_static_path = ["_static"] if os.path.isdir(_static_dir) else []
html_theme_options = {
    # Brand and repository settings
    "source_repo": "https://github.com/flaqship/qc-executor",
    "source_branch": "main",
    "source_directory": "docs/",
    
    # Logo configuration
    "logo": "_static/logo.svg",              # Custom package logo (optional)
    "logo_dark": "_static/logo-dark.svg",    # Dark theme variant (optional)
    
    # Display options
    "show_source_link": True,
    "sidebar_hide_name": False,
    # Depth of the left-sidebar navigation. 2 = pages/sections only,
    # 3 = also list classes, 4 = also list individual methods/functions.
    "navigation_depth": 3,

    # Default theme mode: "light", "dark", or "auto"
    "default_mode": "auto",
}
html_favicon = '_static/favicon.png'

# -- Versioned documentation (flaQship theme) --------------------------------
# Build a multi-version site with the standard sphinx-build: point each release
# at its own directory under a shared root and the theme regenerates
# versions.json + a root redirect and shows the header version switcher, e.g.
#   sphinx-build -b html docs docs/_build/0.1.0
#   FLAQSHIP_VERSION=dev sphinx-build -b html docs docs/_build/dev
# Everything lands under docs/_build. The switcher uses page-relative links, so
# it works at any hosting sub-path or local server root with no extra config.
flaqship_versioned = True
