"""Sphinx configuration for Myriad documentation."""

import os
import sys

# Add project root to path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "Myriad"
copyright = "2025, Robin Henry"
author = "Robin Henry"

# Version info
try:
    from myriad import __version__

    release = __version__
    version = ".".join(release.split(".")[:2])
except ImportError:
    release = "0.1.0"
    version = "0.1"

# -- General configuration ---------------------------------------------------

# Extensions
extensions = [
    "sphinx.ext.autodoc",  # Auto-generate docs from docstrings
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "myst_nb",  # Notebook support for tutorials
    "sphinx_copybutton",  # Add copy button to code blocks
]

# MyST Parser configuration
myst_enable_extensions = [
    "colon_fence",  # ::: can be used instead of ``` for fenced code blocks
    "deflist",  # Definition lists
    "dollarmath",  # Dollar math syntax for LaTeX
    "fieldlist",  # Field lists
    "html_admonition",  # HTML-based admonitions
    "html_image",  # HTML image support
    "linkify",  # Auto-convert URLs to links
    "replacements",  # Text replacements
    "smartquotes",  # Smart quotes
    "substitution",  # Substitutions
    "tasklist",  # Task lists
]

myst_heading_anchors = 3  # Auto-generate anchors for headings up to level 3

# -- Notebook execution (myst-nb) -------------------------------------------
nb_execution_mode = "force"  # Always re-execute so docs reflect current code
nb_execution_timeout = 120  # Per-cell timeout in seconds
nb_execution_raise_on_error = True  # Fail the build on notebook errors

# Templates path
templates_path = ["_templates"]

# Patterns to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Master document
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

# Theme
html_theme = "furo"

# Theme options
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/robinhenry/myriad-jax",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/robinhenry/myriad-jax",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                    <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 """
            """7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94"""
            """-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52"""
            """.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08"""
            """-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 """
            """2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25"""
            """.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0"""
            """-4.42-3.58-8-8-8z"></path>
                </svg>
            """,
            "class": "",
        },
    ],
}

# Static files
html_static_path = ["stylesheets"]

# Custom CSS
html_css_files = [
    "extra.css",
]

# Title
html_title = "Myriad Documentation"

# Short title for navigation bar
html_short_title = "Myriad"

# Favicon
# html_favicon = "assets/favicon.png"

# Show "Edit on GitHub" links
html_context = {
    "display_github": True,
    "github_user": "robinhenry",
    "github_repo": "myriad-jax",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- Options for autodoc -----------------------------------------------------

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Type hints configuration
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_type_aliases = {
    "PRNGKey": "jax.Array",
    "Array": "jax.Array",
    "ArrayTree": "chex.ArrayTree",
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = autodoc_type_aliases
napoleon_attr_annotations = True

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    "chex": ("https://chex.readthedocs.io/en/latest/", None),
}
