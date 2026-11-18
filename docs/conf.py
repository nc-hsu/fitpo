"""Sphinx configuration for the fitpo documentation.

The pages are Markdown (MyST); the example pages are MyST Markdown notebooks
that are executed during the build, so every figure and number they show
comes from the installed fitpo. The API reference is generated from the
numpy-style docstrings of the public ``fitpo`` namespace.
"""

from importlib.metadata import version as _version

# -- Project -----------------------------------------------------------------

project = "fitpo"
author = "Nicholas Clemett"
copyright = "Nicholas Clemett"
# Read from the installed package, so the version is written only in
# pyproject.toml.
release = _version("fitpo")
version = release

# -- Extensions ---------------------------------------------------------------

extensions = [
    "myst_nb",  # Markdown pages and executable Markdown notebooks
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # numpy-style docstrings
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

# MyST: $...$ and $$...$$ maths, and LaTeX environments such as align.
myst_enable_extensions = ["dollarmath", "amsmath"]

# Notebooks run during the build; results are cached between builds, and any
# error in a cell fails the build rather than appearing on the page.
nb_execution_mode = "cache"
nb_execution_raise_on_error = True
nb_execution_timeout = 300

# -- API reference ------------------------------------------------------------

autosummary_generate = True
# docs/_templates/autosummary/class.rst documents a class without the methods
# it inherits (FitResult would otherwise list tuple.count and tuple.index).
templates_path = ["_templates"]
# Types go next to each parameter in the description, keeping signatures short.
autodoc_typehints = "description"
# Show type names without their module path (e.g. "OriginMode", not
# "fitpo._common.OriginMode"), since the private modules are not public.
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
# Attributes sections become attribute entries, so the FitResult fields can be
# linked to.
napoleon_use_ivar = False

# The literal mode types are private type aliases; the values they allow are
# listed in each function's parameter description, so they are not linked.
nitpick_ignore = [
    ("py:class", f"fitpo.{mod}.{alias}")
    for mod, alias in [
        ("_common", "OriginMode"),
        ("_deluca", "DelucaMode"),
        ("_bilinear", "BilinearMode"),
        ("_envelope", "EnvelopeMode"),
    ]
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}


def _numpy_typing_public(app, env, node, contnode):  # type: ignore[no-untyped-def]
    """Link numpy types in annotations to where numpy documents them.

    Annotations such as ``npt.ArrayLike`` resolve to the private module in
    which numpy defines them (``numpy._typing``), while numpy documents them
    under ``numpy.typing`` as data; and numpy documents ``numpy.float64`` as
    an attribute, not a class. Sphinx looks both up as classes, so the
    reference is retargeted and resolved again through intersphinx.
    """
    from sphinx.ext.intersphinx import missing_reference

    target = node.get("reftarget", "")
    if target.startswith("numpy._typing."):
        node["reftarget"] = "numpy.typing." + target.rsplit(".", 1)[-1]
        node["reftype"] = "data"
    elif target == "numpy.float64":
        node["reftype"] = "attr"
    else:
        return None
    return missing_reference(app, env, node, contnode)


def setup(app):  # type: ignore[no-untyped-def]
    """Register the numpy typing redirect with Sphinx."""
    app.connect("missing-reference", _numpy_typing_public)


# -- Output -------------------------------------------------------------------

html_theme = "furo"
html_title = f"fitpo {release}"
exclude_patterns = ["_build", "jupyter_execute", ".jupyter_cache", "Thumbs.db", ".DS_Store"]
