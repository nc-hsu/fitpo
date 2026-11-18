# Contributing to fitpo

Bug reports, ideas and pull requests are welcome. Please open an
[issue](https://github.com/nc-hsu/fitpo/issues) first for larger changes, so
we can agree on the approach.

## Setting up

You need Python 3.12 or later. Clone the repository and install fitpo in
editable mode with the development tools:

```bash
git clone https://github.com/nc-hsu/fitpo.git
cd fitpo
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

## Checks

Run these before opening a pull request; CI runs the same commands.

```bash
python -m pytest              # tests and the docstring examples (doctests)
ruff check .                  # lint
ruff format --check .         # formatting (run `ruff format .` to fix)
mypy                          # strict type checks on src/ and tests/
```

## Documentation

The documentation is built with Sphinx from the `docs/` folder:

```bash
python -m sphinx -W -n -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to view it.

## Conventions

- Prefer functions over classes; results are `NamedTuple`s.
- Docstrings in numpy style, with an `Examples` section that runs as a
  doctest for every public function.
- Every saved matplotlib figure uses `fig.savefig(..., facecolor="w")`.
- Add an entry under `Unreleased` in
  [CHANGELOG.md](https://github.com/nc-hsu/fitpo/blob/main/CHANGELOG.md).
