# Installation

fitpo needs Python 3.12 or later. Install it from PyPI:

```bash
pip install fitpo
```

Its only dependencies are numpy and scipy. To plot results as in the examples,
also install matplotlib:

```bash
pip install "fitpo[plot]"
```

fitpo is alpha software, so its results may change between versions. For
work that must be reproducible, pin the version you used, e.g.
`fitpo==0.1.0`, in your requirements.

## Development version

To work on fitpo itself, install it from the source in editable mode with the
development tools:

```bash
git clone https://github.com/nc-hsu/fitpo.git
cd fitpo
pip install -e ".[dev]"
```

The [contributing guide](https://github.com/nc-hsu/fitpo/blob/main/CONTRIBUTING.md)
lists the tests and checks to run.
