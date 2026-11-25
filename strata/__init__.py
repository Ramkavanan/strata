"""STRATA benchmark harness."""

__version__ = "0.2.0"


def __getattr__(name):  # lazy, so `import strata` stays light
    if name in ("evaluate",):
        from .api import evaluate

        return evaluate
    if name in ("load_dataset", "Golden", "Dataset"):
        from . import datasets

        return getattr(datasets, name)
    raise AttributeError(name)
