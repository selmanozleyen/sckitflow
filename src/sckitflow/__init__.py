from importlib.metadata import version

from sckitflow import core, data, dataset, trainer
from sckitflow._predict import predict_adata
from sckitflow._run import RunRngs, RunSpec, load_run, run_rngs, save_run

__version__ = version("sckitflow")

__all__ = [
    "predict_adata",
    "save_run",
    "load_run",
    "run_rngs",
    "RunRngs",
    "RunSpec",
    "__version__",
    "core",
    "data",
    "dataset",
    "trainer",
]
