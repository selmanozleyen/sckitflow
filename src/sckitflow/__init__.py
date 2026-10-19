from importlib.metadata import version

from sckitflow import core, data, dataset, trainer
from sckitflow._model import Model, ModelConfig
from sckitflow._predict import predict_adata
from sckitflow.core import methods, probability_paths

__version__ = version("sckitflow")

__all__ = [
    "predict_adata",
    "Model",
    "ModelConfig",
    "__version__",
    "core",
    "data",
    "dataset",
    "trainer",
]
