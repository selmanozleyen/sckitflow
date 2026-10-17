import pytest
import torch
from pydantic import ValidationError

from sckitflow.trainer._optim import OptimizerConfig


def test_optimizer_config_coerces_yaml_strings():
    config = OptimizerConfig(kwargs={"lr": "1e-4", "betas": [0.9, 0.99]})
    assert config.kwargs == {"lr": 1e-4, "betas": (0.9, 0.99)}
    assert isinstance(config.build([torch.nn.Parameter(torch.zeros(1))]), torch.optim.Adam)


@pytest.mark.parametrize(
    ("spec", "match"),
    [({"name": "Optimizer"}, "has no optimizer"), ({"name": "Nope"}, "has no optimizer"), ({"kwargs": {"lr_": 1e-3}}, "Extra inputs are not permitted"), ({"kwargs": {"lr": "fast"}}, "valid number")],
)
def test_optimizer_config_rejects(spec, match):
    with pytest.raises(ValidationError, match=match):
        OptimizerConfig(**spec)
