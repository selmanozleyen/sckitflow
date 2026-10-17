import pytest
from pydantic import ValidationError

from sckitflow.trainer._optim import OptimizerConfig


@pytest.mark.parametrize("name", ["Optimizer", "Nope"])
def test_optimizer_config_rejects_non_optimizers(name):
    with pytest.raises(ValidationError, match="has no optimizer"):
        OptimizerConfig(name=name)
