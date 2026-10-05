from sckitflow.core.coupling._config import (
    CouplingConfig,
    IndependentCouplingConfig,
    OTLinearCouplingConfig,
    OTQuadraticCouplingConfig,
)
from sckitflow.core.coupling._coupling import independent_coupling, ot_linear_coupling, ot_quadratic_coupling

__all__ = [
    "CouplingConfig",
    "IndependentCouplingConfig",
    "OTLinearCouplingConfig",
    "OTQuadraticCouplingConfig",
    "independent_coupling",
    "ot_linear_coupling",
    "ot_quadratic_coupling",
]
