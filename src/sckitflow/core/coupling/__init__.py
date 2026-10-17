from sckitflow.core.coupling._config import (
    CouplingConfig,
    ExactCouplingConfig,
    FusedGromovWassersteinCouplingConfig,
    GromovWassersteinCouplingConfig,
    IndependentCouplingConfig,
    PartialCouplingConfig,
    SinkhornCouplingConfig,
    UnbalancedCouplingConfig,
)
from sckitflow.core.coupling._coupling import independent_coupling, ot_linear_coupling, ot_quadratic_coupling

__all__ = [
    "CouplingConfig",
    "IndependentCouplingConfig",
    "ExactCouplingConfig",
    "SinkhornCouplingConfig",
    "PartialCouplingConfig",
    "UnbalancedCouplingConfig",
    "GromovWassersteinCouplingConfig",
    "FusedGromovWassersteinCouplingConfig",
    "independent_coupling",
    "ot_linear_coupling",
    "ot_quadratic_coupling",
]
