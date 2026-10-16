from dataclasses import dataclass

import numpy as np

from sckitflow.data.containers._base import BaseData
from sckitflow.data.containers._state import StateData

__all__ = ["CouplingData"]


@dataclass(frozen=True)
class CouplingData(BaseData):
    """Container class for coupling data.

    At least one of the two terms must be set. The three valid modes are:

    * **Linear only** (``state_lin`` set, ``state_quad=None``) — standard OT/Wasserstein
      coupling in a shared space.
    * **Fused** (both set) — fused Gromov-Wasserstein: shared axes plus per-side geometry.
    * **Quadratic only** (``state_lin=None``, ``state_quad`` set) — pure
      Gromov-Wasserstein: source and target are matched on intra-domain geometry
      alone, with no shared axes.

    :param state_lin: Container for the linear term of samples. May be `None` for
        pure Gromov-Wasserstein. When `None`, `state_quad` must be set.
    :type state_lin: class: `StateData | None`

    :param state_quad: Container for the quadratic term of samples.
        Defaults to `None`.
    :type state_quad: class: `StateData | None`
    """

    state_lin: StateData | None = None
    state_quad: StateData | None = None

    def __post_init__(self):
        if self.state_lin is None and self.state_quad is None:
            raise ValueError("CouplingData requires at least one of `state_lin` / `state_quad`.")
        if self.state_lin is not None and self.state_quad is not None:
            self.state_lin.assert_same_len(self.state_quad)

    def __repr__(self) -> str:
        parts = [f"n_obs={len(self)}"]

        if self.state_lin is not None:
            lin_shape = self.state_lin.X.shape
            lin_dims = lin_shape[1:] if len(lin_shape) > 1 else ()
            parts.append(f"linear(spatial_dims={lin_dims})")
        else:
            parts.append("linear=None")

        if self.state_quad is not None:
            quad_shape = self.state_quad.X.shape
            quad_dims = quad_shape[1:] if len(quad_shape) > 1 else ()
            parts.append(f"quadratic(spatial_dims={quad_dims})")
        else:
            parts.append("quadratic=None")

        return f"{self.__class__.__name__}({', '.join(parts)})"

    def __len__(self) -> int:
        # Length is defined by whichever term is present. When both are set,
        # `__post_init__` already checked they agree, so picking either is fine.
        return len(self.state_lin if self.state_lin is not None else self.state_quad)

    def __getitem__(self, idxs: np.ndarray | slice) -> "CouplingData":
        state_lin = None if self.state_lin is None else self.state_lin[idxs]
        state_quad = None if self.state_quad is None else self.state_quad[idxs]
        return self.__class__(
            state_lin,
            state_quad=state_quad,
        )

    def assert_same_spatial_dims(
        self,
        other: "CouplingData",
    ) -> None:
        """Checks that the current coupling data shares the same spatial dimensions with another.

        The check is done over the linear term only, when present: it is the factor in
        the product space shared by source and target distributions. With no linear term
        on both sides (pure Gromov-Wasserstein), there are no shared axes to compare and
        the quadratic widths are free to differ -- matching geometry, not coordinates.
        """
        self_has_lin = self.state_lin is not None
        other_has_lin = other.state_lin is not None
        if self_has_lin != other_has_lin:
            raise ValueError(
                "Coupling data disagree on whether a linear term is present: "
                f"found {'linear' if self_has_lin else 'linear-free'} and "
                f"{'linear' if other_has_lin else 'linear-free'}."
            )
        if not self_has_lin:
            return

        n_dims_self = self.state_lin.X.shape[1]
        n_dims_other = other.state_lin.X.shape[1]
        if n_dims_self != n_dims_other:
            msg = (
                "Coupling data should share the same number of spatial "
                f"dimensions for the linear term, found {n_dims_self} and {n_dims_other}."
            )
            raise ValueError(msg)
