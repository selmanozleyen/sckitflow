from dataclasses import field
from typing import Annotated, Any, Unpack

import torch
from scfit.params import Default, ParamsComponent
from scfit.registry import component

from sckitflow.core._data_utils import (
    expand_conditioning,
    get_tensor_dict_from_data,
    prepare_latent_inference,
)
from sckitflow.core._types import PredictionData, StepData
from sckitflow.core.methods._base import AbstractFlowMethod, FlowParams, InferenceMethodConfig
from sckitflow.core.methods.inference._utils import aggregate_predictions
from sckitflow.core.solvers import ODESolver

__all__ = ["ODEInference", "ODEParams", "ODEConfig"]


class ODEParams(FlowParams, total=False):
    """`FlowParams` plus the ODE solver settings."""

    solver_kwargs: Annotated[dict[str, Any], Default({})]
    """Forwarded to the ODE solver. ``method`` defaults to ``"euler"``."""
    return_trajectory: Annotated[bool, Default(False)]
    """Return the whole trajectory instead of only the endpoint."""
    n_steps: Annotated[int, Default(100)]
    """Discretization steps for the solver."""
    n_samples: Annotated[int | None, Default(None)]
    """Samples per batch element. Required when ``generate_from_noise`` is ``True``."""


class ODEInference(AbstractFlowMethod):
    """ODE inference from an underlying velocity-field module.

    The module is expected to implement ``.get_vf_fn`` to compile a
    ``vf_fn(t, xt)`` function compatible with ``torchdiffeq``.

    Constructed from the module and the flow configuration:

    .. code-block:: python

        inference = ODEInference(module, probability_path=..., time_sampler=..., n_steps=50)

    Passing the same module and configuration to a flow training method
    (e.g. :class:`~sckitflow.core.methods.training.CFMTraining`) gives
    both the same probability path, time sampler, noise sampler, module, dtype,
    and device.
    """

    def __init__(
        self, module: torch.nn.Module, *, latent: torch.Tensor | None = None, **params: Unpack[ODEParams]
    ) -> None:
        """Initializes the ODE inference method.

        :param module: An initialized neural module the method builds upon.
        :param latent: Optional initial latent state; when provided, sampling
            from the noise distribution is skipped. Must already be on the
            configured device and dtype. Live, so not one of the params.
        """
        super().__init__(module, **params)
        p = self._params
        self._solver_kwargs = p["solver_kwargs"]
        self._return_trajectory = p["return_trajectory"]
        self._n_steps = p["n_steps"]
        self._latent = latent
        self._n_samples = p["n_samples"]

    def predict(self, step_data: StepData) -> PredictionData:
        """Integrates the ODE and returns the aggregated prediction.

        The dynamics are read through the shared specs: ``self.probability_path``,
        ``self.time_sampler``, ``self.noise_sampler``, ``self.generate_from_noise``,
        ``self.module``. Device and dtype come from the batch itself.
        """
        # ---- 0. Guard, when generating from noise we need n_samples ----
        if self.generate_from_noise and self.n_samples is None:
            raise ValueError("When generating from noise, you need to provide the number of samples with `n_samples`")
        if self.generate_from_noise and self.noise_sampler is None:
            raise TypeError("When generating from noise, you need to provide a noise_sampler.")

        # ----- 1. Prepare latent (noise) -----
        target_state = step_data["target_state"]
        if self.latent is None:
            latent = prepare_latent_inference(
                step_data["source_state"],
                step_data["target_state"],
                self.noise_sampler,
                n_samples=self.n_samples,
                generate_from_noise=self.generate_from_noise,
            )
        else:
            # the only tensor not built by the loader, so the only one to place
            latent = self.latent.to(device=target_state.device, dtype=target_state.dtype)

        # ---- 2. Get optional source ----
        source_state = step_data["source_state"]

        # ----- 3. Build conditioning dict -----
        condition_dict = {
            **get_tensor_dict_from_data(step_data["target_condition_data"]),
            **get_tensor_dict_from_data(step_data["target_group_data"]),
        }

        # ----- 4. Expand conditioning to match latent dimensions -----
        condition_dict, source_expanded = expand_conditioning(
            latent,
            condition_dict,
            source_state,
        )

        # ----- 5. Configure ODE solver -----
        solver_kwargs = dict(self.solver_kwargs or {})
        solver_kwargs.setdefault("method", "euler")
        method = solver_kwargs.pop("method")

        time_grid = torch.linspace(0.0, 1.0, steps=self.n_steps, device=latent.device, dtype=latent.dtype)
        solver = ODESolver(
            self.module,
            method=method,
            vf_kwargs={"condition_dict": condition_dict, "source": source_expanded},
            device_id=str(latent.device),
        )

        # ----- 6. Integrate -----
        predictions = solver.solve(
            latent,
            time_grid,
            solver_kwargs=solver_kwargs,
            return_trajectory=self.return_trajectory,
        )

        # ----- 7. Aggregate predictions (averaging, reshaping) -----
        X, traj, raw_samples = aggregate_predictions(
            predictions=predictions,
            latent_shape=latent.shape,
            return_trajectory=self.return_trajectory,
        )
        return PredictionData(X=X, traj=traj, raw_samples=raw_samples)

    @property
    def solver_kwargs(self) -> dict[str, Any] | None:
        return self._solver_kwargs

    @property
    def return_trajectory(self) -> bool:
        return self._return_trajectory

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def latent(self) -> torch.Tensor | None:
        return self._latent

    @property
    def n_samples(self) -> int | None:
        return self._n_samples


@component("inference_method.ode")
class ODEConfig(ParamsComponent, InferenceMethodConfig):
    """ODE inference over a trained velocity field."""

    params: ODEParams = field(default_factory=lambda: ODEParams())

    def build(self, module: torch.nn.Module) -> ODEInference:
        return ODEInference(module, **self.params)
