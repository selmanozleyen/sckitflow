from __future__ import annotations

from typing import Any

import torch
from pydantic import Field, PositiveInt, model_validator
from scfit.registry import component

from sckitflow.core._data_utils import (
    expand_conditioning,
    get_tensor_dict_from_data,
    prepare_latent_inference,
)
from sckitflow.core._types import PredictionData, StepData
from sckitflow.core.methods._base import AbstractFlowMethod, AbstractFlowMethodConfig, InferenceMethodConfig
from sckitflow.core.methods.inference._utils import aggregate_predictions
from sckitflow.core.solvers import ODESolver
from sckitflow.data._dims import DataDimensions

__all__ = ["ODEInference", "ODEInferenceConfig"]


class ODEInference(AbstractFlowMethod):
    """ODE inference from an underlying velocity-field module.

    The module is expected to implement ``.get_vf_fn`` to compile a
    ``vf_fn(t, xt)`` function compatible with ``torchdiffeq``.

    Constructed from the module, the flow configuration, and the data shape spec:

    .. code-block:: python

        inference = ODEInference(
            module,
            ODEInferenceConfig(probability_path=..., n_steps=50),
            dims=dm.get_data_dimensionalities(adata),
        )
    """

    def __init__(
        self,
        module: torch.nn.Module,
        config: ODEInferenceConfig | None = None,
        *,
        dims: DataDimensions,
        latent: torch.Tensor | None = None,
    ) -> None:
        """Initializes the ODE inference method.

        :param module: An initialized neural module the method builds upon.
        :param config: The flow and solver settings; ``None`` takes every default.
        :param dims: The shape spec of the data the module was built for. Its
            ``state_dim`` sizes the latent when inference is unpaired.
        :param latent: Optional initial latent state; when provided, sampling from
            the noise distribution is skipped. It is moved to the batch's device
            and dtype. Live, so not part of the config.
        """
        self.config = ODEInferenceConfig() if config is None else config
        super().__init__(module, self.config, dims=dims)
        self._latent = latent

    def predict(self, step_data: StepData, *, generator: torch.Generator) -> PredictionData:
        """Integrates the ODE and returns the aggregated prediction.

        The latent is drawn at ``(B, dims.state_dim)`` (or ``(N, B, dims.state_dim)`` with
        ``n_samples``), with ``B`` and the reference device/dtype coming from
        ``source_state`` when present, else the conditioning tensors. A caller-supplied
        ``latent`` overrides both.
        """
        source_state = step_data["source_state"]

        # ----- 1. Build conditioning dict -----
        condition_dict = {
            **get_tensor_dict_from_data(step_data["target_condition_data"]),
            **get_tensor_dict_from_data(step_data["target_group_data"]),
        }

        # ----- 2. Prepare latent (noise) -----
        if self.latent is None:
            latent = prepare_latent_inference(
                source_state,
                condition_dict,
                self.noise_sampler,
                self.dims,
                n_samples=self.config.n_samples,
                generate_from_noise=self.generate_from_noise,
                generator=generator,
            )
        else:
            # `self.latent` is the only tensor not built by the loader, so it is the
            # only one that has to be placed. Reference order matches
            # `prepare_latent_inference`: source first, else first conditioning tensor.
            reference = source_state if source_state is not None else next(iter(condition_dict.values()))
            latent = self.latent.to(device=reference.device, dtype=reference.dtype)

        # ----- 3. Expand conditioning to match latent dimensions -----
        condition_dict, source_expanded = expand_conditioning(
            latent,
            condition_dict,
            source_state,
        )

        # ----- 4. Configure ODE solver -----
        solver_kwargs = dict(self.config.solver_kwargs)
        solver_kwargs.setdefault("method", "euler")
        method = solver_kwargs.pop("method")

        time_grid = torch.linspace(0.0, 1.0, steps=self.config.n_steps, device=latent.device, dtype=latent.dtype)
        solver = ODESolver(
            self.module,
            method=method,
            vf_kwargs={"condition_dict": condition_dict, "source": source_expanded},
            device_id=str(latent.device),
        )

        # ----- 5. Integrate -----
        predictions = solver.solve(
            latent,
            time_grid,
            solver_kwargs=solver_kwargs,
            return_trajectory=self.config.return_trajectory,
        )

        # ----- 6. Aggregate predictions (averaging, reshaping) -----
        X, traj, raw_samples = aggregate_predictions(
            predictions=predictions,
            latent_shape=latent.shape,
            return_trajectory=self.config.return_trajectory,
        )
        return PredictionData(X=X, traj=traj, raw_samples=raw_samples)

    @property
    def latent(self) -> torch.Tensor | None:
        return self._latent


@component("inference_method.ode", builds=ODEInference)
class ODEInferenceConfig(AbstractFlowMethodConfig, InferenceMethodConfig):
    """ODE inference over a trained velocity field: the flow settings plus the solver's."""

    solver_kwargs: dict[str, Any] = {}
    """Forwarded to the ODE solver. ``method`` defaults to ``"euler"``."""
    return_trajectory: bool = False
    """Return the whole trajectory instead of only the endpoint."""
    n_steps: int = Field(default=100, ge=2)
    """Points on the solver's time grid, both ends included."""
    n_samples: PositiveInt | None = None
    """Samples per batch element. Required when ``generate_from_noise`` is ``True``."""

    @model_validator(mode="after")
    def _noise_needs_samples(self) -> ODEInferenceConfig:
        if self.generate_from_noise and self.n_samples is None:
            raise ValueError("generating from noise needs `n_samples`.")
        return self

    def build(self, module: torch.nn.Module, *, dims: DataDimensions) -> ODEInference:
        return ODEInference(module, self, dims=dims)
