from typing import Any

import numpy as np
import torch
from scfit.params import ParamsComponent
from scfit.registry import component

from sckitflow.core._data_utils import (
    get_tensor_dict_from_data,
    prepare_latent_train,
)
from sckitflow.core._types import StepData
from sckitflow.core.methods._base import AbstractFlowMethod, AbstractFlowMethodParams, TrainingMethodConfig

__all__ = ["CFMTraining", "CFMTrainingConfig"]


class CFMTraining(AbstractFlowMethod):
    """Conditional Flow Matching training method.

    Constructed from the module and the flow configuration:

    .. code-block:: python

        method = CFMTraining(module, probability_path=..., time_sampler=...)

    Passing the same module and configuration to a flow inference method
    (e.g. :class:`~sckitflow.core.methods.inference.ODEInference`) gives both
    the same probability path, time sampler, noise sampler and module.
    """

    def compute_loss(
        self, step_data: StepData, *, generator: torch.Generator, rng: np.random.Generator
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """The flow-matching loss. Noise, time and path samples are drawn from ``generator``; ``rng`` is unused."""
        # ---- Get source and target states from step data ----
        # The batch is the reference: the loader built it in the module's dtype
        # and Lightning placed it, so nothing here is coerced.
        target = step_data["target_state"]
        source = step_data["source_state"]

        # ---- Get conditioning data from step data ----
        cond = {
            **get_tensor_dict_from_data(step_data["target_condition_data"]),
            **get_tensor_dict_from_data(step_data["target_group_data"]),
        }

        # ---- Sample latent (noise) – shape (batch_size, dim) -----
        latent = prepare_latent_train(
            source,
            target,
            self.noise_sampler,
            generate_from_noise=self.generate_from_noise,
            generator=generator,
        )
        batch_size = latent.shape[0]

        # ---- Sample time ----
        t = self.time_sampler((batch_size,), generator=generator, device=latent.device, dtype=latent.dtype)

        # ---- Sample from probability path: interpolant and velocity ----
        xt = self.probability_path.compute_xt(t, latent, target, generator=generator)
        ut = self.probability_path.compute_ut(t, xt, latent, target)

        # ---- Predict velocity field and compute loss ----
        vt = self.module(t, xt, condition_dict=cond, source=source)
        loss = torch.nn.functional.mse_loss(vt, ut)
        return loss, {"loss": loss.item()}


@component("training_method.cfm")
class CFMTrainingConfig(ParamsComponent[AbstractFlowMethodParams], TrainingMethodConfig[CFMTraining]):
    """Conditional Flow Matching training."""

    def build(self, module: torch.nn.Module) -> CFMTraining:
        return CFMTraining(module, **self.params)
