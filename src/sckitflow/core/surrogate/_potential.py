import abc
from typing import Any

import torch

from sckitflow._random import generators
from sckitflow.core._data_utils import get_tensor_dict_from_data
from sckitflow.core._types import PredictionData, StepData
from sckitflow.core.methods._base import SupportsInference, _placement_of
from sckitflow.data._datamodule import FlowDataModule
from sckitflow.data._dims import DataDimensions
from sckitflow.data._loader import EvalLoader

__all__ = ["SurrogatePotential"]


def _rows(container: Any) -> int | None:
    """Leading dimension of a tensor or of the first tensor in a mapping, `None` when empty."""
    if container is None:
        return None
    if isinstance(container, torch.Tensor):
        return container.shape[0]
    tensors = [v for v in get_tensor_dict_from_data(container).values() if v is not None]
    return tensors[0].shape[0] if tensors else None


def _pair_leaf(
    step_data: StepData,
    cond_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[StepData, dict[str, torch.Tensor]]:
    r"""Lays a leaf out against the condition dictionary for one prediction.

    A leaf has ``S`` source rows (when a source state is present) and ``T`` target rows. The target rows
    only vary when the leaf carries condition covariates that are not optimized; otherwise they hold the
    leaf's group data, which is constant within a leaf, so a single row stands for all of them. The
    ``P`` pairs are the Cartesian product of the varying source and target rows, and every field is laid
    out as $(P, N, \dots)$ against the $N$ condition samples.

    Only the fields prediction reads are kept: the source side, and the target conditions and groups.
    Other target fields (states, couplings, responses) are dropped, since the conditions replace the
    target side. Everything is moved to ``device``.
    """
    optimized = set(cond_dict)
    extra = {
        k: v for k, v in get_tensor_dict_from_data(step_data.get("target_condition_data")).items() if k not in optimized
    }
    groups = step_data.get("target_group_data")
    S = _rows(step_data.get("source_state"))
    T = _rows(extra) if extra else None
    if S is None and T is None and _rows(groups) is None:
        # nothing to pair: the conditions alone are the batch
        conds = {k: v.to(device) for k, v in cond_dict.items()}
        return {**step_data, "target_condition_data": conds}, conds

    s, t = S or 1, T or 1
    src_idx = torch.arange(s, device=device).repeat_interleave(t)
    tgt_idx = torch.arange(t, device=device).repeat(s)
    n_pairs, n = s * t, next(iter(cond_dict.values())).shape[0]

    def lay_out(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        x = x.to(device)[idx]
        return x.unsqueeze(1).expand(n_pairs, n, *x.shape[1:])

    def lay_out_container(container: Any, idx: torch.Tensor) -> Any:
        if container is None:
            return None
        if isinstance(container, torch.Tensor):
            return lay_out(container, idx)
        return {k: lay_out(v, idx) for k, v in get_tensor_dict_from_data(container).items()}

    if groups is not None and T is None:
        # groups are constant within a leaf; check it, since one row stands for all of them
        for key, value in get_tensor_dict_from_data(groups).items():
            if not torch.equal(value, value[:1].expand_as(value)):
                raise ValueError(f"Target group data {key!r} varies within a leaf; it cannot stand for one row.")
        groups = {k: v[:1] for k, v in get_tensor_dict_from_data(groups).items()}

    conds = {k: v.to(device).unsqueeze(0).expand(n_pairs, *v.shape) for k, v in cond_dict.items()}
    paired: StepData = {
        key: (lay_out_container(value, src_idx) if key.startswith("source_") else None)
        for key, value in step_data.items()
    }
    paired["target_group_data"] = lay_out_container(groups, tgt_idx)
    paired["target_condition_data"] = {**{k: lay_out(v, tgt_idx) for k, v in extra.items()}, **conds}
    return paired, conds


def _verify_continuous_conditions_dims(
    dims: DataDimensions,
    cond_dict: dict[str, torch.Tensor],
) -> None:
    """Verifies that the continuous condition dictionary respects the dimensionality contract.

    The condition tensor will be checked against the dimensionality contract over the
    trailing dimensions -- that needs to correspond to the entry with the same key in
    `dims`.

    :param dims: `DataDimensions` containing the contract for the data dimensionalities.
    :param cond_dict: Dictionary of `torch.Tensors` whose dimension to verify.
    """
    # Parse the data dimensionality to get the registry for continuous condition
    # covariates. This can be `None`, so we raise an error immediately when this is the
    # case, as the check is to be done over the continuous covariates.
    ref_continuous_dims: dict[str, int] | None = dims.condition_continuous_dims
    # We immediately raise ValueError when the reference continuous dimensions are None.
    if ref_continuous_dims is None:
        raise ValueError("The reference dimensions do not include continuous conditions covariate.")

    # Iterate over the condition dictionary items and verify the trailing dimensions.
    for cond_key, cond_data in cond_dict.items():
        # Raise error here, because at the moment we only support two-dimensional
        # condition data for optimization.
        if cond_data.ndim != 2:
            raise ValueError(f"Only two-dimensional inputs supported for the moment, but {cond_data.ndim} were found.")

        # Raise error when the condition key does not appear in the dimensionality.
        if cond_key not in ref_continuous_dims:
            raise ValueError(
                f"Condition key {cond_key} does not appear in the reference dimensions {sorted(ref_continuous_dims)}."
            )
        # Raise error when the dimensionalities mismatch
        ref_dim = ref_continuous_dims[cond_key]
        query_dim = cond_data.shape[-1]
        if query_dim != ref_dim:
            raise ValueError(f"Shape mismatch for condition key {cond_key} -- got {query_dim} but expected {ref_dim}.")


class SurrogatePotential(abc.ABC, torch.nn.Module):
    """Abstract class for surrogate potentials.

    Instances must implement the `get_response` and `compute_raw_potential` methods, satisfying
    the contract specified by the method signature. A surrogate potential is defined
    in terms of an inference module, a target response and a context data module.
    As input for the `forward` method, it will take a dictionary of continuous condition covariates.

    Surrogate potentials will be bound to the datamodule that was used to initialize them; they will always
    use prediction data loaders retrieved from the context data module.

    It follows the sequence of three steps:
    1. Query the inference module with the input condition dictionary (`inferer.predict`).
    2. Extract the response tensor from the prediction (`get_response`).
    3. Evaluate the raw potential on the masked response (`compute_raw_potential`).

    In addition, it also allows to mask custom dimensions out from the evaluation of the potential.
    """

    def __init__(
        self,
        inferer: SupportsInference,
        ystar: torch.Tensor,
        datamodule: FlowDataModule,
        mask: torch.Tensor | None = None,
        seed: int = 0,
    ) -> None:
        """Initializes the surrogate potential.

        :param inferer: An instantiated object satisfying the `SupportsInference` protocol.
        :param ystar: A `torch.Tensor` holding the target response for the potential evaluation.
        :param datamodule: The context `FlowDataModule` to run optimization over.
        :param mask: A boolean mask of shape (D,) selecting the response dimensions to evaluate the potential on.
            Coerced to bool; index lists are not supported. Defaults to None, in which case all dimensions are selected.
        :param seed: Integer to seed random number generation. Defaults to `0`.
        """
        # We squeeze the target tensor and expect it to be a flat tensor.
        ystar = torch.atleast_1d(torch.squeeze(ystar))
        if ystar.ndim != 1:
            raise ValueError(f"Invalid shape for target response, found {ystar.shape}")
        # For the mask, we do the same. Squeeze when available, otherwise
        # initialize it as ones from the target. Then check that the shape matches the
        # one of the target response. Its shape needs to match the one of the target.
        if mask is not None and mask.dtype not in (torch.bool, torch.uint8):
            raise ValueError("mask must be a per-dimension boolean mask, not an index list.")
        mask = torch.atleast_1d(torch.squeeze(mask)) if mask is not None else torch.ones_like(ystar)
        mask = mask.bool()
        if mask.shape != ystar.shape:
            raise ValueError(
                "The mask and the target tensor should have the same shape -- "
                f"found mask of shape {mask.shape}, but target has shape {ystar.shape}."
            )

        # Initialize parent class and register tensor buffers.
        super().__init__()
        self.register_buffer("_ystar", ystar)
        self.register_buffer("_mask", mask)

        # Set attributes.
        self._datamodule = datamodule
        self._inferer = inferer
        self._seed = seed

    @abc.abstractmethod
    def get_response(self, pred_data: PredictionData) -> torch.Tensor:
        """Extracts the response tensor from the prediction data.

        :param pred_data: The raw output of ``inferer.predict``. When the leaf has source or target rows,
            its samples carry leading cell axes (the ``P`` leaf pairs, and any samples the inferer draws)
            before the ``N`` condition axis.
        :returns: A tensor of shape ``(N, D)``: the readout averaged over those cell axes, i.e. the
            Monte Carlo mean response of each condition. ``N`` matches the leading dimension of every
            condition tensor in ``cond_dict`` and ``D`` equals ``len(ystar)``. Average the readout before
            the potential, not the potential over cells, since a nonlinear potential does not commute
            with the mean.
            Masking with ``self.mask`` along the last axis yields ``(N, D')``,
            which is what ``compute_raw_potential`` receives.
        """

    @abc.abstractmethod
    def compute_raw_potential(self, y: torch.Tensor, ystar: torch.Tensor) -> torch.Tensor:
        """Evaluates the raw potential.

        :param y: Masked predictions, shape ``(N, D')``, where ``N`` is the
            number of optimization samples in the condition dictionary and
            ``D'`` is the number of unmasked response dimensions.
        :param ystar: Masked target response, shape ``(D',)``. Broadcasts
            against ``y``.
        :returns: Per-sample potential, shape ``(N,)``.
        """

    def forward(self, cond_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """The potential of each condition, averaged over the leaves of the context data module.

        :param cond_dict: A dictionary of input tensors, each of shape ``(N, D)``.
        :returns: Shape ``(N,)``.
        """
        return self.population_potentials(cond_dict, n_populations=1)[0]

    def population_potentials(self, cond_dict: dict[str, torch.Tensor], n_populations: int) -> torch.Tensor:
        """The potential of each condition for ``n_populations`` independently drawn populations.

        Each population draws its own generator per leaf, so the spread across populations is the
        forward model's predictive uncertainty: ``population_potentials(c, n).std(0)``.

        :param cond_dict: A dictionary of input tensors, each of shape ``(N, D)``.
        :returns: Shape ``(n_populations, N)``.
        """
        _verify_continuous_conditions_dims(self._datamodule.data_dims, cond_dict)
        if not len(cond_dict):
            raise ValueError("Cannot compute the potential with an empty condition dictionary.")
        n = next(iter(cond_dict.values())).shape[0]
        for cond_key, cond_data in cond_dict.items():
            if cond_data.shape[0] != n:
                raise ValueError(
                    f"Condition data at key {cond_key} has the wrong number of optimization samples -- "
                    f"found {cond_data.shape[0]} but expected {n}."
                )
        device, _ = _placement_of(self.inferer.module)
        leaves = list(self.predict_dl)
        if not leaves:
            raise ValueError("The context data module has no prediction leaves.")
        populations = []
        for population in range(n_populations):
            potentials = []
            for i, (step_data, leaf) in enumerate(leaves):
                generator, _ = generators(self._seed, i, population, device=device)
                paired_step_data, _ = _pair_leaf(step_data, cond_dict, device)
                y = self.get_response(self.inferer.predict(paired_step_data, generator=generator))
                if y.shape[-1] != self.ystar.shape[-1]:
                    raise ValueError(
                        f"get_response returned a tensor whose last dim is {y.shape[-1]}, "
                        f"but the target response has dim {self.ystar.shape[-1]}."
                    )
                pot = self.compute_raw_potential(y[..., self._mask], self.ystar[self._mask])
                if pot.shape != (n,):
                    raise ValueError(
                        f"Potential value for leaf {leaf} has the wrong shape -- got {pot.shape}, but expected {(n,)}."
                    )
                potentials.append(pot)
            populations.append(torch.stack(potentials).mean(0))
        return torch.stack(populations)

    @property
    def predict_dl(self) -> EvalLoader:
        """The context datamodule's prediction loader, rebuilt per call so `set_predict_data` is honored."""
        return self._datamodule.predict_dataloader()

    @property
    def datamodule(self) -> FlowDataModule:
        """Returns the underlying data module for the context."""
        return self._datamodule

    @property
    def inferer(self) -> SupportsInference:
        """Returns the underlying forward model used to compute the potential."""
        return self._inferer

    @property
    def mask(self) -> torch.Tensor:
        """Returns the mask for the considered response axes."""
        return self._mask

    @property
    def ystar(self) -> torch.Tensor:
        """Returns the target response."""
        return self._ystar

    @property
    def seed(self) -> int:
        """Returns the seed to control random number generation."""
        return self._seed
