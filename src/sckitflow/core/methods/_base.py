import abc
from typing import Any, Protocol, runtime_checkable

import torch

from sckitflow.core._data_utils import subscript_step_data
from sckitflow.core._types import MatchFn, PredictionData, SamplerFn, StepData
from sckitflow.core.probability_paths._probability_paths import BaseProbabilityPath, LinearDiracProbabilityPath

__all__ = [
    # The two contracts
    "SupportsTraining",
    "SupportsInference",
    # Code reuse for implementations -- not contracts.
    # Completeness is answered by the two Protocols above, which cover
    # implementations that inherit nothing from us as well.
    "AbstractMethod",
    "AbstractFlowMethod",
    # Matching
    "BaseMatcher",
    "Matcher",
    # Matched training
    "MatchedTrainingMethod",
]


# -------------------- The two contracts --------------------
# These are the only structural types the library dispatches on. Every class
# below is there to share code between implementations, never to be type-tested.
@runtime_checkable
class SupportsTraining(Protocol):
    """A module to train, plus `compute_loss`."""

    @property
    def module(self) -> torch.nn.Module: ...
    def compute_loss(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]: ...


@runtime_checkable
class SupportsInference(Protocol):
    """A module to run, plus `predict`."""

    @property
    def module(self) -> torch.nn.Module: ...
    def predict(self, step_data: StepData) -> PredictionData: ...


# -------------------- Shared implementation --------------------
class AbstractMethod:
    """Holds the neural module a method is built on.

    Purely for code reuse between implementations -- never type-test against
    this, use `SupportsTraining` / `SupportsInference`.

    Construction has no side effects: the module is stored as given, never
    moved or retyped, so handing one module to a training and an inference
    method is safe. Placement is the caller's -- ``module.to(device, dtype)``
    before constructing, and ``module.train(mode)`` to switch modes. Inside
    `compute_loss` / `predict` the batch is the reference for device and dtype.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        """Keeps `module` as given.

        :param module: An initialized `torch.nn.Module` the method builds upon,
            already on the device and dtype you want to run in.
        """
        self._module = module

    @property
    def module(self) -> torch.nn.Module:
        return self._module


class AbstractFlowMethod(AbstractMethod):
    """Adds the flow configuration that flow trainers and flow predictors share."""

    def __init__(
        self,
        module: torch.nn.Module,
        probability_path: BaseProbabilityPath | None = None,
        time_sampler: SamplerFn | None = None,
        noise_sampler: SamplerFn | None = None,
        generate_from_noise: bool = False,
    ) -> None:
        """Initializes the module storage plus the flow configuration.

        :param module: An initialized neural module the method builds upon.
        :param probability_path: Optional `BaseProbabilityPath`. Defaults to a
            `LinearDiracProbabilityPath`.
        :param time_sampler: Optional callable sampling times in [0, 1].
            Defaults to `torch.rand`.
        :param noise_sampler: Optional callable sampling source noise.
            Defaults to `torch.randn`.
        :param generate_from_noise: When `True`, interpolation starts from the
            noise distribution even if source states are present (source
            information is passed as extra conditioning instead).
        """
        super().__init__(module)

        if generate_from_noise and noise_sampler is None:
            raise TypeError("When generating from noise, you need to provide a noise_sampler.")

        self._probability_path = LinearDiracProbabilityPath() if probability_path is None else probability_path
        self._noise_sampler = torch.randn if noise_sampler is None else noise_sampler
        self._time_sampler = torch.rand if time_sampler is None else time_sampler
        self._generate_from_noise = generate_from_noise

    @property
    def probability_path(self) -> BaseProbabilityPath:
        return self._probability_path

    @property
    def noise_sampler(self) -> SamplerFn | None:
        return self._noise_sampler

    @property
    def time_sampler(self) -> SamplerFn:
        return self._time_sampler

    @property
    def generate_from_noise(self) -> bool:
        return self._generate_from_noise


# -------------------- Matching --------------------
class BaseMatcher(abc.ABC):
    """Base class for matching methods.

    Stores the `match_fn` callable used to match source and target populations.
    """

    def __init__(self, match_fn: MatchFn) -> None:
        """Initializes the matching method.

        :param match_fn: A callable satisfying `MatchFn`, used to match source
            and target populations from a batch of data.
        """
        self._match_fn = match_fn

    @abc.abstractmethod
    def match(self, step_data: StepData) -> StepData: ...

    @property
    def match_fn(self) -> MatchFn:
        return self._match_fn


class Matcher(BaseMatcher):
    """Public matching method.

    Returns ``step_data`` unchanged when no source coupling data is present
    or when ``match_fn`` yields no indices; otherwise returns a subscripted
    copy aligned on the matched indices.
    """

    def match(self, step_data: StepData) -> StepData:
        """Matches the input state data using the underlying `match_fn`.

        Returns `step_data` unchanged when neither `source_coupling_lin` nor
        `source_coupling_quad` are present, or when `match_fn` returns either
        `src_idxs` or `tgt_idxs` as `None`.
        """
        source_lin = step_data["source_coupling_lin"]
        source_quad = step_data["source_coupling_quad"]
        target_lin = step_data["target_coupling_lin"]
        target_quad = step_data["target_coupling_quad"]

        if source_lin is None and source_quad is None:
            return step_data

        src_idxs, tgt_idxs = self.match_fn(
            source_lin=source_lin,
            target_lin=target_lin,
            source_quad=source_quad,
            target_quad=target_quad,
        )

        if src_idxs is None or tgt_idxs is None:
            return step_data

        return subscript_step_data(step_data, src_idxs=src_idxs, tgt_idxs=tgt_idxs)


# -------------------- Matched training --------------------
class MatchedTrainingMethod:
    """Runs a matcher over the batch, then delegates to the wrapped training method.

    Satisfies `SupportsTraining` structurally, so it is usable anywhere a plain
    training method is -- including wrapped again.
    """

    def __init__(self, method: SupportsTraining, matcher: BaseMatcher) -> None:
        """Initializes the matched training method.

        :param method: The training method to wrap around.
        :param matcher: The matcher pairing source and target, e.g.
            ``Matcher(match_fn)``. Taken rather than built, so a `BaseMatcher`
            subclass can be used in its place.
        """
        self._method = method
        self._matcher = matcher

    def compute_loss(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._method.compute_loss(self._matcher.match(step_data))

    @property
    def method(self) -> SupportsTraining:
        """The wrapped training method."""
        return self._method

    @property
    def matcher(self) -> BaseMatcher:
        """The matcher used to pair the data."""
        return self._matcher

    @property
    def module(self) -> torch.nn.Module:
        """The wrapped method's module; `SupportsTraining` requires it."""
        return self._method.module
