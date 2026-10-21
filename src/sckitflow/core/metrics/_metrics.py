from typing import Any

import torch
from torchmetrics import Metric

__all__ = [
    "rbf_kernel_torch",
    "EnergyDistance",
    "MaximumMeanDiscrepancy",
]


def rbf_kernel_torch(x1: torch.Tensor, x2: torch.Tensor, gamma: float = 1.0):
    return torch.exp(-gamma * torch.cdist(x1, x2) ** 2)


class EnergyDistance(Metric):
    """Energy distance between predictions and targets, averaged over updates.

    :param max_samples: Score at most this many rows of ``pred`` and of ``target`` per update, drawn
        at random with a fixed seed. The cost is quadratic in the rows, and validation scores every
        sample of every cell. ``None`` scores all rows.
    """

    def __init__(self, dist_sync_on_step=False, max_samples: int | None = None):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.max_samples = max_samples

        self.add_state("energy_distance_raw", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def _subsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_samples is None or len(x) <= self.max_samples:
            return x
        generator = torch.Generator(device=x.device).manual_seed(0)
        return x[torch.randperm(len(x), generator=generator, device=x.device)[: self.max_samples]]

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred, target = self._subsample(pred), self._subsample(target)
        sigma_pred = torch.cdist(pred, pred).mean()
        sigma_target = torch.cdist(target, target).mean()
        delta = torch.cdist(pred, target).mean()
        self.energy_distance_raw += 2.0 * delta - sigma_pred - sigma_target
        self.total += 1

    def compute(self) -> Any:
        return self.energy_distance_raw / self.total


class MaximumMeanDiscrepancy(Metric):
    def __init__(self, gammas: list[float] = None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        if gammas is None:
            self.gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
        else:
            self.gammas = gammas

        self.add_state("mmd_raw", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0, dtype=torch.float32), dist_reduce_fx="sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        # computing mmd
        mmds = []
        for gamma in self.gammas:
            xx = rbf_kernel_torch(pred, pred, gamma=gamma).mean()
            xy = rbf_kernel_torch(pred, target, gamma=gamma).mean()
            yy = rbf_kernel_torch(target, target, gamma=gamma).mean()
            mmds.append(xx + yy - 2 * xy)
        self.mmd_raw += torch.nanmean(torch.tensor(mmds))
        self.total += 1

    def compute(self) -> Any:
        return self.mmd_raw / self.total
