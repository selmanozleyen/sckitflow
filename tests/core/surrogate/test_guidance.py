"""Guided sampling on a toy design prior: eight training designs on the unit circle, target off the circle."""

import math

import pytest
import torch

from sckitflow.core.surrogate import guided_sample

DESIGNS = torch.stack([torch.tensor([math.cos(a), math.sin(a)]) for a in torch.arange(8) * 2 * math.pi / 8])
TARGET = torch.tensor([3.0, 0.0])


def velocity(t, x):
    """Exact velocity of the linear path from N(0, I) to the empirical distribution of DESIGNS."""
    t = float(t)
    log_w = -((x[:, None] - t * DESIGNS[None]) ** 2).sum(-1) / (2 * (1 - t) ** 2)
    return (torch.softmax(log_w, -1) @ DESIGNS - x) / (1 - t)


def potential(x):
    return ((x - TARGET) ** 2).sum(-1)


def nearest_design(x):
    distance, index = torch.cdist(x, DESIGNS).min(1)
    return distance, index


@pytest.fixture
def x0():
    return torch.randn(256, 2, generator=torch.Generator().manual_seed(0))


def test_without_guidance_samples_the_prior(x0):
    x = guided_sample(velocity, potential, x0, n_steps=200, guidance_scale=0.0)
    distance, index = nearest_design(x)
    assert distance.max() < 1e-3
    assert torch.unique(index).numel() == len(DESIGNS)


def test_guidance_tilts_toward_the_target_and_stays_on_the_prior(x0):
    unguided = guided_sample(velocity, potential, x0, n_steps=200, guidance_scale=0.0)
    guided = guided_sample(velocity, potential, x0, n_steps=200, guidance_scale=0.3)
    distance, index = nearest_design(guided)
    assert distance.max() < 1e-3  # every sample is a training design
    assert potential(guided).mean() < potential(unguided).mean()
    assert torch.bincount(index, minlength=len(DESIGNS)).argmax() == 0  # the design closest to the target
    assert torch.unique(index).numel() > 1  # still diverse


def test_minimizing_the_potential_directly_collapses_off_the_prior(x0):
    """The baseline guidance replaces: gradient descent on the potential alone."""
    x = x0.clone().requires_grad_(True)
    optimizer = torch.optim.SGD([x], lr=0.1)
    for _ in range(300):
        optimizer.zero_grad()
        potential(x).sum().backward()
        optimizer.step()
    distance, _ = nearest_design(x.detach())
    assert distance.min() > 1.5  # off the support
    assert x.detach().std(0).max() < 1e-4  # every sample at one point


def test_schedule_and_constant_scale_agree(x0):
    constant = guided_sample(velocity, potential, x0, n_steps=50, guidance_scale=0.3)
    schedule = guided_sample(velocity, potential, x0, n_steps=50, guidance_scale=lambda _: 0.3)
    torch.testing.assert_close(constant, schedule)


def test_runs_under_no_grad(x0):
    with torch.no_grad():
        x = guided_sample(velocity, potential, x0, n_steps=20, guidance_scale=0.3)
    assert x.shape == x0.shape and not x.requires_grad
