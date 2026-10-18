from collections.abc import Callable

import torch

from sckitflow.core._types import VfFn

__all__ = ["guided_sample"]


def guided_sample(
    velocity: VfFn,
    potential: Callable[[torch.Tensor], torch.Tensor],
    x0: torch.Tensor,
    *,
    n_steps: int = 100,
    guidance_scale: float | Callable[[float], float] = 1.0,
) -> torch.Tensor:
    r"""Samples designs from a flow prior steered toward low ``potential`` (LabCompass loss guidance).

    Integrates $\mathrm{d}z_t = \big(v_t(z_t) - b_t \nabla_z \Psi(f_t(z_t))\big)\,\mathrm{d}t$ from noise at
    $t = 0$ to designs at $t = 1$ with Euler steps, where $f_t(z) = z + (1 - t)\,v_t(z)$ is the one-step
    estimate of the clean design. Autograd differentiates through $f_t$, so the gradient is
    $J_{f_t}^\top \nabla_x \Psi(f_t(z))$. This approximately samples the prior tilted by $e^{-\Psi}$,
    i.e. the posterior over designs when $\Psi = -\log p(y^\star \mid x)$, and keeps designs on the
    prior's support where minimizing $\Psi$ directly does not.

    :param velocity: The design prior's velocity field, called as ``velocity(t, x)`` with a 0-d ``t``,
        e.g. ``MLPVelocity.get_vf_fn()``.
    :param potential: Maps designs of shape ``(N, D)`` to potentials of shape ``(N,)``, e.g.
        ``lambda x: surrogate({"dose": x})`` for a :class:`SurrogatePotential`.
    :param x0: Initial noise of shape ``(N, D)``, drawn from the prior's source distribution.
    :param n_steps: Euler steps from ``t = 0`` to ``t = 1``.
    :param guidance_scale: $b_t$, a constant or a function of ``t``. The right schedule depends on the
        prior's probability path; with ``0`` this samples the prior.
    :returns: Designs of shape ``(N, D)``.
    """
    scale = guidance_scale if callable(guidance_scale) else (lambda _: guidance_scale)
    z, dt = x0.detach(), 1.0 / n_steps
    for step in range(n_steps):
        t = step * dt
        with torch.enable_grad():
            x = z.requires_grad_(True)
            v = velocity(torch.tensor(t, dtype=x.dtype, device=x.device), x)
            (grad,) = torch.autograd.grad(potential(x + (1 - t) * v).sum(), x)
        z = (z + dt * (v - scale(t) * grad)).detach()
    return z
