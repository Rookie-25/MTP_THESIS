r"""Closed-form Black-Scholes reference values.

These formulas are the *ground truth* for the Phase-1 test suite: the
Monte-Carlo price and the AAD Greeks are both benchmarked against them, which
separates "my differentiation is wrong" from "my simulator is wrong".

Everything is written with differentiable ``torch`` ops on scalar tensors, so
the same functions can also be run through autograd as a sanity check on the
analytic derivatives themselves.
"""

from __future__ import annotations

import math
from typing import Union

import torch
from torch import Tensor

__all__ = [
    "standard_normal_cdf",
    "standard_normal_pdf",
    "black_scholes_call",
    "black_scholes_call_delta",
    "black_scholes_call_vega",
    "black_scholes_call_rho",
    "equity_forward_value",
]

ScalarLike = Union[float, Tensor]

_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _t(value: ScalarLike, ref: Tensor) -> Tensor:
    """Coerce ``value`` to a 0-dim tensor matching ``ref``'s device and dtype."""
    if isinstance(value, Tensor):
        return value.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(float(value), device=ref.device, dtype=ref.dtype)


def standard_normal_cdf(x: Tensor) -> Tensor:
    r"""Standard normal CDF :math:`\Phi(x)`, implemented via ``erf``."""
    return 0.5 * (1.0 + torch.erf(x * _INV_SQRT_2))


def standard_normal_pdf(x: Tensor) -> Tensor:
    r"""Standard normal PDF :math:`\varphi(x)`."""
    return _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def _d1_d2(
    s0: Tensor, strike: Tensor, rate: Tensor, sigma: Tensor, maturity: Tensor
) -> tuple[Tensor, Tensor]:
    r"""Return the Black-Scholes :math:`d_1, d_2` terms."""
    vol_sqrt_t = sigma * torch.sqrt(maturity)
    d1 = (torch.log(s0 / strike) + (rate + 0.5 * sigma * sigma) * maturity) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def black_scholes_call(
    s0: ScalarLike,
    strike: ScalarLike,
    rate: ScalarLike,
    sigma: ScalarLike,
    maturity: ScalarLike,
    *,
    notional: float = 1.0,
    reference: Tensor | None = None,
) -> Tensor:
    r"""Black-Scholes price of a European call.

    .. math:: C = S_0\Phi(d_1) - K e^{-rT}\Phi(d_2)

    Args:
        s0: Spot.
        strike: Strike :math:`K`.
        rate: Continuously compounded risk-free rate :math:`r`.
        sigma: Volatility :math:`\sigma`.
        maturity: Time to maturity :math:`T` in years.
        notional: Contract multiplier.
        reference: Optional tensor supplying the target device/dtype when all
            other inputs are plain floats. Defaults to ``float64`` on CPU.

    Returns:
        0-dim tensor holding the price.
    """
    ref = reference if reference is not None else torch.zeros((), dtype=torch.float64)
    s0_t, k_t = _t(s0, ref), _t(strike, ref)
    r_t, sig_t, t_t = _t(rate, ref), _t(sigma, ref), _t(maturity, ref)
    d1, d2 = _d1_d2(s0_t, k_t, r_t, sig_t, t_t)
    price = s0_t * standard_normal_cdf(d1) - k_t * torch.exp(-r_t * t_t) * standard_normal_cdf(d2)
    return notional * price


def black_scholes_call_delta(
    s0: ScalarLike,
    strike: ScalarLike,
    rate: ScalarLike,
    sigma: ScalarLike,
    maturity: ScalarLike,
    *,
    notional: float = 1.0,
    reference: Tensor | None = None,
) -> Tensor:
    r"""Analytic Delta :math:`\partial C/\partial S_0 = \Phi(d_1)`."""
    ref = reference if reference is not None else torch.zeros((), dtype=torch.float64)
    d1, _ = _d1_d2(_t(s0, ref), _t(strike, ref), _t(rate, ref), _t(sigma, ref), _t(maturity, ref))
    return notional * standard_normal_cdf(d1)


def black_scholes_call_vega(
    s0: ScalarLike,
    strike: ScalarLike,
    rate: ScalarLike,
    sigma: ScalarLike,
    maturity: ScalarLike,
    *,
    notional: float = 1.0,
    reference: Tensor | None = None,
) -> Tensor:
    r"""Analytic Vega :math:`\partial C/\partial\sigma = S_0\sqrt{T}\varphi(d_1)`.

    Note:
        This is Vega per **unit** of volatility (not per volatility point).
    """
    ref = reference if reference is not None else torch.zeros((), dtype=torch.float64)
    s0_t, t_t = _t(s0, ref), _t(maturity, ref)
    d1, _ = _d1_d2(s0_t, _t(strike, ref), _t(rate, ref), _t(sigma, ref), t_t)
    return notional * s0_t * torch.sqrt(t_t) * standard_normal_pdf(d1)


def black_scholes_call_rho(
    s0: ScalarLike,
    strike: ScalarLike,
    rate: ScalarLike,
    sigma: ScalarLike,
    maturity: ScalarLike,
    *,
    notional: float = 1.0,
    reference: Tensor | None = None,
) -> Tensor:
    r"""Analytic Rho :math:`\partial C/\partial r = K T e^{-rT}\Phi(d_2)`."""
    ref = reference if reference is not None else torch.zeros((), dtype=torch.float64)
    k_t, r_t, t_t = _t(strike, ref), _t(rate, ref), _t(maturity, ref)
    _, d2 = _d1_d2(_t(s0, ref), k_t, r_t, _t(sigma, ref), t_t)
    return notional * k_t * t_t * torch.exp(-r_t * t_t) * standard_normal_cdf(d2)


def equity_forward_value(
    s0: ScalarLike,
    strike: ScalarLike,
    rate: ScalarLike,
    maturity: ScalarLike,
    *,
    notional: float = 1.0,
    reference: Tensor | None = None,
) -> Tensor:
    r"""Value of a long equity forward: :math:`N\,(S_0 - K e^{-rT})`.

    Used as the closed-form check for the linear (swap) leg. Because the payoff
    is linear, Delta is exactly ``notional`` and Vega is exactly zero -- a
    strong, noise-free test of the AAD plumbing.
    """
    ref = reference if reference is not None else torch.zeros((), dtype=torch.float64)
    s0_t, k_t = _t(s0, ref), _t(strike, ref)
    r_t, t_t = _t(rate, ref), _t(maturity, ref)
    return notional * (s0_t - k_t * torch.exp(-r_t * t_t))
