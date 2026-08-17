r"""Vectorised payoffs and Monte-Carlo valuation on pre-generated paths.

Two instrument families are covered in Phase 1:

**European call** -- the non-linear benchmark. Its payoff
:math:`(S_T-K)^+` is Lipschitz but kinked at the strike, which is exactly the
regime where pathwise (AAD) Greeks are unbiased while bump-and-revalue needs
common random numbers to be usable.

**Equity total-return swap leg (forward-equivalent)** -- the linear benchmark
and the building block for Phase 2. Its mark-to-market

.. math:: V_t = N\left(S_t - K\right)e^{-r(T-t)}

is produced *for every path and every time step*, i.e. an
``(n_paths, n_steps + 1)`` exposure surface. That surface is precisely the
input required by the CVA/DVA expected-exposure machinery in ``src/xva``.

All functions are pure and fully vectorised: no Python loop runs over paths or
over time, so the whole valuation is a handful of fused tensor ops that the
autograd tape can differentiate in one reverse sweep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Union

import torch
from torch import Tensor

from src.models.gbm import GBMSimulator, ScalarLike, simulate_gbm

__all__ = [
    "MCPrice",
    "SwapLeg",
    "PriceFn",
    "european_call_payoff",
    "european_put_payoff",
    "mc_discounted_price",
    "european_call_price",
    "equity_forward_mtm",
    "portfolio_swap_mtm",
    "portfolio_swap_price",
    "resolve_rate_and_drift",
    "make_european_call_price_fn",
    "make_portfolio_swap_price_fn",
]

#: A valuation closure mapping a parameter dictionary to a scalar price. The
#: closure must capture a *fixed* Brownian sample so that repeated calls with
#: perturbed parameters share common random numbers.
PriceFn = Callable[[Mapping[str, Tensor]], Tensor]


@dataclass(frozen=True)
class MCPrice:
    """Result of a Monte-Carlo valuation.

    Attributes:
        value: Differentiable 0-dim tensor holding the MC estimate. Call
            ``backward()`` on it (or pass it to :func:`torch.autograd.grad`) to
            obtain AAD sensitivities.
        std_error: Detached 0-dim tensor with the sample standard error
            :math:`\\hat{s}/\\sqrt{M}`. Conservative under antithetic sampling,
            because paired paths are negatively correlated.
        n_paths: Number of paths used.
    """

    value: Tensor
    std_error: Tensor
    n_paths: int

    def __float__(self) -> float:
        return float(self.value.detach())

    @property
    def confidence_interval(self) -> tuple[float, float]:
        """Two-sided 95% MC confidence interval for the price."""
        centre = float(self.value.detach())
        half_width = 1.959963984540054 * float(self.std_error)
        return centre - half_width, centre + half_width


@dataclass(frozen=True)
class SwapLeg:
    """A single linear (forward-equivalent) leg of an equity total-return swap.

    Attributes:
        notional: Signed notional :math:`N`. Negative means the leg is paid.
        strike: Contractual forward price :math:`K`.
        maturity: Leg maturity :math:`T_i` in years. Must lie on the
            simulation grid.
    """

    notional: float
    strike: float
    maturity: float

    def __post_init__(self) -> None:
        if self.maturity <= 0.0:
            raise ValueError(f"leg maturity must be positive, got {self.maturity}")


def _t(value: ScalarLike, ref: Tensor) -> Tensor:
    """Coerce ``value`` to a 0-dim tensor matching ``ref``'s device and dtype."""
    if isinstance(value, Tensor):
        return value.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(float(value), device=ref.device, dtype=ref.dtype)


def _check_paths(paths: Tensor) -> None:
    if paths.ndim != 2:
        raise ValueError(f"paths must have shape (n_paths, n_steps + 1), got {tuple(paths.shape)}")


def _nearest_grid_index(times: Tensor, target: float) -> int:
    """Return the grid index closest to ``target``, validating the match.

    Args:
        times: Monotone observation grid of shape ``(n_steps + 1,)``.
        target: Requested time in years.

    Returns:
        Integer index into ``times``.

    Raises:
        ValueError: If ``target`` is further than half a step from the grid,
            which would silently reprice the instrument at the wrong date.
    """
    grid = times.detach().reshape(-1)
    index = int(torch.argmin(torch.abs(grid - target)))
    if grid.numel() > 1:
        step = float(grid[1] - grid[0])
        if abs(float(grid[index]) - target) > 0.5 * step + 1e-12:
            raise ValueError(
                f"time {target} is not on the simulation grid "
                f"(closest grid point is {float(grid[index])})"
            )
    return index


def european_call_payoff(terminal_prices: Tensor, strike: ScalarLike) -> Tensor:
    r"""Terminal payoff :math:`(S_T - K)^+`.

    Args:
        terminal_prices: Tensor of shape ``(n_paths,)``.
        strike: Strike :math:`K`.

    Returns:
        Tensor of shape ``(n_paths,)``.

    Note:
        The kink at :math:`S_T = K` is a null set under the GBM law, so autograd
        returns the correct pathwise derivative almost surely. ``clamp_min``
        assigns subgradient ``0`` exactly at the kink.
    """
    return torch.clamp(terminal_prices - _t(strike, terminal_prices), min=0.0)


def european_put_payoff(terminal_prices: Tensor, strike: ScalarLike) -> Tensor:
    r"""Terminal payoff :math:`(K - S_T)^+`."""
    return torch.clamp(_t(strike, terminal_prices) - terminal_prices, min=0.0)


def mc_discounted_price(
    payoff: Tensor,
    rate: ScalarLike,
    maturity: ScalarLike,
) -> MCPrice:
    r"""Discount a per-path payoff to :math:`t=0` and average it.

    Args:
        payoff: Per-path payoff of shape ``(n_paths,)``.
        rate: Continuously compounded discount rate :math:`r`.
        maturity: Payment date :math:`T` in years.

    Returns:
        The :class:`MCPrice` estimate.

    Raises:
        ValueError: If ``payoff`` is not 1-dimensional.
    """
    if payoff.ndim != 1:
        raise ValueError(f"payoff must be 1-dimensional, got shape {tuple(payoff.shape)}")

    discount = torch.exp(-_t(rate, payoff) * _t(maturity, payoff))
    discounted = discount * payoff
    n_paths = discounted.shape[0]
    value = discounted.mean()
    if n_paths > 1:
        std_error = discounted.detach().std(unbiased=True) / math.sqrt(n_paths)
    else:
        std_error = torch.zeros((), device=payoff.device, dtype=payoff.dtype)
    return MCPrice(value=value, std_error=std_error, n_paths=n_paths)


def european_call_price(
    paths: Tensor,
    strike: ScalarLike,
    rate: ScalarLike,
    maturity: ScalarLike,
    *,
    notional: float = 1.0,
) -> MCPrice:
    r"""Monte-Carlo price of a European call on the terminal column of ``paths``.

    Args:
        paths: Simulated paths of shape ``(n_paths, n_steps + 1)``.
        strike: Strike :math:`K`.
        rate: Discount rate :math:`r`.
        maturity: Maturity :math:`T`, matching the final grid point.
        notional: Contract multiplier.

    Returns:
        The :class:`MCPrice` estimate; ``value`` remains attached to the
        autograd graph of ``paths``.
    """
    _check_paths(paths)
    payoff = notional * european_call_payoff(paths[:, -1], strike)
    return mc_discounted_price(payoff, rate, maturity)


def equity_forward_mtm(
    paths: Tensor,
    times: Tensor,
    strike: ScalarLike,
    rate: ScalarLike,
    maturity: float,
    *,
    notional: float = 1.0,
) -> Tensor:
    r"""Mark-to-market surface of a long equity forward / TRS leg.

    .. math:: V_t = N\,(S_t - K)\,e^{-r(T-t)},\qquad t \le T,

    and :math:`V_t = 0` for :math:`t > T` (the leg has settled). At
    :math:`t = T` this collapses to the settlement amount
    :math:`N(S_T - K)`.

    Args:
        paths: Paths of shape ``(n_paths, n_steps + 1)``.
        times: Observation grid of shape ``(n_steps + 1,)``.
        strike: Contractual forward price :math:`K`.
        rate: Discount rate :math:`r`.
        maturity: Leg maturity :math:`T`.
        notional: Signed notional.

    Returns:
        Exposure surface of shape ``(n_paths, n_steps + 1)``, differentiable
        w.r.t. every model parameter feeding ``paths``.

    Raises:
        ValueError: On shape mismatch between ``paths`` and ``times``.
    """
    _check_paths(paths)
    if times.ndim != 1 or times.shape[0] != paths.shape[1]:
        raise ValueError(
            f"times must have shape ({paths.shape[1]},) to match paths, got {tuple(times.shape)}"
        )

    times_row = times.to(device=paths.device, dtype=paths.dtype).reshape(1, -1)
    time_to_maturity = torch.clamp(maturity - times_row, min=0.0)
    discount = torch.exp(-_t(rate, paths) * time_to_maturity)
    alive = (times_row <= maturity + 1e-12).to(paths.dtype)
    return notional * alive * (paths - _t(strike, paths)) * discount


def portfolio_swap_mtm(
    paths: Tensor,
    times: Tensor,
    legs: Sequence[SwapLeg],
    rate: ScalarLike,
) -> Tensor:
    r"""Aggregate mark-to-market surface of a portfolio of linear swap legs.

    Netting is assumed within the portfolio (single netting set), so leg MtMs
    are summed before any exposure floor is applied downstream.

    Args:
        paths: Paths of shape ``(n_paths, n_steps + 1)``.
        times: Observation grid of shape ``(n_steps + 1,)``.
        legs: Legs to aggregate. Signed notionals encode pay/receive.
        rate: Discount rate :math:`r`.

    Returns:
        Netted exposure surface of shape ``(n_paths, n_steps + 1)``.

    Raises:
        ValueError: If ``legs`` is empty.
    """
    if not legs:
        raise ValueError("portfolio must contain at least one leg")

    total = torch.zeros_like(paths)
    for leg in legs:
        total = total + equity_forward_mtm(
            paths, times, leg.strike, rate, leg.maturity, notional=leg.notional
        )
    return total


def portfolio_swap_price(
    paths: Tensor,
    times: Tensor,
    legs: Sequence[SwapLeg],
    rate: ScalarLike,
) -> MCPrice:
    r"""Monte-Carlo present value of a portfolio of linear swap legs.

    Each leg is settled at its own maturity and discounted to :math:`t=0`:

    .. math:: V_0 = \sum_i N_i\, e^{-r T_i}\, \mathbb{E}\!\left[S_{T_i} - K_i\right].

    Args:
        paths: Paths of shape ``(n_paths, n_steps + 1)``.
        times: Observation grid of shape ``(n_steps + 1,)``.
        legs: Legs to value. Every ``leg.maturity`` must lie on ``times``.
        rate: Discount rate :math:`r`.

    Returns:
        The :class:`MCPrice` estimate of the netted portfolio value.

    Raises:
        ValueError: If ``legs`` is empty or a leg maturity is off-grid.
    """
    _check_paths(paths)
    if not legs:
        raise ValueError("portfolio must contain at least one leg")

    rate_t = _t(rate, paths)
    per_path = torch.zeros(paths.shape[0], device=paths.device, dtype=paths.dtype)
    for leg in legs:
        index = _nearest_grid_index(times, leg.maturity)
        maturity_t = _t(leg.maturity, paths)
        settlement = leg.notional * (paths[:, index] - _t(leg.strike, paths))
        per_path = per_path + settlement * torch.exp(-rate_t * maturity_t)

    n_paths = per_path.shape[0]
    value = per_path.mean()
    if n_paths > 1:
        std_error = per_path.detach().std(unbiased=True) / math.sqrt(n_paths)
    else:
        std_error = torch.zeros((), device=paths.device, dtype=paths.dtype)
    return MCPrice(value=value, std_error=std_error, n_paths=n_paths)


def resolve_rate_and_drift(
    params: Mapping[str, Tensor],
    fallback_rate: Optional[ScalarLike],
) -> tuple[ScalarLike, ScalarLike]:
    """Extract the discount rate and simulation drift from a parameter dict.

    Conventions:
        * ``params["rate"]`` -- if present, used as **both** the drift and the
          discount rate, so the resulting sensitivity is total Rho.
        * ``params["mu"]`` -- if present, overrides the drift only (real-world
          measure); the discount rate still comes from ``rate``.

    Args:
        params: Parameter dictionary supplied to the price closure.
        fallback_rate: Rate captured at closure-construction time.

    Returns:
        ``(rate, drift)``.

    Raises:
        KeyError: If no rate is available from either source.
    """
    if "rate" in params:
        rate: ScalarLike = params["rate"]
    elif fallback_rate is not None:
        rate = fallback_rate
    else:
        raise KeyError("no discount rate available: pass params['rate'] or the 'rate' argument")
    drift: ScalarLike = params["mu"] if "mu" in params else rate
    return rate, drift


def make_european_call_price_fn(
    simulator: GBMSimulator,
    dW: Tensor,
    strike: float,
    *,
    rate: Optional[ScalarLike] = None,
    notional: float = 1.0,
) -> PriceFn:
    r"""Build a differentiable European-call valuation closure with fixed randomness.

    The returned closure captures ``dW``, which is what makes AAD and
    bump-and-revalue directly comparable: both differentiate the *same*
    realisation of the Monte-Carlo estimator, so their difference isolates the
    differentiation scheme rather than sampling noise.

    Args:
        simulator: Configured :class:`~src.models.gbm.GBMSimulator`.
        dW: Fixed Brownian increments of shape ``(n_paths, n_steps)``.
        strike: Strike :math:`K`.
        rate: Default discount rate, used unless ``params["rate"]`` is given.
        notional: Contract multiplier.

    Returns:
        A callable ``price_fn(params) -> Tensor`` expecting keys ``"s0"`` and
        ``"sigma"``, optionally ``"rate"`` and ``"mu"``.
    """

    def price_fn(params: Mapping[str, Tensor]) -> Tensor:
        resolved_rate, drift = resolve_rate_and_drift(params, rate)
        paths = simulate_gbm(params["s0"], drift, params["sigma"], dW, simulator.dt)
        return european_call_price(
            paths, strike, resolved_rate, simulator.maturity, notional=notional
        ).value

    return price_fn


def make_portfolio_swap_price_fn(
    simulator: GBMSimulator,
    dW: Tensor,
    legs: Sequence[SwapLeg],
    *,
    rate: Optional[ScalarLike] = None,
) -> PriceFn:
    r"""Build a differentiable portfolio-swap valuation closure with fixed randomness.

    Args:
        simulator: Configured :class:`~src.models.gbm.GBMSimulator`.
        dW: Fixed Brownian increments of shape ``(n_paths, n_steps)``.
        legs: Portfolio legs.
        rate: Default discount rate, used unless ``params["rate"]`` is given.

    Returns:
        A callable ``price_fn(params) -> Tensor`` expecting keys ``"s0"`` and
        ``"sigma"``, optionally ``"rate"`` and ``"mu"``.

    Note:
        A linear leg has *analytic* Vega zero, since
        :math:`\mathbb{E}[S_T] = S_0 e^{\mu T}` does not involve
        :math:`\sigma`. The finite-sample MC estimate is only zero up to
        Monte-Carlo error, but because ``dW`` is fixed, AAD and
        bump-and-revalue reproduce the *same* residual -- which makes this a
        sharp regression test of the AAD wiring on a degenerate Greek.
    """
    legs = tuple(legs)
    times = simulator.time_grid()

    def price_fn(params: Mapping[str, Tensor]) -> Tensor:
        resolved_rate, drift = resolve_rate_and_drift(params, rate)
        paths = simulate_gbm(params["s0"], drift, params["sigma"], dW, simulator.dt)
        return portfolio_swap_price(paths, times, legs, resolved_rate).value

    return price_fn
