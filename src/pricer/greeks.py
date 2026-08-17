r"""Reverse-mode AAD Greeks and a bump-and-revalue reference implementation.

Why AAD
-------
For a valuation :math:`V(\theta_1,\dots,\theta_n)` implemented as a
computational graph of cost :math:`C`, reverse-mode adjoint algorithmic
differentiation returns **all** :math:`n` sensitivities in a single backward
sweep of cost :math:`O(C)` -- typically 2-5x the forward pass, *independent of*
:math:`n`. Bump-and-revalue needs :math:`2n` extra forward passes for central
differences, i.e. :math:`O(nC)`. On a realistic XVA book with hundreds of risk
factors that is the difference between an overnight batch and an intraday one,
and it is the central performance claim this project sets out to demonstrate.

Why the finite-difference reference still matters
------------------------------------------------
AAD differentiates *the code*, not the mathematics. A bug in the tape (a
detached tensor, an in-place write, an accidental ``float`` cast) produces a
silently wrong -- but perfectly smooth -- gradient. Bump-and-revalue is an
independent oracle. Two conditions make the comparison meaningful:

1. **Common random numbers.** Both methods must differentiate the same MC
   realisation. This is enforced by requiring the caller to pass a
   :data:`~src.pricer.options.PriceFn` closure that has already captured a
   fixed Brownian sample. Without CRN the bumped estimator's noise
   (:math:`O(1/(h\sqrt{M}))`) dwarfs the signal.
2. **Double precision.** With ``float32`` (:math:`\epsilon \approx 1.2\times
   10^{-7}`) a relative bump of :math:`10^{-4}` leaves roughly three
   significant digits in the difference quotient -- far short of a
   :math:`10^{-3}` agreement target. ``float64`` is therefore mandatory for
   validation; ``float32`` remains the right choice for GPU throughput once
   correctness is established.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import torch
from torch import Tensor

from src.pricer.options import PriceFn

__all__ = [
    "GreekResult",
    "GreekComparison",
    "aad_greeks",
    "finite_difference_greeks",
    "bump_and_revalue_greeks",
    "compare_greeks",
    "format_comparison",
]


@dataclass(frozen=True)
class GreekResult:
    """Sensitivities produced by one differentiation method.

    Attributes:
        method: Human-readable method tag, e.g. ``"aad"`` or ``"fd-central"``.
        price: Base valuation at the unperturbed parameters.
        greeks: Mapping from parameter name to :math:`\\partial V/\\partial\\theta`.
        wall_time_s: Wall-clock seconds spent producing *all* sensitivities,
            including the base valuation. CUDA work is synchronised before the
            clock is stopped, so the number is comparable across devices.
        n_valuations: Number of full forward valuations consumed. ``1`` for AAD
            (plus one backward sweep), ``2n + 1`` for central differences.
    """

    method: str
    price: float
    greeks: Dict[str, float]
    wall_time_s: float
    n_valuations: int


@dataclass(frozen=True)
class GreekComparison:
    """Element-wise error report between a reference and a candidate result.

    Attributes:
        reference: The result treated as ground truth.
        candidate: The result under test.
        absolute_error: ``|candidate - reference|`` per parameter.
        relative_error: ``|candidate - reference| / max(|reference|, floor)``
            per parameter.
        price_absolute_error: Absolute difference of the two base prices.
    """

    reference: GreekResult
    candidate: GreekResult
    absolute_error: Dict[str, float]
    relative_error: Dict[str, float]
    price_absolute_error: float

    @property
    def max_absolute_error(self) -> float:
        """Largest absolute error across parameters (``0.0`` if empty)."""
        return max(self.absolute_error.values(), default=0.0)

    @property
    def max_relative_error(self) -> float:
        """Largest relative error across parameters (``0.0`` if empty)."""
        return max(self.relative_error.values(), default=0.0)

    @property
    def speedup(self) -> float:
        """Candidate-over-reference wall-time ratio (``inf`` on a zero clock)."""
        if self.candidate.wall_time_s <= 0.0:
            return float("inf")
        return self.reference.wall_time_s / self.candidate.wall_time_s


def _synchronize(device: torch.device) -> None:
    """Block until queued CUDA work completes, so timings are not optimistic."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _as_leaf(value: Tensor | float, *, dtype: Optional[torch.dtype] = None) -> Tensor:
    """Return a fresh 0-dim autograd leaf holding ``value``.

    Cloning and detaching isolates the differentiation from any graph the caller
    may already have attached to ``value``, which prevents gradient
    contamination when the same parameter dict is reused across methods.

    Args:
        value: Scalar tensor or Python float.
        dtype: Optional dtype override.

    Returns:
        A 0-dim tensor with ``requires_grad=True``.

    Raises:
        ValueError: If ``value`` is a non-scalar tensor.
    """
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"parameters must be scalar, got shape {tuple(value.shape)}")
        leaf = value.detach().clone().reshape(())
    else:
        leaf = torch.as_tensor(float(value), dtype=dtype or torch.float64)
    if dtype is not None:
        leaf = leaf.to(dtype)
    return leaf.requires_grad_(True)


def _check_scalar_price(price: Tensor) -> None:
    if not isinstance(price, Tensor):
        raise TypeError(f"price_fn must return a torch.Tensor, got {type(price).__name__}")
    if price.numel() != 1:
        raise ValueError(
            f"price_fn must return a scalar; got shape {tuple(price.shape)}. "
            "Reduce over paths (e.g. .mean()) inside the closure."
        )


def aad_greeks(
    price_fn: PriceFn,
    params: Mapping[str, Tensor | float],
    *,
    wrt: Optional[Iterable[str]] = None,
    retain_graph: bool = False,
) -> GreekResult:
    r"""Compute all requested sensitivities in **one** reverse-mode sweep.

    Args:
        price_fn: Valuation closure with fixed randomness. See
            :data:`~src.pricer.options.PriceFn`.
        params: Parameter values. Non-tensor entries are promoted to
            ``float64`` leaves.
        wrt: Subset of parameter names to differentiate. Defaults to every key
            of ``params``. Names outside this subset are still passed to
            ``price_fn`` (as leaves) but no gradient is requested for them.
        retain_graph: Keep the tape alive after the sweep, e.g. to compute a
            second-order sensitivity afterwards.

    Returns:
        A :class:`GreekResult` tagged ``"aad"`` with ``n_valuations == 1``.

    Raises:
        KeyError: If a name in ``wrt`` is absent from ``params``.
        ValueError: If ``price_fn`` does not return a scalar, or if a requested
            parameter turns out to be disconnected from the price (a strong
            signal that the tape is broken -- typically a ``.detach()``, an
            ``.item()``, or a NumPy round-trip inside the closure).
    """
    names = list(params) if wrt is None else list(wrt)
    missing = [name for name in names if name not in params]
    if missing:
        raise KeyError(f"parameters requested in 'wrt' but absent from 'params': {missing}")

    leaves: MutableMapping[str, Tensor] = {key: _as_leaf(val) for key, val in params.items()}
    targets = [leaves[name] for name in names]
    device = targets[0].device if targets else torch.device("cpu")

    _synchronize(device)
    start = perf_counter()

    price = price_fn(leaves)
    _check_scalar_price(price)

    grads = torch.autograd.grad(
        outputs=price.reshape(()),
        inputs=targets,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    _synchronize(device)
    elapsed = perf_counter() - start

    disconnected = [name for name, grad in zip(names, grads) if grad is None]
    if disconnected:
        raise ValueError(
            f"no gradient path from the price to {disconnected}; the autograd tape is "
            "broken (look for .detach(), .item(), torch.no_grad(), or a NumPy "
            "conversion inside price_fn)"
        )

    return GreekResult(
        method="aad",
        price=float(price.detach()),
        greeks={name: float(grad.detach()) for name, grad in zip(names, grads)},
        wall_time_s=elapsed,
        n_valuations=1,
    )


def finite_difference_greeks(
    price_fn: PriceFn,
    params: Mapping[str, Tensor | float],
    *,
    wrt: Optional[Iterable[str]] = None,
    rel_step: float = 1e-4,
    scheme: str = "central",
    step_floor: float = 1.0,
) -> GreekResult:
    r"""Compute sensitivities by bump-and-revalue (the validation oracle).

    The bump for parameter :math:`\theta` is
    :math:`h = \texttt{rel\_step}\cdot\max(|\theta|, \texttt{step\_floor})`.
    The floor keeps the step well-scaled for parameters near zero (a bare
    relative step would collapse to ``0`` at :math:`\theta = 0`).

    Args:
        price_fn: Valuation closure with **fixed** randomness -- this is what
            supplies common random numbers and is non-negotiable for accuracy.
        params: Parameter values.
        wrt: Subset of names to bump. Defaults to all keys.
        rel_step: Relative bump size. With ``float64`` and a central scheme the
            total error is minimised near :math:`10^{-5}`-:math:`10^{-4}`:
            truncation falls as :math:`O(h^2)` while round-off grows as
            :math:`O(\epsilon/h)`.
        scheme: ``"central"`` (:math:`O(h^2)`, ``2n + 1`` valuations) or
            ``"forward"`` (:math:`O(h)`, ``n + 1`` valuations).
        step_floor: Lower bound on the parameter magnitude used to scale ``h``.

    Returns:
        A :class:`GreekResult` tagged ``"fd-central"`` or ``"fd-forward"``.

    Raises:
        KeyError: If a name in ``wrt`` is absent from ``params``.
        ValueError: On an unknown ``scheme``, a non-positive ``rel_step``, or a
            non-scalar price.
    """
    if scheme not in {"central", "forward"}:
        raise ValueError(f"scheme must be 'central' or 'forward', got {scheme!r}")
    if rel_step <= 0.0:
        raise ValueError(f"rel_step must be positive, got {rel_step}")

    names = list(params) if wrt is None else list(wrt)
    missing = [name for name in names if name not in params]
    if missing:
        raise KeyError(f"parameters requested in 'wrt' but absent from 'params': {missing}")

    base: Dict[str, Tensor] = {}
    for key, value in params.items():
        if isinstance(value, Tensor):
            base[key] = value.detach().clone().reshape(())
        else:
            base[key] = torch.as_tensor(float(value), dtype=torch.float64)

    device = base[names[0]].device if names else torch.device("cpu")

    def evaluate(overrides: Mapping[str, Tensor]) -> float:
        trial = dict(base)
        trial.update(overrides)
        price = price_fn(trial)
        _check_scalar_price(price)
        return float(price.detach())

    _synchronize(device)
    start = perf_counter()

    with torch.no_grad():
        base_price = evaluate({})
        n_valuations = 1
        greeks: Dict[str, float] = {}

        for name in names:
            theta = base[name]
            step = rel_step * max(abs(float(theta)), step_floor)
            up = evaluate({name: theta + step})
            n_valuations += 1
            if scheme == "central":
                down = evaluate({name: theta - step})
                n_valuations += 1
                greeks[name] = (up - down) / (2.0 * step)
            else:
                greeks[name] = (up - base_price) / step

    _synchronize(device)
    elapsed = perf_counter() - start

    return GreekResult(
        method=f"fd-{scheme}",
        price=base_price,
        greeks=greeks,
        wall_time_s=elapsed,
        n_valuations=n_valuations,
    )


#: Explicit alias matching the risk-management term of art.
bump_and_revalue_greeks = finite_difference_greeks


def compare_greeks(
    reference: GreekResult,
    candidate: GreekResult,
    *,
    relative_floor: float = 1e-8,
) -> GreekComparison:
    """Build an error report between two :class:`GreekResult` objects.

    Args:
        reference: Ground truth (typically the finite-difference or analytic
            result).
        candidate: Result under test (typically AAD).
        relative_floor: Lower bound on ``|reference|`` when forming the
            relative error, so that a near-zero reference Greek does not
            produce a meaningless ratio.

    Returns:
        The :class:`GreekComparison` report.

    Raises:
        ValueError: If the two results do not cover the same parameter names.
    """
    if set(reference.greeks) != set(candidate.greeks):
        raise ValueError(
            "cannot compare results over different parameters: "
            f"{sorted(reference.greeks)} vs {sorted(candidate.greeks)}"
        )

    absolute: Dict[str, float] = {}
    relative: Dict[str, float] = {}
    for name, ref_value in reference.greeks.items():
        error = abs(candidate.greeks[name] - ref_value)
        absolute[name] = error
        relative[name] = error / max(abs(ref_value), relative_floor)

    return GreekComparison(
        reference=reference,
        candidate=candidate,
        absolute_error=absolute,
        relative_error=relative,
        price_absolute_error=abs(candidate.price - reference.price),
    )


def format_comparison(comparison: GreekComparison) -> str:
    """Render a :class:`GreekComparison` as a fixed-width console table.

    Args:
        comparison: The report to render.

    Returns:
        A multi-line string suitable for printing in benchmarks or test output.
    """
    ref, cand = comparison.reference, comparison.candidate
    lines = [
        f"{'param':<10}{ref.method:>18}{cand.method:>18}{'abs err':>14}{'rel err':>12}",
        "-" * 72,
    ]
    for name in sorted(comparison.absolute_error):
        lines.append(
            f"{name:<10}{ref.greeks[name]:>18.10f}{cand.greeks[name]:>18.10f}"
            f"{comparison.absolute_error[name]:>14.3e}{comparison.relative_error[name]:>12.3e}"
        )
    lines.append("-" * 72)
    lines.append(
        f"price: {ref.method}={ref.price:.10f}  {cand.method}={cand.price:.10f}  "
        f"|diff|={comparison.price_absolute_error:.3e}"
    )
    lines.append(
        f"valuations: {ref.method}={ref.n_valuations} ({ref.wall_time_s * 1e3:.2f} ms)  "
        f"{cand.method}={cand.n_valuations} ({cand.wall_time_s * 1e3:.2f} ms)  "
        f"speedup={comparison.speedup:.2f}x"
    )
    return "\n".join(lines)
