"""PyTest configuration: make the repository root importable and share fixtures.

The ``pythonpath`` setting in ``pyproject.toml`` normally handles the import
path; this module is a belt-and-braces fallback so ``pytest`` also works when
invoked from an arbitrary working directory or with an older plugin set.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="session", autouse=True)
def _deterministic_torch() -> None:
    """Pin global RNG state so unseeded helpers cannot make tests flaky."""
    torch.manual_seed(20260813)


@pytest.fixture(params=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """Parametrise a test over CPU and (when present) CUDA."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    return torch.device(request.param)


# ==========================================================================
# Strict tensor->scalar conversion check (opt-in)
# ==========================================================================
#
# Converting a graph-attached tensor to a Python scalar (`float(t)`, `t.item()`)
# emits a UserWarning on newer PyTorch builds but NOT on older ones -- torch
# 2.4.0 is silent where Colab's build warns. That version gap let three such
# sites accumulate unnoticed and cost three separate round-trips to find.
#
# This hook implements the check directly, so it is version-independent. It is
# opt-in because it monkeypatches Tensor.__float__ / Tensor.item for the whole
# session:
#
#     STRICT_TENSOR_SCALAR=1 python -m pytest tests/ -q
#
# `pyproject.toml` additionally promotes PyTorch's own warning to an error, so
# builds that *do* emit it fail loudly without needing this flag.
_STRICT_SCALAR_HITS: list = []


def pytest_configure(config: "pytest.Config") -> None:
    """Optionally patch tensor->scalar conversions to record grad-attached use."""
    if os.environ.get("STRICT_TENSOR_SCALAR") != "1":
        return

    original_float = torch.Tensor.__float__
    original_item = torch.Tensor.item

    def _record(tensor: torch.Tensor, kind: str) -> None:
        if not tensor.requires_grad:
            return
        frames = [
            frame
            for frame in traceback.extract_stack()[:-2]
            if "test_" in frame.filename or f"{os.sep}src{os.sep}" in frame.filename
        ]
        if frames:
            last = frames[-1]
            _STRICT_SCALAR_HITS.append((last.filename, last.lineno, last.line, kind))

    def patched_float(self):  # type: ignore[no-untyped-def]
        _record(self, "float()")
        return original_float(self)

    def patched_item(self):  # type: ignore[no-untyped-def]
        _record(self, ".item()")
        return original_item(self)

    torch.Tensor.__float__ = patched_float  # type: ignore[method-assign]
    torch.Tensor.item = patched_item  # type: ignore[method-assign]


def pytest_sessionfinish(session, exitstatus) -> None:  # type: ignore[no-untyped-def]
    """Report any graph-attached scalar conversions found during the session."""
    if os.environ.get("STRICT_TENSOR_SCALAR") != "1":
        return
    if not _STRICT_SCALAR_HITS:
        print("\nSTRICT TENSOR->SCALAR: clean\n")
        return

    counts: dict = {}
    for entry in _STRICT_SCALAR_HITS:
        counts[entry] = counts.get(entry, 0) + 1
    print(
        f"\nSTRICT TENSOR->SCALAR: {len(counts)} site(s) convert a "
        "requires_grad tensor -- add .detach() before the conversion:\n"
    )
    for (filename, lineno, source, kind), count in sorted(counts.items()):
        print(f"  {Path(filename).name}:{lineno}  [{kind}, {count}x]")
        print(f"      {(source or '').strip()}")
    print()
