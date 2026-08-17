"""PyTest configuration: make the repository root importable and share fixtures.

The ``pythonpath`` setting in ``pyproject.toml`` normally handles the import
path; this module is a belt-and-braces fallback so ``pytest`` also works when
invoked from an arbitrary working directory or with an older plugin set.
"""

from __future__ import annotations

import sys
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
