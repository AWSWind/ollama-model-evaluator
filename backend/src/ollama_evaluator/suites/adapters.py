"""Process-global registry of public-benchmark adapters.

This module exposes the three-function surface the CLI (``convert``
and ``run --dataset-mode``) uses to look adapters up by their
user-visible name:

* :func:`register_adapter` — add (or replace) an adapter keyed by its
  :attr:`BenchmarkAdapter.ADAPTER_NAME`.
* :func:`get_adapter` — retrieve a previously registered adapter;
  raises :class:`UnknownAdapterError` (a :class:`KeyError` subclass)
  when the name is not registered.
* :func:`list_adapters` — enumerate every registered name in sorted
  order so CLI help text and tests are deterministic.

Re-registration replaces the prior entry, matching the
:mod:`ollama_evaluator.metrics` registry's semantics. That lets tests
install lightweight fakes without needing a special "unregister" hook
and lets ``uvicorn --reload`` recover cleanly after a source reload.

The per-adapter modules (``mmlu``, ``hellaswag``, ``truthfulqa``,
``gsm8k``, ``humaneval``) each call :func:`register_adapter` at import
time; the auto-import at the bottom of this module ensures that a
single ``import ollama_evaluator.suites.adapters`` is enough to make
every v1 adapter resolvable.
"""

from __future__ import annotations

from .adapter_base import BenchmarkAdapter


class UnknownAdapterError(KeyError):
    """Raised by :func:`get_adapter` when the adapter name is not registered.

    Subclasses :class:`KeyError` so callers that use ``dict``-style
    patterns still catch the error; the distinct class lets the CLI
    render a better message when the user asks for an unknown adapter.
    """


_REGISTRY: dict[str, BenchmarkAdapter] = {}
"""Process-global ``ADAPTER_NAME -> BenchmarkAdapter`` mapping."""


def register_adapter(adapter: BenchmarkAdapter) -> None:
    """Register ``adapter`` under its :attr:`ADAPTER_NAME` in the process registry.

    Re-registration with the same name replaces the prior entry. This
    is load-bearing for tests that install fakes and for import-time
    reloads. The function performs no structural validation beyond
    reading ``adapter.ADAPTER_NAME``; future versions may assert a
    non-empty name and a callable ``rows_to_suite``.
    """
    _REGISTRY[adapter.ADAPTER_NAME] = adapter


def get_adapter(name: str) -> BenchmarkAdapter:
    """Return the adapter registered under ``name``.

    Raises :class:`UnknownAdapterError` (a :class:`KeyError` subclass)
    with ``name`` as the sole argument when no such adapter is
    registered. The error's ``args[0]`` is the missing name so callers
    surfacing the error to users (CLI, REST API) can render it
    directly.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - trivial re-raise
        raise UnknownAdapterError(name) from exc


def list_adapters() -> list[str]:
    """Return every registered adapter name in lexicographically sorted order."""
    return sorted(_REGISTRY)


__all__ = [
    "UnknownAdapterError",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]


# ---------------------------------------------------------------------------
# Auto-import of the built-in adapters.
# ---------------------------------------------------------------------------
#
# The benchmark adapter modules register themselves at import time. The
# imports live at the bottom of this module so ``register_adapter`` is
# defined before any adapter module runs its ``register_adapter(...)``
# call. A ``_BUILTINS_REGISTERED`` guard makes ``importlib.reload`` cheap
# without being strictly required for correctness (registration has
# replacement semantics).

from . import gsm8k as _gsm8k  # noqa: E402, F401
from . import hellaswag as _hellaswag  # noqa: E402, F401
from . import humaneval as _humaneval  # noqa: E402, F401
from . import mmlu as _mmlu  # noqa: E402, F401
from . import truthfulqa as _truthfulqa  # noqa: E402, F401

_BUILTINS_REGISTERED = False


def _ensure_builtins_registered() -> None:
    """Register every built-in adapter exactly once per interpreter."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    register_adapter(_mmlu.MMluAdapter())
    register_adapter(_hellaswag.HellaSwagAdapter())
    register_adapter(_truthfulqa.TruthfulQaAdapter())
    register_adapter(_gsm8k.Gsm8kAdapter())
    register_adapter(_humaneval.HumanEvalAdapter())
    _BUILTINS_REGISTERED = True


_ensure_builtins_registered()
