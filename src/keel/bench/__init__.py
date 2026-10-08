from __future__ import annotations

__all__ = ["main", "run_batched", "run_sequential", "scenarios"]


def __getattr__(name: str) -> object:
    # Re-exported lazily so `keel --help` does not pay for importing the whole
    # scheduler and engine stack.
    if name in {"main", "run_batched", "run_sequential", "scenarios", "Scenario"}:
        from keel.bench import runner

        return getattr(runner, name)
    raise AttributeError(name)
