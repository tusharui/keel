from __future__ import annotations

from typing import ClassVar


class KeelError(Exception):
    """Base for everything this package raises deliberately.

    ``retryable`` is a class-level property, not something a raise site can flip.
    Deciding it once per error type keeps retry policy and error taxonomy in the
    same place, and stops call sites from quietly marking a failure retryable to
    make their own retry loop terminate.
    """

    retryable: ClassVar[bool] = False


# --- configuration and validation -------------------------------------------------


class ConfigError(KeelError):
    """Invalid settings. Fails fast at startup rather than mid-traffic."""


class ValidationError(KeelError):
    """Request is malformed. Retrying an identical request cannot succeed."""


class SchemaRepairExhausted(ValidationError):
    """Model output could not be coerced into the requested schema."""


# --- engine -----------------------------------------------------------------------


class EngineError(KeelError):
    retryable: ClassVar[bool] = True


class ContextLengthExceeded(EngineError):
    """Prompt plus requested completion does not fit the model's window.

    Terminal by default: no amount of retrying shortens the prompt, and the
    engine should have rejected this at admission time.
    """

    retryable: ClassVar[bool] = False


class RequestCancelled(KeelError):
    retryable: ClassVar[bool] = False


# --- backend ----------------------------------------------------------------------


class BackendError(KeelError):
    """Failure originating outside our process, typically from a model server."""

    retryable: ClassVar[bool] = True


class RateLimited(BackendError):
    """Backend is shedding load. Back off and retry, ideally with jitter."""


class BackendTimeout(BackendError):
    pass


class BackendUnavailable(BackendError):
    """Connection refused or DNS failure. Distinct from a timeout: retrying a
    timeout may duplicate work that already succeeded."""


class CircuitOpen(BackendError):
    """Breaker is open. Short-circuit instead of paying the connection cost."""

    retryable: ClassVar[bool] = False


# --- policy -----------------------------------------------------------------------


class BudgetExceeded(KeelError):
    """Tenant is out of budget. Terminal for this request, and worth surfacing
    as a distinct 402 rather than a generic 429."""

    retryable: ClassVar[bool] = False


class QuotaExceeded(KeelError):
    """Rate limit rejection from the local limiter rather than the backend."""

    retryable: ClassVar[bool] = False


# --- orchestration ----------------------------------------------------------------


class DAGError(KeelError):
    pass


class CycleDetected(DAGError):
    pass


class MissingDependency(DAGError):
    pass


class EvalGateFailed(DAGError):
    """Regression gate rejected the change. This is the point of the gate."""
