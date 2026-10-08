from __future__ import annotations

import pytest

from keel import errors


@pytest.mark.parametrize(
    ("exc", "retryable"),
    [
        (errors.BackendTimeout, True),
        (errors.RateLimited, True),
        (errors.BackendUnavailable, True),
        (errors.BackendError, True),
        (errors.EngineError, True),
        (errors.CircuitOpen, False),
        (errors.ContextLengthExceeded, False),
        (errors.BudgetExceeded, False),
        (errors.QuotaExceeded, False),
        (errors.ValidationError, False),
        (errors.RequestCancelled, False),
    ],
)
def test_retryability_is_declared_not_inferred(
    exc: type[errors.KeelError], retryable: bool
) -> None:
    assert exc("boom").retryable is retryable


def test_subclasses_inherit_parent_policy() -> None:
    assert issubclass(errors.RateLimited, errors.BackendError)
    assert errors.RateLimited("x").retryable is errors.BackendError.retryable


def test_every_error_descends_from_the_package_base() -> None:
    for name in dir(errors):
        obj = getattr(errors, name)
        if isinstance(obj, type) and issubclass(obj, BaseException):
            if obj.__module__ != errors.__name__:
                continue
            assert issubclass(obj, errors.KeelError), name
