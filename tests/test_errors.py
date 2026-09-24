"""Pin the package-wide catch-all contract promised by ``errors.py``.

The :mod:`pr_council.errors` module docstring promises that
``PrCouncilError`` is "the single base class every other exception in this
project derives from, so a caller can catch all package-originated failures
with one ``except PrCouncilError``". These tests pin exactly that contract:
every package-defined exception must subclass ``PrCouncilError`` and be
catchable through it. Shared infrastructure errors are owned by ``localmcp``;
this contract covers only application-defined failures.
"""

import pytest

from pr_council.config import ConfigError
from pr_council.errors import PrCouncilError
from pr_council.review.models import ReviewError

_PACKAGE_ERRORS = [ConfigError, ReviewError]


def test_base_error_is_an_exception_subclass():
    assert issubclass(PrCouncilError, Exception)


@pytest.mark.parametrize("error_type", _PACKAGE_ERRORS)
def test_package_error_subclasses_the_catch_all_base(error_type):
    assert issubclass(error_type, PrCouncilError)


@pytest.mark.parametrize("error_type", _PACKAGE_ERRORS)
def test_package_error_is_caught_by_the_catch_all_base(error_type):
    caught: PrCouncilError | None = None
    try:
        raise error_type("boom")
    except PrCouncilError as exc:
        caught = exc
    assert isinstance(caught, error_type)
    assert str(caught) == "boom"
