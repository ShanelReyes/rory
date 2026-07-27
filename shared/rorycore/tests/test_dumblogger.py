import pytest
from rory.core.logger.Dumblogger import DumbLogger


@pytest.fixture
def dumb_logger():
    return DumbLogger()


def test_debug_returns_none(dumb_logger):
    result = dumb_logger.debug("test debug")
    assert result is None


def test_info_returns_none(dumb_logger):
    result = dumb_logger.info("test info")
    assert result is None


def test_error_returns_none(dumb_logger):
    result = dumb_logger.error("test error")
    assert result is None


def test_all_methods_no_exception(dumb_logger):
    dumb_logger.debug(123)
    dumb_logger.info(None)
    dumb_logger.error({"key": "value"})
