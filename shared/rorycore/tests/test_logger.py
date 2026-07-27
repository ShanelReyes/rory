import pytest
import os
import tempfile
import logging
from rory.core.logger.Logger import create_logger


@pytest.fixture
def temp_log_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_create_logger_default_name():
    logger = create_logger(
        LOG_PATH="/tmp",
        LOG_FILENAME="default_test",
        add_error_log=False,
    )
    assert logger is not None
    assert isinstance(logger, logging.Logger)
    assert logger.name == "default"


def test_create_logger_custom_name():
    logger = create_logger(
        name="test_custom",
        LOG_PATH="/tmp",
        LOG_FILENAME="custom_test",
        add_error_log=False,
    )
    assert logger.name == "test_custom"


def test_create_logger_writes_to_file(temp_log_dir):
    log_filename = "test_log"
    logger = create_logger(
        name="test_file",
        LOG_PATH=temp_log_dir,
        LOG_FILENAME=log_filename,
        add_error_log=False,
        console_handler_filter=lambda record: True,
        file_handler_filter=lambda record: True,
    )
    logger.setLevel(logging.DEBUG)

    logger.info("test message")

    log_file = os.path.join(temp_log_dir, f"{log_filename}.csv")
    assert os.path.exists(log_file)

    with open(log_file, "r") as f:
        content = f.read()

    assert "test message" in content


def test_create_logger_error_log(temp_log_dir):
    log_filename = "test_error"
    logger = create_logger(
        name="test_error_log",
        LOG_PATH=temp_log_dir,
        LOG_FILENAME=log_filename,
        add_error_log=True,
        console_handler_filter=lambda record: True,
        file_handler_filter=lambda record: True,
    )
    logger.setLevel(logging.DEBUG)

    logger.error("error occurred")

    error_file = os.path.join(temp_log_dir, f"{log_filename}-error.log")
    assert os.path.exists(error_file)


def test_create_logger_without_error_log(temp_log_dir):
    log_filename = "test_no_error"
    create_logger(
        name="test_no_err",
        LOG_PATH=temp_log_dir,
        LOG_FILENAME=log_filename,
        add_error_log=False,
    )

    error_file = os.path.join(temp_log_dir, f"{log_filename}-error.log")
    assert not os.path.exists(error_file)


def test_create_logger_debug_level(temp_log_dir):
    log_filename = "test_debug"
    logger = create_logger(
        name="test_debug_logger",
        LOG_PATH=temp_log_dir,
        LOG_FILENAME=log_filename,
        add_error_log=False,
        console_handler_filter=lambda record: True,
        file_handler_filter=lambda record: True,
    )
    logger.setLevel(logging.DEBUG)

    logger.debug("debug message")
    logger.info("info message")

    log_file = os.path.join(temp_log_dir, f"{log_filename}.csv")
    assert os.path.exists(log_file)
