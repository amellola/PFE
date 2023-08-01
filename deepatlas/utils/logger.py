"""Logging setup for agatha package."""

import logging
import sys


class MaxLevelFilter(logging.Filter):
    """Filter class that ignores entries at and above a specified maximum logging level."""

    def __init__(self, min_ignore_level: int):
        """Constructor.

        Args:
            min_ignore_level: Min level of records to filter.
        """
        super().__init__()
        self.min_ignore_level = min_ignore_level

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out message with level at or above `self.min_ignore_level`.

        Args:
            record (logging.LogRecord): The record to filter.

        Returns:
            bool: True if the record should be logged.
        """
        return record.levelno < self.min_ignore_level


def setup_logger(debug: bool = False) -> None:
    """Set up the logging structure of the entire package."""
    min_level_stdout = logging.INFO
    if debug:
        min_level_stdout = logging.DEBUG
    min_level_stderr = logging.WARNING

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(min_level_stdout)
    stdout_handler.addFilter(MaxLevelFilter(min_level_stderr))

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(min_level_stderr)

    logging.basicConfig(
        handlers=[stdout_handler, stderr_handler],
        level=min_level_stdout,
        format="%(asctime)s %(levelname)s %(module)s %(message)s",
        force=True,
    )
