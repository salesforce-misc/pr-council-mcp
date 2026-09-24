"""Shared pytest fixtures for the pr-council-mcp test suite.

``configure_logging`` (invoked directly and via ``server.main``) mutates
process-global logging state:

- the root logger's handlers and level;
- the ``"fastmcp"`` logger's handlers, ``propagate`` flag, and level (it strips
  the stderr-bound ``RichHandler``\\ s fastmcp attaches at import, flips
  ``propagate`` to ``True``, and may reset the level fastmcp pins to ``INFO``);
- ``logging.lastResort`` and ``logging.raiseExceptions``;
- the process-global warnings-capture hook (``logging.captureWarnings(True)``).

This single autouse fixture snapshots and restores all of it so a per-test
``FileHandler`` pointed at a ``tmp_path``, a toggled global flag, or a
neutralized ``"fastmcp"`` logger never leaks into later tests.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _restore_logging_state() -> Iterator[None]:
    """Snapshot and restore global logging state around each test."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    # configure_logging() permanently mutates the "fastmcp" logger (strips its
    # handlers, flips propagate, and may reset its level); snapshot it too.
    fastmcp_logger = logging.getLogger("fastmcp")
    saved_fastmcp_handlers = fastmcp_logger.handlers[:]
    saved_fastmcp_propagate = fastmcp_logger.propagate
    saved_fastmcp_level = fastmcp_logger.level
    saved_last_resort = logging.lastResort
    saved_raise_exceptions = logging.raiseExceptions
    # The zero-handler "No handlers could be found for logger X" warning fires at
    # most once per process, gated by this manager flag. Tests that exercise the
    # pre-configure window reset it to False to make that path observable, which
    # would otherwise leak into later tests; snapshot it alongside the other
    # process-global logging flags.
    saved_emitted_no_handler_warning = root.manager.emittedNoHandlerWarning
    # configure_logging() calls logging.captureWarnings(True), which mutates the
    # process-global warnings hook; snapshot it so it does not leak across tests.
    saved_warnings_showwarning = logging._warnings_showwarning  # type: ignore[attr-defined]
    saved_showwarning = warnings.showwarning
    try:
        yield
    finally:
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)
        for handler in fastmcp_logger.handlers[:]:
            fastmcp_logger.removeHandler(handler)
        for handler in saved_fastmcp_handlers:
            fastmcp_logger.addHandler(handler)
        fastmcp_logger.propagate = saved_fastmcp_propagate
        fastmcp_logger.setLevel(saved_fastmcp_level)
        logging.lastResort = saved_last_resort
        logging.raiseExceptions = saved_raise_exceptions
        root.manager.emittedNoHandlerWarning = saved_emitted_no_handler_warning
        logging._warnings_showwarning = saved_warnings_showwarning  # type: ignore[attr-defined]
        warnings.showwarning = saved_showwarning
