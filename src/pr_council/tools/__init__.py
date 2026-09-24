"""Explicit aggregate of every MCP tool module's ``TOOLS`` list."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pr_council.tools.review as review

TOOLS: list[Callable[..., Any]] = [*review.TOOLS]
