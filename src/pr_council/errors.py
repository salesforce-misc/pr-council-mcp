"""Root exception hierarchy for pr-council-mcp.

``PrCouncilError`` is the single base class every error this project raises
to callers derives from, so a caller can catch all package-originated failures
with one ``except PrCouncilError``. Domain-specific errors subclass it.
(Private control-flow signals that never cross a call boundary -- e.g.
``review.git._OutputLimitExceeded``, which is always translated to a
``ReviewError`` before returning -- intentionally stay outside this hierarchy.)
"""


class PrCouncilError(Exception):
    """Base class for all pr-council-mcp errors."""
