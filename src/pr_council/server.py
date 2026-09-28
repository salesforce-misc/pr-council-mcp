"""Thin executable declaration for the shared localmcp stdio server."""

from __future__ import annotations

import localmcp

import pr_council.tools as tools
from pr_council.config import SERVER_NAME, parse_config
from pr_council.server_setup import PrCouncilServer
from pr_council.workflows.review.runtime import ReviewRuntime

server = PrCouncilServer(
    name=SERVER_NAME,
    tools=tools.TOOLS,
    config_parser=parse_config,
    runtime_factory=ReviewRuntime,
)


def main() -> None:
    localmcp.main(server)


if __name__ == "__main__":
    main()
