"""Render branch-aware coverage.xml totals as a GitHub Pages SVG badge."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path


def coverage_percent(report: Path) -> int:
    root = ET.parse(report).getroot()
    if root.tag != "coverage":
        raise ValueError("expected a coverage.xml report")

    counts = (
        "lines-covered",
        "lines-valid",
        "branches-covered",
        "branches-valid",
    )
    try:
        lines_covered, lines_valid, branches_covered, branches_valid = (int(root.attrib[name]) for name in counts)
    except (KeyError, ValueError) as exc:
        raise ValueError("coverage.xml is missing valid line or branch counts") from exc

    if not (0 <= lines_covered <= lines_valid and 0 <= branches_covered <= branches_valid):
        raise ValueError("coverage.xml has inconsistent line or branch counts")

    covered = lines_covered + branches_covered
    valid = lines_valid + branches_valid
    if valid == 0:
        raise ValueError("coverage.xml has no measurable lines or branches")

    # Coverage.py's total combines line and branch opportunities. Round to the
    # nearest whole percent for a compact badge, with halves rounded upward.
    return (200 * covered + valid) // (2 * valid)


def render_svg(percent: int) -> str:
    if percent < 60:
        color = "#e05d44"
    elif percent < 80:
        color = "#dfb317"
    else:
        color = "#4c1"

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="112" height="20"
  role="img" aria-label="coverage: {percent}%">
  <clipPath id="round"><rect width="112" height="20" rx="3"/></clipPath>
  <g clip-path="url(#round)">
    <rect width="64" height="20" fill="#555"/>
    <rect x="64" width="48" height="20" fill="{color}"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="32" y="14">coverage</text>
    <text x="88" y="14">{percent}%</text>
  </g>
</svg>
"""


def render_index(svg_name: str, percent: int) -> str:
    return f"""<!doctype html>
<html lang="en">
  <meta charset="utf-8">
  <title>Coverage badge</title>
  <p><img src="{escape(svg_name, quote=True)}" alt="coverage: {percent}%"></p>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="coverage.py XML report")
    parser.add_argument("output", type=Path, help="SVG output path")
    args = parser.parse_args()

    percent = coverage_percent(args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_svg(percent), encoding="utf-8")
    (args.output.parent / "index.html").write_text(render_index(args.output.name, percent), encoding="utf-8")


if __name__ == "__main__":
    main()
