"""Pure helpers for reading and interpreting unified PR diffs."""

from __future__ import annotations


def commentable_lines(diff: str) -> dict[str, set[int]]:
    """Return added/right-side lines accepted by GitHub inline comments."""
    result: dict[str, set[int]] = {}
    current: str | None = None
    new_line = 0
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[6:]
            result.setdefault(current, set())
        elif line.startswith("@@"):
            try:
                token = next(part for part in line.split() if part.startswith("+"))
                new_line = int(token[1:].split(",", 1)[0])
            except (StopIteration, ValueError):
                current = None
        elif current is not None and line.startswith("+") and not line.startswith("+++"):
            result[current].add(new_line)
            new_line += 1
        elif current is not None and not line.startswith("-"):
            new_line += 1
    return result
