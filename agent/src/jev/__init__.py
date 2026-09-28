"""Jev (TypeSafe's "System One" model) wrapper.

Jev doesn't write text, it just answers typed questions (choice / score /
yes-no) about some state. We use it as a second opinion before opening
positions, see gate.py.
"""

from src.jev.client import (
    JevAnswer,
    JevClient,
    JevError,
    JevResult,
    boolean,
    choice,
    score,
)

__all__ = [
    "JevAnswer",
    "JevClient",
    "JevError",
    "JevResult",
    "boolean",
    "choice",
    "score",
]
