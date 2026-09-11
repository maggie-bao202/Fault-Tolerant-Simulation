"""Shared error type for the PPM frontend.

``PPMUnsupported`` subclasses :class:`ftsim.backends.UnsupportedGate` so that
callers already guarding compilation with ``except UnsupportedGate`` (e.g.
``ftsim.pipeline.check_unitary``) keep working unchanged.
"""

from __future__ import annotations

from ..backends import UnsupportedGate


class PPMUnsupported(UnsupportedGate):
    """The PPM frontend cannot lower this circuit / code / IR shape (yet)."""


__all__ = ["PPMUnsupported"]
