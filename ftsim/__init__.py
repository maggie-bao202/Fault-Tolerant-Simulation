"""ftsim -- end-to-end fault-tolerant logical-circuit simulation on LightStim.

A logical ``{H, S, T, CNOT}`` circuit is compiled to one physical circuit with
``lightstim.ir.CircuitBuilder`` (detectors auto-generated), run under
circuit-level depolarising noise, and scored by post-selection.

    run_pipeline(lc, p, code="h6", ...)               # process a logical circuit
    run_pipeline(lc, p, protocol="memory", ...)       # d-round memory experiment
    run_pipeline(None, p, protocol="factory",         # benchmark a magic factory
                 zero_lvl_distill=True)

Axes:
* **code**     -- ``"h6" | "rotated_surface" | "repetition" | "color" | ...``
* **protocol** -- ``"processor" | "memory" | "factory"``
* **magic-state factory** (for T / factory) -- one of the flags
  ``zero_lvl_distill=`` (recipe), ``cultivation`` (bool), ``distillation=`` (level int)

``T`` is the Magic-H6 Clifford proxy (stim-only, no non-Clifford dependency).
"""

from __future__ import annotations

from .logical_gates import LogicalCircuit, parse
from .pipeline import FTReport, check_unitary, run_pipeline, list_protocols
from .backends import list_codes

__all__ = [
    "LogicalCircuit",
    "parse",
    "FTReport",
    "run_pipeline",
    "check_unitary",
    "list_codes",
    "list_protocols",
]
