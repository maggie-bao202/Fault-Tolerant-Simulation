"""Pauli-based-computation (PPM) front-end for ftsim -- **opt-in**.

A logical ``{H,S,T,CNOT}`` circuit is lowered by PennyLane/Catalyst's PPM passes
(``to_ppr -> commute_ppr -> merge_ppr_ppm``) to a Pauli-based-computation form: a
sequence of Pauli Product Rotations (``pi/4`` / ``pi/2`` Clifford, ``pi/8``
non-Clifford) plus terminal single-qubit ``Z`` measurements.  Only the
**Clifford part** is compiled to an encoded ``stim.Circuit`` (each Clifford PPR
re-synthesised from transversal ``{H, S, CNOT}``).  Every non-Clifford ``pi/8``
rotation hands off to :func:`ftsim.ppm.magic.magic_injection`, which currently
raises -- the seam where magic-state creation + injection will land.

Use it via ``run_pipeline(lc, p, frontend="ppm")`` or directly::

    from ftsim.ppm import compile_ppm
    comp = compile_ppm(bench, "h6")          # bench = lc + lc.inverse()

Needs the optional dependency group: ``pip install -e ".[ppm]"``.  Only
:mod:`ftsim.ppm.catalyst_frontend` imports PennyLane/Catalyst (lazily), so the
IR (:mod:`ftsim.ppm.ir`), the MLIR parser (:mod:`ftsim.ppm.mlir_parse`) and the
lowering seam (:func:`ftsim.ppm.encoded.lower_pbc`) are usable without it.
"""

from __future__ import annotations

from ._errors import PPMUnsupported
from .encoded import PPM_SUPPORTED_CODES, compile_ppm, lower_pbc
from .ir import PPM, PPR, PBCCircuit
from .magic import magic_injection

__all__ = [
    "compile_ppm",
    "lower_pbc",
    "PBCCircuit",
    "PPR",
    "PPM",
    "PPMUnsupported",
    "PPM_SUPPORTED_CODES",
    "magic_injection",
]
