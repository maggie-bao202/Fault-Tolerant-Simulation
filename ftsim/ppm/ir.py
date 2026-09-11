"""The Pauli-based-computation (PBC) intermediate representation.

A ``LogicalCircuit`` is lowered by Catalyst's PPM passes to a list of

* **Pauli Product Rotations** -- :class:`PPR`, ``exp(-i * angle * sign * P)`` with
  ``angle in {pi/2, pi/4, pi/8}``.  ``pi/2`` and ``pi/4`` are Clifford; ``pi/8``
  (a ``T``) is the only non-Clifford rotation and is left for magic-state
  injection.
* **terminal Pauli Product Measurements** -- :class:`PPM`, a destructive
  measurement of a Pauli product ``P``.

:class:`PBCCircuit` bundles the two lists (program order) and
:meth:`PBCCircuit.partition` splits the rotations into the Clifford part (which
:mod:`ftsim.ppm.encoded` lowers to stim) and the non-Clifford ``pi/8`` part
(which currently raises in :func:`ftsim.ppm.magic.magic_injection`).

This module is pure stdlib -- it does *not* import PennyLane/Catalyst, so the
lowering seam (:func:`ftsim.ppm.encoded.lower_pbc`) is unit-testable without the
optional ``ftsim[ppm]`` dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

_PAULI = frozenset({"X", "Y", "Z"})
_ANGLES = ("pi/2", "pi/4", "pi/8")


def _normalise(paulis, qubits) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
    """Drop identity factors, upper-case, and sort ``(pauli, qubit)`` pairs by
    ascending qubit index."""
    paulis = tuple(paulis)
    qubits = tuple(qubits)
    if len(paulis) != len(qubits):
        raise ValueError(
            f"paulis and qubits must have equal length; got {len(paulis)} "
            f"vs {len(qubits)}"
        )
    pairs = []
    for p, q in zip(paulis, qubits):
        p = str(p).upper()
        if p in ("I", ""):
            continue
        if p not in _PAULI:
            raise ValueError(f"Pauli letter must be one of X, Y, Z, I; got {p!r}")
        pairs.append((int(q), p))
    if not pairs:
        raise ValueError("a Pauli product needs at least one non-identity factor")
    qs = [q for q, _ in pairs]
    if len(set(qs)) != len(qs):
        raise ValueError(f"a Pauli product touches each qubit at most once; got {qs}")
    pairs.sort()
    return tuple(p for _, p in pairs), tuple(q for q, _ in pairs)


@dataclass(frozen=True)
class PPR:
    """A Pauli Product Rotation ``exp(-i * angle * sign * P)``.

    ``paulis[i]`` acts on logical qubit ``qubits[i]``; ``qubits`` is ascending
    and every ``paulis`` letter is in ``{X, Y, Z}`` (identity factors dropped on
    construction).
    """

    paulis: Tuple[str, ...]
    qubits: Tuple[int, ...]
    angle: str                 # "pi/2" | "pi/4" | "pi/8"
    sign: int = 1              # +1 | -1

    def __init__(self, paulis, qubits, angle: str, sign: int = 1):
        p, q = _normalise(paulis, qubits)
        if angle not in _ANGLES:
            raise ValueError(f"angle must be one of {_ANGLES}; got {angle!r}")
        if sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1; got {sign!r}")
        object.__setattr__(self, "paulis", p)
        object.__setattr__(self, "qubits", q)
        object.__setattr__(self, "angle", angle)
        object.__setattr__(self, "sign", int(sign))

    @property
    def is_clifford(self) -> bool:
        return self.angle in ("pi/2", "pi/4")

    @property
    def weight(self) -> int:
        return len(self.paulis)

    def __repr__(self) -> str:
        body = "".join(f"{p}{q}" for p, q in zip(self.paulis, self.qubits))
        s = "-" if self.sign < 0 else ""
        return f"PPR({s}{body} @ {self.angle})"


@dataclass(frozen=True)
class PPM:
    """A destructive terminal Pauli Product Measurement of ``sign * P``.

    ``sign`` records the measurement's Pauli sign; it does not affect ftsim
    scoring (the noiseless majority vote in :mod:`ftsim.score` absorbs it).
    """

    paulis: Tuple[str, ...]
    qubits: Tuple[int, ...]
    sign: int = 1

    def __init__(self, paulis, qubits, sign: int = 1):
        p, q = _normalise(paulis, qubits)
        if sign not in (1, -1):
            raise ValueError(f"sign must be +1 or -1; got {sign!r}")
        object.__setattr__(self, "paulis", p)
        object.__setattr__(self, "qubits", q)
        object.__setattr__(self, "sign", int(sign))

    @property
    def weight(self) -> int:
        return len(self.paulis)

    def __repr__(self) -> str:
        body = "".join(f"{p}{q}" for p, q in zip(self.paulis, self.qubits))
        s = "-" if self.sign < 0 else ""
        return f"PPM({s}{body})"


@dataclass
class PBCCircuit:
    """A Pauli-based-computation circuit: PPRs in program order followed by
    terminal PPMs.  ``n_qubits`` is the logical qubit count."""

    n_qubits: int
    rotations: List[PPR] = field(default_factory=list)
    measurements: List[PPM] = field(default_factory=list)

    def partition(self) -> Tuple[List[PPR], List[PPR]]:
        """``(clifford_pprs, non_clifford_pi8_pprs)`` -- program order preserved
        within each list."""
        cliff = [r for r in self.rotations if r.is_clifford]
        pi8 = [r for r in self.rotations if not r.is_clifford]
        return cliff, pi8

    @property
    def is_clifford(self) -> bool:
        return all(r.is_clifford for r in self.rotations)

    @property
    def t_count(self) -> int:
        return sum(1 for r in self.rotations if not r.is_clifford)


__all__ = ["PPR", "PPM", "PBCCircuit"]
