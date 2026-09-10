"""Logical ``T``-state teleportation for the ``[[6,2,2]]`` / Magic-H6 code.

:func:`t_teleport_lines` consumes a distilled magic block (prepared by
:func:`ftsim.resources.zero_level_distillation.magic_factory_lines`): transversal
``CX`` (data control, magic target), destructive ``MZ`` of the magic block, then
the fixed transversal :data:`MAGIC_FIXUP`.  The ``+1`` branch (no non-Clifford
``S`` byproduct) is post-selected, so no Pauli byproduct is left to track.

Emitted as **circuit-text lines** (``list[str]``) for the ``clifft`` backend.
Manifests use **relative** measurement ids; the caller adds its running offset.
``manifest['postselect']`` is a list of tuples of rel ids -- the XOR of each
group must be 0.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..qec.h6 import SLOT0, STAB_SUPPORT
from ..resources.zero_level_distillation import magic_factory_lines

# Transversal-Clifford fixup on the data patch after the teleportation
# measurement.  Empty for MAGIC_INPUT="TT" (gadget is already a clean logical T).
MAGIC_FIXUP: Tuple[str, ...] = ()

Group = Tuple[int, ...]
Manifest = Dict[str, object]


def _g(name: str, qubits: Sequence[int]) -> str:
    return f"{name} " + " ".join(map(str, qubits))


def _invert_fixup(fixup: Sequence[str]) -> Tuple[str, ...]:
    inv = {"H": "H", "X": "X", "S": "S_DAG", "S_DAG": "S"}
    return tuple(inv[g] for g in reversed(list(fixup)))


def t_teleport_lines(
    data: Sequence[int],
    magic_data: Sequence[int],
    magic_xanc: Sequence[int],
    magic_zanc: Sequence[int],
    magic_bare: Sequence[int],
    *,
    se_rounds: int = 1,
    dag: bool = False,
    fixup: Sequence[str] | None = None,
) -> Tuple[List[str], Manifest]:
    """One logical ``T`` (or ``T_DAG`` if ``dag``) on ``data``, consuming a magic block."""
    if len(data) != 6 or len(magic_data) != 6:
        raise ValueError("t_teleport_lines needs 6 + 6 data-qubit indices.")
    if fixup is None:
        fixup = _invert_fixup(MAGIC_FIXUP) if dag else MAGIC_FIXUP

    lines, fac = magic_factory_lines(
        magic_data, magic_xanc, magic_zanc, magic_bare, se_rounds=se_rounds, dag=dag
    )
    m = int(fac["meas"])
    postselect: List[Group] = list(fac["postselect"])  # type: ignore[arg-type]

    # transversal CX: data control -> magic target
    lines.append(_g("CX", [q for c, t in zip(data, magic_data) for q in (c, t)]))

    # destructive Z readout of the magic block (6 recs)
    lines.append(_g("M", magic_data))
    mr = {j: m + j for j in range(6)}
    m += 6

    for supp in STAB_SUPPORT:                       # magic Z-stabilizer consistency
        postselect.append(tuple(mr[j] for j in supp))
    # S-branch bit: magic slot-0 logical-Z parity == 0  -> no non-Clifford byproduct
    postselect.append(tuple(mr[j] for j in SLOT0))

    for g in fixup:                                 # fixed transversal-Clifford fixup
        lines.append(_g(g, data))

    return lines, {"meas": m, "postselect": postselect}


__all__ = ["MAGIC_FIXUP", "t_teleport_lines"]
