"""Zero-level distillation of a ``[[6,2,2]]`` / Magic-H6 magic state.

:func:`magic_factory_lines` prepares an encoded distilled magic block on 6 data
qubits, scored by **post-selection** (``[[6,2,2]]`` is ``d = 2`` -- detection
only):

``MAGIC_INPUT="TT"`` (the verified recipe): encode ``|++>_L`` with the Magic-H6
encoder, then a **transversal physical ``T``** on the six data qubits.
Transversal physical ``T`` keeps the ``[[6,2,2]]`` codespace invariant and acts
as the logical ``T`` the teleportation gadget needs, so no post-teleport Clifford
fixup is required.  A physical-``T`` fault lands the block outside the codespace
and is heralded by the stabilizer detectors, so post-selection distils it -- this
is the fault-tolerant value of running ``T`` on a *separate*, checked block
rather than directly on the data.  The stabilizer post-selection that projects
the block back into the codespace *is* the (zero-level) distillation.

Emitted as **circuit-text lines** (``list[str]``), not ``stim.Circuit`` objects,
because a real ``T`` gate is not a Stim instruction -- the backend is ``clifft``
(Stim-compatible text + real ``T``).

Manifests use **relative** measurement ids (0-based within the returned block);
the caller adds its running offset.  ``manifest['postselect']`` is a list of
tuples of rel ids -- the XOR of each group must be 0 (a length-1 group is a
plain "bit must be 0").

Known limitation: bare ``[[6,2,2]]`` has no *exact* transversal ``T``, so a
``T`` conjugated by ``H`` on the same qubit carries ~7% residual coherent error
(surfaced by ``ftsim`` as ``FTReport.noiseless_error_floor``).  Clifford gates
and non-``H``-conjugated ``T`` are exact.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..qec.h6 import encoder_lines, se_round_lines

# How the encoded magic block is prepared before stabilizer post-selection:
#   "TT" -> encode |++>_L, then transversal physical T   (verified; recommended)
#   "A"  -> |A>^6 = T|+>^6 physical inputs, then the encoder
#   "H"  -> |H>^6 = RY(pi/4)|0>^6 physical inputs, then the encoder
MAGIC_INPUT: str = "TT"

Group = Tuple[int, ...]
Manifest = Dict[str, object]


def _g(name: str, qubits: Sequence[int]) -> str:
    return f"{name} " + " ".join(map(str, qubits))


def magic_factory_lines(
    data: Sequence[int],
    xanc: Sequence[int],
    zanc: Sequence[int],
    bare: Sequence[int],
    *,
    se_rounds: int = 1,
    dag: bool = False,
) -> Tuple[List[str], Manifest]:
    """Prepare an encoded distilled magic block on the 6 ``data`` globals.

    ``dag=True`` distils a ``T_DAG``-type magic state instead of ``T``.
    """
    if len(data) != 6:
        raise ValueError("magic_factory_lines needs 6 data-qubit indices.")
    tstr = "T_DAG" if dag else "T"
    dq = " ".join(map(str, data))
    lines: List[str] = [_g("R", data)]
    if MAGIC_INPUT == "TT": #transversal physical T
        lines += encoder_lines(data)               # |++>_L  (in codespace)
        lines.append(f"{tstr} {dq}")               # transversal physical T
    elif MAGIC_INPUT == "H":
        lines.append(f"R_Y({-0.25 if dag else 0.25}) {dq}")   # |H>^6
        lines += encoder_lines(data)
    else:  # "A"
        lines.append(f"H {dq}")
        lines.append(f"{tstr} {dq}")               # |A>^6 = T|+>^6
        lines += encoder_lines(data)

    m = 0
    postselect: List[Group] = []
    # Stabilizer post-selection projects the block into the codespace (this is
    # the distillation); these ancilla reads are random on the noiseless
    # circuit, so the caller post-selects them on raw 0.
    for _ in range(max(1, se_rounds)):
        lines += se_round_lines(data, xanc, zanc)
        postselect += [(m + i,) for i in range(4)]
        m += 4

    return lines, {"meas": m, "postselect": postselect, "raw": True}


__all__ = ["MAGIC_INPUT", "magic_factory_lines"]
