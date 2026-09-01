"""Self-contained ``[[6, 2, 2]]`` code data + circuit fragments.

Everything ``ftsim`` needs about the ``[[6, 2, 2]]`` / Magic-H6 base code, with
no external dependencies.  Ported from the CQCL Magic-H6 ``Code614.py`` (the
verbatim ``get_dist_circ`` encoder and the bipartite-coloring extraction round).

Code (labels 0..5 for the six data qubits):

* stabilizers : ``XXXX`` and ``ZZZZ`` on ``{0,1,2,3}`` and on ``{2,3,4,5}``
  (self-dual: X- and Z-check supports coincide).
* logical slot 0 : ``X_L = X0 X2 X4``, ``Z_L = Z0 Z2 Z4``  (support ``{0,2,4}``).
* logical slot 1 : support ``{1,3,5}``  (spectator; ``ftsim`` uses only slot 0).
* distance 2  -> error *detection* only; scoring is by post-selection.

Per-patch physical layout used by :mod:`ftsim.layout` (12 qubits):
``0..5`` data, ``6..7`` X-ancillas, ``8..9`` Z-ancillas, ``10..11`` bare
(H-check) ancillas.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

PATCH_QUBITS = 12
LOCAL_DATA: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
LOCAL_XANC: Tuple[int, ...] = (6, 7)
LOCAL_ZANC: Tuple[int, ...] = (8, 9)
LOCAL_BARE: Tuple[int, ...] = (10, 11)

STAB_SUPPORT: Tuple[Tuple[int, ...], ...] = ((0, 1, 2, 3), (2, 3, 4, 5))
SLOT0: Tuple[int, ...] = (0, 2, 4)
SLOT1: Tuple[int, ...] = (1, 3, 5)

# Magic-H6 encoder |0>^6 -> |++>_L  (verbatim gate order from Code614.py;
# each label i is mapped to the i-th supplied data-qubit index).
_ENCODER: Tuple[Tuple[str, Tuple[int, ...]], ...] = (
    ("H", (0, 1)),
    ("H", (2, 4)),
    ("CX", (2, 3)),
    ("CX", (4, 5)),
    ("CX", (2, 0)),
    ("CX", (3, 1)),
    ("CX", (0, 4)),
    ("CX", (1, 5)),
    ("CX", (4, 2)),
    ("CX", (5, 3)),
)


def _g(name: str, qubits: Sequence[int]) -> str:
    return f"{name} " + " ".join(map(str, qubits))


def encoder_lines(data: Sequence[int]) -> List[str]:
    """Circuit-text lines for the encoder acting on the 6 global ``data`` indices."""
    if len(data) != 6:
        raise ValueError("encoder_lines needs 6 data-qubit indices.")
    return [f"{gate} " + " ".join(str(data[i]) for i in labels)
            for gate, labels in _ENCODER]


def se_round_lines(
    data: Sequence[int], xanc: Sequence[int], zanc: Sequence[int]
) -> List[str]:
    """One stabilizer-extraction round: emits exactly 4 measurements (x0 x1 z0 z1)."""
    lines = [_g("R", [*xanc, *zanc]), _g("H", xanc)]
    for i, supp in enumerate(STAB_SUPPORT):          # X checks: ancilla controls data
        lines.append(_g("CX", [q for j in supp for q in (xanc[i], data[j])]))
    for i, supp in enumerate(STAB_SUPPORT):          # Z checks: data controls ancilla
        lines.append(_g("CX", [q for j in supp for q in (data[j], zanc[i])]))
    lines.append(_g("H", xanc))
    lines.append(_g("M", [*xanc, *zanc]))
    return lines


def hcheck_lines(data: Sequence[int], bare: Sequence[int]) -> List[str]:
    """Bell-pair H-check of X0_L and X1_L (deterministic only on |++>_L; 2 measurements)."""
    a0, a1 = bare
    lines = [_g("R", [a0, a1]), _g("H", [a0]), _g("CX", [a0, a1])]
    for j0, j1 in zip(SLOT0, SLOT1):
        lines.append(_g("CX", [a0, data[j0]]))
        lines.append(_g("CX", [a1, data[j1]]))
    lines += [_g("CX", [a0, a1]), _g("H", [a0]), _g("M", [a0, a1])]
    return lines
