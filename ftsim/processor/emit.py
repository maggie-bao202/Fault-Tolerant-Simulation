"""Stim/clifft circuit-text builder with measurement-record tracking.

`Circ` accumulates instruction lines and a running count of measurements so
that `DETECTOR` / `OBSERVABLE_INCLUDE` can be emitted from *absolute*
measurement ids (0-based, in emission order).  The output text is fed straight
to `clifft.compile`.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..qec.h6 import se_round_lines
from ..qec.layout import Patch


class Circ:
    def __init__(self) -> None:
        self._lines: List[str] = []
        self.n_meas: int = 0
        self.n_obs: int = 0
        self.n_det: int = 0

    # -- raw -------------------------------------------------------------
    def raw(self, line: str) -> None:
        self._lines.append(line)

    def raw_block(self, lines: Sequence[str], n_meas_added: int) -> None:
        """Append pre-built lines that contain their own measurements."""
        self._lines.extend(lines)
        self.n_meas += n_meas_added

    def det_count(self) -> int:
        return self.n_det

    # backwards-compat alias used by compile.py
    def _det_count(self) -> int:
        return self.n_det

    def gate(self, name: str, qubits: Sequence[int]) -> None:
        if qubits:
            self._lines.append(f"{name} " + " ".join(map(str, qubits)))

    def phys_t(self, qubits: Sequence[int], dag: bool = False) -> None:
        self.gate("T_DAG" if dag else "T", qubits)

    # -- measurement --------------------------------------------------------
    def measure(self, qubits: Sequence[int], basis: str = "Z") -> List[int]:
        """Append M/MX/MR and return the absolute measurement ids produced."""
        op = {"Z": "M", "X": "MX", "ZR": "MR"}[basis]
        self.gate(op, qubits)
        ids = list(range(self.n_meas, self.n_meas + len(qubits)))
        self.n_meas += len(qubits)
        return ids

    def _rec(self, mid: int) -> str:
        return f"rec[{mid - self.n_meas}]"

    def detector(self, mids: Sequence[int], coords: Sequence[float] | None = None) -> int:
        tag = f"({','.join(map(str, coords))}) " if coords else ""
        self._lines.append(
            "DETECTOR " + tag + " ".join(self._rec(m) for m in mids)
        )
        self.n_det += 1
        return self.n_det - 1

    def observable(self, mids: Sequence[int], idx: int | None = None) -> int:
        k = self.n_obs if idx is None else idx
        self._lines.append(
            f"OBSERVABLE_INCLUDE({k}) " + " ".join(self._rec(m) for m in mids)
        )
        self.n_obs = max(self.n_obs, k + 1)
        return k

    def exp_val(self, pauli_terms: Sequence[str]) -> None:
        self._lines.append("EXP_VAL " + "*".join(pauli_terms))

    # -- [[6,2,2]] building blocks ----------------------------------------
    def se_round(self, patches: Sequence[Patch]) -> Dict[str, List[int]]:
        """One stabilizer-extraction round for `patches`.

        The per-patch circuit is delegated to
        :func:`ftsim.qec.h6.se_round_lines` -- the single source of truth for the
        SE round -- so this only fans it out over patches and records the
        absolute measurement ids produced (4 per patch, in order x0, x1, z0, z1).

        Returns {patch.name: [x0, x1, z0, z1]} absolute measurement ids.
        """
        out: Dict[str, List[int]] = {}
        for p in patches:
            base = self.n_meas
            self.raw_block(se_round_lines(p.data, p.xanc, p.zanc), 4)
            out[p.name] = list(range(base, base + 4))
        return out

    def text(self) -> str:
        return "\n".join(self._lines) + "\n"
