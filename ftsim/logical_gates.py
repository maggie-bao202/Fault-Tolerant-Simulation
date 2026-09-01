"""The input type: a logical circuit over the universal set {H, S, T, CNOT}.

A ``LogicalCircuit`` is just an ordered list of gate tuples:

    ("H", q)          logical Hadamard on qubit q
    ("S", q)          logical phase gate on qubit q
    ("T", q)          logical pi/8 gate on qubit q  (the only non-Clifford)
    ("CNOT", c, t)    logical CNOT, control c, target t
    ("X", q)          logical Pauli X (handy for state prep / tests)

Qubit indices are ``0..n-1`` with ``n`` inferred from the highest index used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

Gate = Tuple  # ("H", q) | ("S", q) | ("T", q) | ("X", q) | ("CNOT", c, t) | ("S_DAG", q) | ("T_DAG", q)

_1Q = {"H", "S", "S_DAG", "T", "T_DAG", "X"}
_2Q = {"CNOT"}
_CLIFFORD = {"H", "S", "S_DAG", "X", "CNOT"}
_INV_1Q = {"H": "H", "X": "X", "S": "S_DAG", "S_DAG": "S", "T": "T_DAG", "T_DAG": "T"}


@dataclass(frozen=True)
class LogicalCircuit:
    """An ordered list of ``{H, S, T, X, CNOT}`` gates on ``n_qubits`` logical qubits."""

    gates: Tuple[Gate, ...]

    def __init__(self, gates: Iterable[Gate]): #list of tuples
        norm: List[Gate] = []
        for g in gates:
            g = tuple(g)
            name = g[0].upper()
            #check for correct format
            if name in _1Q:
                if len(g) != 2 or not isinstance(g[1], int) or g[1] < 0:
                    raise ValueError(f"Bad 1-qubit gate {g!r}; expected (name, qubit>=0).")
                norm.append((name, g[1]))
            elif name in _2Q:
                if len(g) != 3 or g[1] == g[2] or min(g[1], g[2]) < 0:
                    raise ValueError(
                        f"Bad CNOT {g!r}; expected ('CNOT', control, target) distinct >= 0."
                    )
                norm.append((name, int(g[1]), int(g[2])))
            else:
                raise ValueError(f"Unsupported gate {name!r}. Use one of {_1Q | _2Q}.")
        object.__setattr__(self, "gates", tuple(norm))

    # -- derived properties -------------------------------------------------

    @property
    def n_qubits(self) -> int:
        m = -1
        for g in self.gates:
            m = max(m, *g[1:])
        return m + 1

    @property
    def t_count(self) -> int:
        return sum(1 for g in self.gates if g[0] in ("T", "T_DAG"))

    def inverse(self) -> "LogicalCircuit":
        """Dagger: reverse order, invert each gate (CNOT is self-inverse)."""
        inv: List[Gate] = []
        for g in reversed(self.gates):
            if g[0] == "CNOT":
                inv.append(g)
            else:
                inv.append((_INV_1Q[g[0]], g[1]))
        return LogicalCircuit(inv)

    @property
    def is_clifford(self) -> bool:
        return all(g[0] in _CLIFFORD for g in self.gates)

    def qubits_of(self, g: Gate) -> Tuple[int, ...]:
        return tuple(g[1:])

    def layers(self) -> List[List[Gate]]:
        """Greedy depth layers: each layer is a set of gates on disjoint qubits."""
        layers: List[List[Gate]] = []
        frontier: dict[int, int] = {}  # qubit -> index of last layer touching it
        for g in self.gates:
            qs = self.qubits_of(g)
            start = max((frontier.get(q, -1) for q in qs), default=-1) + 1
            if start == len(layers):
                layers.append([])
            layers[start].append(g)
            for q in qs:
                frontier[q] = start
        return layers

    def __iter__(self):
        return iter(self.gates)

    def __len__(self):
        return len(self.gates)

    def __repr__(self):
        body = ", ".join(
            f"{g[0]}{g[1:]}" if len(g) == 2 else f"{g[0]}{tuple(g[1:])}" for g in self.gates
        )
        return f"LogicalCircuit(n={self.n_qubits}, t={self.t_count}, [{body}])"


def parse(seq: Sequence) -> LogicalCircuit:
    """Convenience: ``parse([("H", 0), ("CNOT", 0, 1)])``."""
    return LogicalCircuit(seq)
