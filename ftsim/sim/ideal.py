"""Dense state-vector reference for the intended logical unitary.

Used to check that the compiled ``[[6,2,2]]`` circuit really implements the
input ``{H, S, T, CNOT}`` circuit: we compare clifft ``EXP_VAL`` probes on the
noiseless compiled circuit against ``<psi|P|psi>`` computed here.

Little-endian convention: qubit 0 is the least-significant bit of the state
index, matching how ``ftsim.compile`` assigns logical qubit -> patch and how the
probes are built.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..logical_gates import LogicalCircuit

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_1Q = {
    "H": _H, "S": _S, "T": _T, "X": _X,
    "S_DAG": _S.conj().T, "T_DAG": _T.conj().T,
}

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": _X,
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _apply_1q(state: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    state = state.reshape([2] * n)
    state = np.tensordot(U, state, axes=([1], [n - 1 - q]))
    state = np.moveaxis(state, 0, n - 1 - q)
    return state.reshape(-1)


def _apply_cnot(state: np.ndarray, c: int, t: int, n: int) -> np.ndarray:
    idx = np.arange(2**n)
    ctrl = (idx >> c) & 1
    flipped = idx ^ (ctrl << t)
    return state[flipped]


def state_of(lc: LogicalCircuit, n: int | None = None, input_state: str | None = None) -> np.ndarray:
    """Return the 2**n complex state vector produced by ``lc`` on ``|input_state>``.

    ``input_state`` is a length-n string over ``{'0', '1', '+'}`` (default all '0').
    """
    n = lc.n_qubits if n is None else n
    if n > 14:
        raise ValueError(f"Dense reference only supports n <= 14 (got {n}).")
    input_state = "0" * n if input_state is None else input_state
    if len(input_state) != n:
        raise ValueError("input_state length must equal n.")

    st = np.zeros(2**n, dtype=complex)
    st[0] = 1.0
    for q, ch in enumerate(input_state):
        if ch == "1":
            st = _apply_1q(st, _X, q, n)
        elif ch == "+":
            st = _apply_1q(st, _H, q, n)
        elif ch != "0":
            raise ValueError(f"input_state char {ch!r} not in {{0,1,+}}.")

    for g in lc.gates:
        if g[0] == "CNOT":
            st = _apply_cnot(st, g[1], g[2], n)
        else:
            st = _apply_1q(st, _1Q[g[0]], g[1], n)
    return st


def expectation(state: np.ndarray, pauli: Dict[int, str], n: int) -> float:
    """<state| (tensor of single-qubit Paulis) |state>, real part."""
    psi = state
    out = state.copy()
    for q, p in pauli.items():
        if p == "I":
            continue
        out = _apply_1q(out, _PAULI[p], q, n)
    return float(np.real(np.vdot(psi, out)))


def expected_expvals(
    lc: LogicalCircuit,
    probes: Sequence[Dict[int, str]],
    n: int | None = None,
    input_state: str | None = None,
) -> list[float]:
    """Reference values for a list of probes (each ``{qubit: 'X'|'Y'|'Z'|'I'}``)."""
    n = lc.n_qubits if n is None else n
    st = state_of(lc, n, input_state)
    return [expectation(st, pr, n) for pr in probes]
