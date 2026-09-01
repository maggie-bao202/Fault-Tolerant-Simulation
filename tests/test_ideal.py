import numpy as np
import pytest

from ftsim.logical_gates import LogicalCircuit
from ftsim.sim.ideal import expectation, state_of


def _e(gl, n, pauli, ins=None):
    return round(expectation(state_of(LogicalCircuit(gl), n, ins), pauli, n), 6)


def test_single_qubit_identities():
    assert _e([("H", 0), ("H", 0)], 1, {0: "Z"}) == 1.0          # HH = I
    assert _e([("S", 0), ("S", 0)], 1, {0: "X"}, "+") == -1.0     # SS = Z: Z|+> = |->
    assert _e([("T", 0), ("T", 0), ("S_DAG", 0)], 1, {0: "X"}, "+") == 1.0  # TT S_DAG = I on |+>
    assert _e([("X", 0), ("X", 0)], 1, {0: "Z"}) == 1.0


def test_t_on_plus():
    st = state_of(LogicalCircuit([("H", 0), ("T", 0)]), 1)
    assert abs(expectation(st, {0: "X"}, 1) - 2 ** -0.5) < 1e-9
    assert abs(expectation(st, {0: "Y"}, 1) - 2 ** -0.5) < 1e-9
    assert abs(expectation(st, {0: "Z"}, 1)) < 1e-9


def test_bell_and_ghz():
    st = state_of(LogicalCircuit([("H", 0), ("CNOT", 0, 1)]), 2)
    assert abs(expectation(st, {0: "X", 1: "X"}, 2) - 1) < 1e-9
    assert abs(expectation(st, {0: "Z", 1: "Z"}, 2) - 1) < 1e-9
    st = state_of(LogicalCircuit([("H", 0), ("CNOT", 0, 1), ("CNOT", 1, 2)]), 3)
    assert abs(expectation(st, {0: "X", 1: "X", 2: "X"}, 3) - 1) < 1e-9


def test_inverse_round_trips_to_identity():
    lc = LogicalCircuit([("H", 0), ("T", 0), ("CNOT", 0, 1), ("S", 1), ("T", 1)])
    rt = LogicalCircuit(list(lc.gates) + list(lc.inverse().gates))
    for ins in ("00", "11", "++"):
        st = state_of(rt, 2, ins)
        st0 = state_of(LogicalCircuit([]), 2, ins)
        assert abs(abs(np.vdot(st0, st)) - 1) < 1e-9


def test_layers_are_disjoint():
    lc = LogicalCircuit([("H", 0), ("H", 1), ("CNOT", 0, 1), ("T", 0), ("T", 1)])
    for layer in lc.layers():
        seen: set = set()
        for g in layer:
            qs = set(g[1:])
            assert seen.isdisjoint(qs)
            seen |= qs


def test_reject_bad_gates():
    with pytest.raises(ValueError):
        LogicalCircuit([("CCX", 0, 1, 2)])
    with pytest.raises(ValueError):
        LogicalCircuit([("CNOT", 0, 0)])
