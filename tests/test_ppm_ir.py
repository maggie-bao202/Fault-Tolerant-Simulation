"""The PBC intermediate representation: PPR / PPM / PBCCircuit."""

import pytest

from ftsim.ppm.ir import PPM, PPR, PBCCircuit


def test_ppr_fields_and_normalisation():
    r = PPR(paulis=("Z", "y"), qubits=(2, 0), angle="pi/8", sign=-1)
    # I dropped, letters upper-cased, pairs sorted by ascending qubit
    assert r.paulis == ("Y", "Z")
    assert r.qubits == (0, 2)
    assert r.angle == "pi/8"
    assert r.sign == -1
    assert r.weight == 2
    assert not r.is_clifford


@pytest.mark.parametrize(
    "angle, clifford",
    [("pi/2", True), ("pi/4", True), ("pi/8", False)],
)
def test_ppr_is_clifford(angle, clifford):
    assert PPR(("Z",), (0,), angle).is_clifford is clifford


def test_ppr_drops_identity_factors():
    r = PPR(paulis=("I", "Z", "I"), qubits=(0, 1, 2), angle="pi/4")
    assert r.paulis == ("Z",)
    assert r.qubits == (1,)


def test_ppm_fields():
    m = PPM(paulis=("Z",), qubits=(3,), sign=-1)
    assert m.paulis == ("Z",)
    assert m.qubits == (3,)
    assert m.sign == -1
    assert m.weight == 1


@pytest.mark.parametrize("bad", [
    dict(paulis=("Z", "X"), qubits=(0,), angle="pi/4"),       # length mismatch
    dict(paulis=("Z",), qubits=(0,), angle="pi/3"),           # bad angle
    dict(paulis=("Z",), qubits=(0,), angle="pi/4", sign=0),   # bad sign
    dict(paulis=("W",), qubits=(0,), angle="pi/4"),           # bad letter
    dict(paulis=("Z", "Z"), qubits=(0, 0), angle="pi/4"),     # repeated qubit
    dict(paulis=("I",), qubits=(0,), angle="pi/4"),           # all identity
])
def test_ppr_rejects_bad_shapes(bad):
    with pytest.raises(ValueError):
        PPR(**bad)


def test_pbc_partition_preserves_program_order():
    r0 = PPR(("Z",), (0,), "pi/4")
    r1 = PPR(("Z",), (0,), "pi/8")
    r2 = PPR(("X",), (1,), "pi/2")
    r3 = PPR(("Y",), (1,), "pi/8", sign=-1)
    pbc = PBCCircuit(2, [r0, r1, r2, r3], [PPM(("Z",), (0,)), PPM(("Z",), (1,))])

    cliff, pi8 = pbc.partition()
    assert cliff == [r0, r2]
    assert pi8 == [r1, r3]
    assert pbc.t_count == 2
    assert not pbc.is_clifford


def test_pbc_is_clifford_when_no_pi8():
    pbc = PBCCircuit(1, [PPR(("Z",), (0,), "pi/4")], [PPM(("Z",), (0,))])
    assert pbc.is_clifford
    assert pbc.t_count == 0
    assert pbc.partition() == ([PPR(("Z",), (0,), "pi/4")], [])
