"""Parsing Catalyst ``pbc``-dialect MLIR text into a PBCCircuit.

The fixtures in ``tests/fixtures/ppm/`` are the ``QuantumCompilationStage`` MLIR
emitted by ``qml.qjit`` (capture on) after ``to_ppr -> commute_ppr ->
merge_ppr_ppm``, for a few small circuits.  Regenerate them and re-check these
expectations when bumping ``pennylane-catalyst`` (the ``pbc`` syntax is
version-sensitive -- see ``ftsim.ppm.catalyst_frontend.TESTED_CATALYST``).
"""

from pathlib import Path

import pytest

from ftsim.ppm.ir import PPM, PPR
from ftsim.ppm.mlir_parse import parse_pbc
from ftsim.ppm._errors import PPMUnsupported

FIX = Path(__file__).parent / "fixtures" / "ppm"


def _parse(name, n):
    return parse_pbc((FIX / f"{name}.mlir").read_text(encoding="utf-8"), n)


def test_clifford_roundtrip_is_all_pi4_pprs():
    pbc = _parse("bell_roundtrip", 2)
    assert pbc.is_clifford and pbc.t_count == 0
    assert all(r.angle == "pi/4" for r in pbc.rotations)
    assert {r.weight for r in pbc.rotations} == {1, 2}       # 1- and 2-qubit PPRs
    assert pbc.measurements == [PPM(("Z",), (0,)), PPM(("Z",), (1,))]


def test_s_roundtrip_also_clifford():
    pbc = _parse("s_roundtrip", 2)
    assert pbc.is_clifford
    assert PPR(("Z",), (1,), "pi/4") in pbc.rotations
    assert PPR(("Z",), (1,), "pi/4", sign=-1) in pbc.rotations
    assert pbc.measurements == [PPM(("Z",), (0,)), PPM(("Z",), (1,))]


def test_single_t_keeps_one_pi8_rotation():
    pbc = _parse("single_t", 1)
    assert pbc.rotations == [PPR(("Z",), (0,), "pi/8")]
    assert pbc.measurements == [PPM(("Z",), (0,))]
    assert not pbc.is_clifford and pbc.t_count == 1


def test_t_circuit_pi8_signs_and_multibody_and_threading():
    pbc = _parse("t_circuit", 2)
    # T(0) -> +pi/8 Z0 ; adjoint(T)(1) -> -pi/8 Z1 (threaded through the CNOT PPRs)
    pi8 = [r for r in pbc.rotations if r.angle == "pi/8"]
    assert pi8 == [PPR(("Z",), (0,), "pi/8"), PPR(("Z",), (1,), "pi/8", sign=-1)]
    assert PPR(("Z", "X"), (0, 1), "pi/4") in pbc.rotations   # 2-qubit CNOT PPR
    assert pbc.measurements == [PPM(("Z",), (0,)), PPM(("Z",), (1,))]
    assert pbc.t_count == 2


def test_conditioned_ppr_is_rejected():
    text = '%r = pbc.ppr ["Z"](2) %q0 cond(%m) : !quantum.bit'
    with pytest.raises(PPMUnsupported, match="feed-forward"):
        parse_pbc(
            "%q0 = quantum.extract %r[ 0 ] : x\n" + text +
            '\n%o = quantum.namedobs %r[ PauliZ ] : !quantum.obs\n', 1)


def test_unresolved_qubit_ssa_is_rejected():
    with pytest.raises(PPMUnsupported, match="resolve qubit SSA"):
        parse_pbc('%o = quantum.namedobs %qX[ PauliZ ] : !quantum.obs\n', 1)


def test_no_terminal_measurement_is_rejected():
    text = (
        "%q0 = quantum.extract %r[ 0 ] : x\n"
        '%2 = pbc.ppr ["Z"](4) %q0 : !quantum.bit\n'
    )
    with pytest.raises(PPMUnsupported, match="no terminal measurements"):
        parse_pbc(text, 1)
