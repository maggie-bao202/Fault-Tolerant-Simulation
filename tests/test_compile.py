"""Noiseless compiled circuits: deterministic detectors, exact Clifford unitary,
T circuits build and run (Clifford proxy: T acts as identity)."""

import pytest

from ftsim import LogicalCircuit, check_unitary, run_pipeline
from ftsim.factory import make_magic_source
from ftsim.processor import compile_logical_circuit
from ftsim.score import reference

H6_CLIFFORD = [
    [("H", 0)],
    [("H", 0), ("CNOT", 0, 1)],
    [("S", 0), ("H", 0), ("S", 0)],
    [("H", 0), ("CNOT", 0, 1), ("CNOT", 1, 2)],
    [("S", 0), ("S", 0)],
    [("X", 0)],
]

H6_T = [
    [("T", 0)],
    [("T", 0), ("T", 0)],
    [("S", 0), ("T", 0)],
    [("T", 0), ("T", 1), ("CNOT", 0, 1), ("S", 0)],
]


def _compiled(gl, code="h6", ins=None, magic=None):
    lc = LogicalCircuit(gl)
    n = lc.n_qubits
    return compile_logical_circuit(
        lc, code, input_state=(ins or "0" * n), magic_source=magic
    )


@pytest.mark.parametrize("gl", H6_CLIFFORD)
def test_h6_clifford_noiseless_is_deterministic_and_exact(gl):
    comp = _compiled(gl)
    d, o = comp.circuit.compile_detector_sampler().sample(4000, separate_observables=True)
    assert not d.any()
    comp.circuit.detector_error_model(decompose_errors=False)
    ok, detail = check_unitary(LogicalCircuit(gl), shots=20_000)
    assert ok, detail
    assert max(v for v in detail.values() if isinstance(v, float)) < 0.02


@pytest.mark.parametrize("gl", H6_CLIFFORD)
def test_h6_clifford_zero_noise_is_perfect(gl):
    rep = run_pipeline(LogicalCircuit(gl), p=0.0, shots=20_000, check=False)
    assert rep.post_selection_rate == 1.0
    assert rep.logical_error_rate == 0.0


@pytest.mark.parametrize("gl", H6_T)
@pytest.mark.parametrize("magic_kw", [{"zero_lvl_distill": True}, {"distillation": 1}])
def test_h6_t_circuit_compiles_and_scores(gl, magic_kw):
    lc = LogicalCircuit(gl)
    magic = make_magic_source(**magic_kw)
    comp = compile_logical_circuit(
        LogicalCircuit(list(lc.gates) + list(lc.inverse().gates)),
        "h6", magic_source=magic, input_state="0" * lc.n_qubits,
    )
    assert comp.sbranch_detectors                 # each T adds a no-S-branch detector
    ref = reference(comp)
    assert 0.0 < ref.noiseless_yield <= 1.0
    d, _ = comp.circuit.compile_detector_sampler().sample(2000, separate_observables=True)
    assert not d[:, comp.qec_detectors].any()
    assert 0.0 < d[:, comp.sbranch_detectors].mean() < 1.0


def test_h6_t_proxy_is_identity_on_the_logical_state():
    r = run_pipeline(LogicalCircuit([("T", 0), ("T", 0)]), p=0.0, shots=20_000,
                     zero_lvl_distill=True)
    assert r.logical_error_rate == 0.0
    assert r.noiseless_error_floor == 0.0
    assert "identity" in r.unitary_check_detail.get("note", "")
    assert r.magic == "h6_zero_level_distillation"


def test_rotated_surface_x_cnot_noiseless_exact():
    comp = _compiled([("X", 0), ("CNOT", 0, 1)], code="rotated_surface", ins="00")
    d, _ = comp.circuit.compile_detector_sampler().sample(2000, separate_observables=True)
    assert not d.any()
    ok, detail = check_unitary(
        LogicalCircuit([("X", 0), ("CNOT", 0, 1)]), code="rotated_surface", shots=15_000
    )
    assert ok, detail
