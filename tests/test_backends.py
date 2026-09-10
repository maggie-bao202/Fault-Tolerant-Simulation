"""Code registry + the CodeSpec / Driver contract (codes carry no magic logic)."""

import pytest

from ftsim import list_codes, list_protocols
from ftsim.backends import CodeSpec, Driver, UnsupportedGate, get as get_code
from ftsim.logical_gates import LogicalCircuit
from ftsim.processor import compile_logical_circuit


def test_registries():
    assert {"h6", "rotated_surface", "repetition", "color"} <= set(list_codes())
    assert list_protocols() == ["processor", "memory", "factory"]


def test_code_specs_point_at_lightstim():
    from lightstim.ir.qec_patch import QECPatch
    for name in list_codes():
        spec = get_code(name)
        assert isinstance(spec, CodeSpec)
        assert isinstance(spec.patch_factory(), QECPatch)


def test_codespec_has_no_magic_field():
    # magic-state behaviour is a protocol (ftsim.factory), not a code property
    assert not hasattr(get_code("h6"), "magic")
    assert not hasattr(get_code("h6"), "supports_magic")


def test_unknown_code_raises():
    with pytest.raises(KeyError):
        get_code("nonesuch")


def test_driver_lays_out_multi_patch_system():
    d = Driver(get_code("h6"), 2, magic=3)
    assert d.data_names == ["q0", "q1"]
    assert d.magic_names == ["m0", "m1", "m2"]
    assert d.system.num_qubits == 5 * 10
    all_data = [q for n in d.data_names + d.magic_names for q in d.data(n)]
    assert len(all_data) == len(set(all_data)) == 5 * 6


def test_gate_dispatch_uses_lightstim_opset_methods():
    spec = get_code("h6")
    assert spec.gate_methods["H"] == "transversal_hadamard"
    assert spec.gate_methods["CNOT"] == "transversal_cnot"
    assert callable(getattr(spec.op_set, spec.gate_methods["S"]))


def test_rotated_surface_hs_are_unsupported():
    with pytest.raises(UnsupportedGate):
        compile_logical_circuit(LogicalCircuit([("H", 0)]), "rotated_surface", input_state="0")


def test_generic_code_has_no_gate_set():
    with pytest.raises(UnsupportedGate):
        compile_logical_circuit(LogicalCircuit([("X", 0)]), "repetition", input_state="0")


def test_t_without_magic_source_is_rejected():
    with pytest.raises(ValueError):
        compile_logical_circuit(LogicalCircuit([("T", 0)]), "h6", input_state="0")
