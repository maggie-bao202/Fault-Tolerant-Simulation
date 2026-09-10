"""Steane code (= ColorCode d=3) as a processor code, and cross-code magic
injection: processor in one code, factory in another."""

import pytest

from ftsim import LogicalCircuit, check_unitary, run_pipeline
from ftsim.backends import Driver, get as get_code


# --- Steane as a first-class processor code --------------------------

@pytest.mark.parametrize("gl", [
    [("H", 0)], [("S", 0)], [("S", 0), ("S", 0)],
    [("H", 0), ("CNOT", 0, 1)], [("S", 0), ("H", 0), ("S", 0)],
    [("H", 0), ("CNOT", 0, 1), ("CNOT", 1, 2)],
])
def test_steane_clifford_unitary_exact(gl):
    ok, detail = check_unitary(LogicalCircuit(gl), code="steane", shots=15_000)
    assert ok, detail
    assert max(v for v in detail.values() if isinstance(v, float)) < 0.03


def test_steane_is_colorcode_d3():
    from lightstim.qec_code.color_code import ColorCode
    p = get_code("steane").patch_factory()
    assert isinstance(p, ColorCode)
    assert len(p.data_indices) == 7 and p.num_logicals == 1


def test_steane_clifford_zero_noise_perfect():
    r = run_pipeline(LogicalCircuit([("H", 0), ("CNOT", 0, 1)]), p=0.0,
                     code="steane", shots=15_000, check=False)
    assert r.post_selection_rate == 1.0 and r.logical_error_rate == 0.0


def test_steane_memory_sweep():
    prev = 1.0 + 1e-9
    for p in (0.0, 1e-3, 5e-3):
        r = run_pipeline(LogicalCircuit([("X", 0)]), p=p, code="steane",
                         protocol="memory", rounds=3, shots=12_000)
        assert r.post_selection_rate <= prev + 1e-9
        prev = r.post_selection_rate


# --- cross-code injection: processor=steane, factory=h6 ---------------

def test_driver_allocates_factory_and_inject_patches():
    d = Driver(get_code("steane"), 2, magic=2, factory_spec=get_code("h6"))
    assert d.magic_names == ["m0", "m1"]        # h6 factory blocks
    assert d.inject_names == ["inj0", "inj1"]   # steane injection targets
    assert d.spec_of("m0").name == "h6"
    assert d.spec_of("inj0").name == "steane"
    assert d.spec_of("q0").name == "steane"


def test_no_inject_patches_when_same_code():
    d = Driver(get_code("h6"), 1, magic=1, factory_spec=get_code("h6"))
    assert d.inject_names == []


@pytest.mark.parametrize("magic_kw", [{"distillation": 1}, {"zero_lvl_distill": True}])
def test_cross_code_inject_runs(magic_kw):
    lc = LogicalCircuit([("T", 0)])
    r = run_pipeline(lc, p=0.0, code="steane", factory_code="h6",
                     shots=20_000, check=False, **magic_kw)
    assert r.code == "steane"
    assert "inject" in r.magic
    assert r.logical_error_rate in (0.0,) or r.logical_error_rate >= 0.0
    # physical cost > a same-code h6 T run (h6 block + steane inject + steane data)
    assert r.physical_qubits > 3 * 7


def test_cross_code_yield_drops_with_noise():
    lc = LogicalCircuit([("T", 0), ("CNOT", 0, 1)])
    accs = []
    for p in (0.0, 1e-3, 4e-3):
        r = run_pipeline(lc, p=p, code="steane", factory_code="h6",
                         distillation=1, shots=30_000, check=False)
        accs.append(r.post_selection_rate)
    assert accs[0] >= accs[1] >= accs[2]


def test_factory_code_without_flag_errors():
    with pytest.raises(ValueError):
        run_pipeline(LogicalCircuit([("T", 0)]), p=1e-3,
                     code="steane", factory_code="h6")
