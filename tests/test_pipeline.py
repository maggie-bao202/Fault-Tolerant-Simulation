"""End-to-end: noisy sweeps behave (yield down, LER up with p) across codes."""

import math

import pytest

from ftsim import LogicalCircuit, run_pipeline


def _monotone_sweep(lc, ps, *, code="h6", protocol="processor", shots=20_000, **kw):
    prev_acc, prev_ler = 1.0 + 1e-9, -1.0
    last = None
    for p in ps:
        r = run_pipeline(
            lc, p=p, code=code, protocol=protocol, shots=shots,
            check=(p == 0.0 and protocol == "processor"), **kw
        )
        assert r.post_selection_rate <= prev_acc + 1e-9, (p, r)
        assert r.logical_error_rate >= prev_ler - 2e-3, (p, r)
        prev_acc, prev_ler = r.post_selection_rate, r.logical_error_rate
        last = r
    return last


def test_h6_clifford_sweep_monotonic():
    r = _monotone_sweep(
        LogicalCircuit([("H", 0), ("CNOT", 0, 1)]), (0.0, 1e-3, 3e-3, 1e-2), shots=30_000
    )
    assert r.logical_error_rate > 0.0


def test_h6_clifford_zero_noise_perfect_with_unitary_check():
    r = run_pipeline(
        LogicalCircuit([("S", 0), ("CNOT", 0, 1), ("H", 1)]), p=0.0, shots=20_000
    )
    assert r.post_selection_rate == 1.0
    assert r.logical_error_rate == 0.0
    assert r.unitary_check_passed


def test_h6_t_circuit_runs_and_reports():
    lc = LogicalCircuit([("T", 0), ("CNOT", 0, 1), ("T", 1)])
    r = run_pipeline(lc, p=1e-3, shots=60_000, zero_lvl_distill=True)
    assert r.t_count == 2
    assert r.physical_qubits >= 6 * 10
    assert r.magic == "h6_zero_level_distillation"
    assert not math.isnan(r.logical_error_rate)
    assert r.logical_error_rate_net >= 0.0
    assert r.code == "h6" and r.protocol == "processor"


def test_h6_t_default_magic_is_zero_level():
    r = run_pipeline(LogicalCircuit([("T", 0)]), p=0.0, shots=10_000)
    assert r.magic == "h6_zero_level_distillation"


def test_factory_standalone_zero_level():
    prev = 1.0 + 1e-9
    for p in (0.0, 1e-3, 5e-3):
        r = run_pipeline(None, p=p, protocol="factory", zero_lvl_distill=True, shots=20_000)
        assert r.protocol == "factory"
        assert r.post_selection_rate <= prev + 1e-9
        prev = r.post_selection_rate
    assert r.post_selection_rate < 1.0


def test_factory_standalone_distillation_level_1():
    r = run_pipeline(None, p=1e-3, protocol="factory", distillation=1, shots=20_000)
    assert r.magic == "h6_distillation"
    assert 0.0 <= r.logical_error_rate <= 1.0


def test_cultivation_and_level2_are_not_implemented():
    with pytest.raises(NotImplementedError):
        run_pipeline(LogicalCircuit([("T", 0)]), p=1e-3, cultivation=True)
    with pytest.raises(NotImplementedError):
        run_pipeline(LogicalCircuit([("T", 0)]), p=1e-3, distillation=2)


def test_at_most_one_magic_flag():
    with pytest.raises(ValueError):
        run_pipeline(LogicalCircuit([("T", 0)]), p=1e-3,
                     zero_lvl_distill=True, distillation=1)


def test_rotated_surface_logical_circuit_sweep():
    _monotone_sweep(
        LogicalCircuit([("X", 0), ("CNOT", 0, 1)]), (0.0, 1e-3, 3e-3),
        code="rotated_surface", shots=30_000,
    )


@pytest.mark.parametrize("code", ["h6", "rotated_surface", "repetition", "color"])
def test_memory_protocol_sweep(code):
    prev = 1.0 + 1e-9
    for p in (0.0, 1e-3, 5e-3):
        r = run_pipeline(
            LogicalCircuit([("X", 0)]), p=p, code=code, protocol="memory",
            shots=15_000, rounds=3,
        )
        assert r.post_selection_rate <= prev + 1e-9
        prev = r.post_selection_rate
        assert 0.0 <= r.logical_error_rate <= 1.0
    assert r.post_selection_rate < 1.0
