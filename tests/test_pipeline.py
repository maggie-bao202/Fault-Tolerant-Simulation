"""End-to-end: noisy sweeps behave (yield down, net LER up with p)."""

import math

import pytest

from ftsim.logical_gates import LogicalCircuit
from ftsim.pipeline import run_pipeline


def test_clifford_sweep_monotonic():
    lc = LogicalCircuit([("H", 0), ("CNOT", 0, 1)])
    prev_acc, prev_ler = 1.01, -1.0
    for p in (0.0, 1e-3, 3e-3, 1e-2):
        r = run_pipeline(lc, p=p, shots=20_000, check=(p == 0.0))
        assert r.post_selection_rate <= prev_acc + 1e-9
        assert r.logical_error_rate >= prev_ler - 1e-3
        prev_acc, prev_ler = r.post_selection_rate, r.logical_error_rate
    assert r.logical_error_rate > 0.0            # noise did cause logical errors


def test_clifford_zero_noise_is_perfect():
    r = run_pipeline(LogicalCircuit([("S", 0), ("CNOT", 0, 1), ("H", 1)]),
                     p=0.0, shots=20_000)
    assert r.post_selection_rate == 1.0
    assert r.logical_error_rate == 0.0
    assert r.unitary_check_passed


def test_t_circuit_runs_and_reports():
    lc = LogicalCircuit([("T", 0), ("CNOT", 0, 1), ("T", 1)])
    r = run_pipeline(lc, p=1e-3, shots=40_000)
    assert r.t_count == 2
    assert r.physical_qubits >= 4 * 12            # 2 data + 2*2 magic patches
    assert 0.0 <= r.post_selection_rate <= 1.0
    # net LER (coherent floor removed) is a small positive noise signal
    assert not math.isnan(r.logical_error_rate)
    assert r.logical_error_rate_net >= 0.0


def test_report_str():
    r = run_pipeline(LogicalCircuit([("H", 0)]), p=1e-3, shots=5_000, check=False)
    s = str(r)
    assert "accept=" in s and "LER=" in s
