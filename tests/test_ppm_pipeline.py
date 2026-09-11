"""End-to-end ``run_pipeline(..., frontend="ppm")`` -- needs pennylane +
pennylane-catalyst (the whole module is skipped otherwise)."""

import math

import pytest

pytest.importorskip("catalyst")

from ftsim import LogicalCircuit, run_pipeline  # noqa: E402


@pytest.mark.parametrize("code", ["h6", "steane"])
def test_ppm_clifford_zero_noise_is_perfect(code):
    lc = LogicalCircuit([("H", 0), ("CNOT", 0, 1)])
    r = run_pipeline(lc, p=0.0, code=code, shots=20_000,
                     frontend="ppm", check=False)
    assert r.frontend == "ppm"
    assert r.protocol == "processor"
    assert r.post_selection_rate == 1.0
    assert r.logical_error_rate == 0.0


def test_ppm_noisy_sweep_is_monotone():
    lc = LogicalCircuit([("S", 0), ("CNOT", 0, 1), ("H", 1)])
    prev_acc = 1.0 + 1e-9
    for p in (0.0, 1e-3, 5e-3):
        r = run_pipeline(lc, p=p, shots=40_000, frontend="ppm", check=False)
        assert not math.isnan(r.post_selection_rate)
        assert r.post_selection_rate <= prev_acc + 1e-9, (p, r)
        prev_acc = r.post_selection_rate


@pytest.mark.parametrize("kw", [dict(zero_lvl_distill=True), dict()])
def test_ppm_t_circuit_raises_not_implemented(kw):
    with pytest.raises(NotImplementedError, match="magic-state injection"):
        run_pipeline(LogicalCircuit([("T", 0)]), p=1e-3, shots=10_000,
                     frontend="ppm", check=False, **kw)
