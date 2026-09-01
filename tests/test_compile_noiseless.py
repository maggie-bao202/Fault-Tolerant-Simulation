"""Noiseless compiled circuits: post-selection keeps everything (up to the
distillation / no-S branch), the round-trip is (nearly) identity, and the
Clifford part exactly matches the intended unitary."""

import numpy as np
import pytest

from ftsim.logical_gates import LogicalCircuit
from ftsim.compiler import compile as compile_lc
from ftsim.pipeline import check_unitary, run_pipeline
from ftsim.sim import reference

CLIFFORD_BANK = [
    [],
    [("H", 0)],
    [("H", 0), ("CNOT", 0, 1)],
    [("S", 0), ("H", 0), ("S", 0)],
    [("H", 0), ("CNOT", 0, 1), ("CNOT", 1, 2)],
]

T_BANK = [
    [("T", 0)],
    [("T", 0), ("T", 0)],            # = S
    [("S", 0), ("T", 0)],
    [("T", 0), ("T", 1), ("CNOT", 0, 1), ("S", 0)],
]


@pytest.mark.parametrize("gl", CLIFFORD_BANK)
def test_clifford_noiseless_exact(gl):
    lc = LogicalCircuit(gl)
    ok, detail = check_unitary(lc, shots=20_000)
    assert ok, detail
    assert max(detail.values(), default=0.0) < 0.02
    rep = run_pipeline(lc, p=0.0, shots=20_000, check=False)
    assert rep.post_selection_rate == 1.0
    assert rep.logical_error_rate == 0.0


@pytest.mark.parametrize("gl", T_BANK)
def test_t_noiseless(gl):
    lc = LogicalCircuit(gl)
    ok, detail = check_unitary(lc, shots=20_000)
    assert ok, detail                       # generous atol tolerates coherent T error
    comp = compile_lc(LogicalCircuit(list(lc.gates) + list(lc.inverse().gates)))
    ref = reference(comp, shots=20_000)
    assert ref.noiseless_yield > 0
    # round-trip noiseless coherent floor stays small for non-H-conjugated T
    assert ref.error_floor < 0.05


def test_h_conjugated_t_is_flagged_not_silently_wrong():
    """T sandwiched by H has a known residual coherent error -- it must show up
    in the report's error floor, not be silently absorbed."""
    lc = LogicalCircuit([("H", 0), ("T", 0), ("H", 0)])
    comp = compile_lc(LogicalCircuit(list(lc.gates) + list(lc.inverse().gates)))
    ref = reference(comp, shots=20_000)
    assert ref.error_floor > 0.02          # the coherent imperfection is visible
