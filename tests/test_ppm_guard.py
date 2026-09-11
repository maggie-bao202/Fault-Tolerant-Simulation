"""Guard rails for the opt-in PPM frontend + regression that the default
gate path is untouched.  None of this needs pennylane/catalyst -- every case
fails (or succeeds) before the lazy Catalyst import.
"""

import pytest

from ftsim import LogicalCircuit, list_codes, list_protocols, run_pipeline
from ftsim.backends import UnsupportedGate


@pytest.mark.parametrize("code", ["rotated_surface", "repetition", "color"])
def test_ppm_frontend_rejects_codes_without_transversal_hs(code):
    with pytest.raises(UnsupportedGate):
        run_pipeline(LogicalCircuit([("H", 0), ("CNOT", 0, 1)]), p=0.0,
                     code=code, frontend="ppm", check=False, shots=1)


def test_unknown_frontend_is_rejected():
    with pytest.raises(ValueError, match="frontend must be"):
        run_pipeline(LogicalCircuit([("H", 0)]), p=0.0, frontend="bogus", shots=1)


def test_ppm_frontend_only_for_processor():
    with pytest.raises(ValueError, match="only applies to protocol='processor'"):
        run_pipeline(LogicalCircuit([("X", 0)]), p=0.0, protocol="memory",
                     frontend="ppm", shots=1)


def test_registries_unchanged():
    assert list_protocols() == ["processor", "memory", "factory"]
    assert {"h6", "rotated_surface", "repetition", "color"} <= set(list_codes())


def test_default_frontend_is_gate_and_unchanged():
    r = run_pipeline(LogicalCircuit([("H", 0), ("CNOT", 0, 1)]), p=0.0,
                     shots=20_000)
    assert r.frontend == "gate"
    assert r.post_selection_rate == 1.0
    assert r.logical_error_rate == 0.0
    assert "frontend" not in str(r)          # only shown when != "gate"
