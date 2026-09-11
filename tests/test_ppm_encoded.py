"""Encoded lowering of the Clifford part of a PBC circuit.

Catalyst-free: these drive the encoded Pauli-product-measurement primitive
(``lower_pbc``) directly with hand-built ``PBCCircuit``s.  The end-to-end
``run_pipeline(..., frontend="ppm")`` path is exercised in
``tests/test_ppm_pipeline.py`` (needs ``pennylane-catalyst``).
"""

import numpy as np
import pytest

from ftsim.backends import UnsupportedGate
from ftsim.ppm import PBCCircuit, PPM, PPR, lower_pbc
from ftsim.score import reference


def _identity_pbc(n):
    return PBCCircuit(n, [], [PPM(("Z",), (i,)) for i in range(n)])


@pytest.mark.parametrize("code", ["h6", "steane"])
def test_clifford_roundtrip_pbc_lowers_deterministically(code):
    comp = lower_pbc(_identity_pbc(2), code, input_state="00")

    assert comp.n_logical == 2
    assert comp.sbranch_detectors == []
    assert comp.qec_detectors

    ref = reference(comp)                       # raises if qec dets not det-0
    assert 0.0 < ref.noiseless_yield <= 1.0
    assert ref.error_floor <= 0.02

    d, o = comp.circuit.compile_detector_sampler(seed=0).sample(
        4000, separate_observables=True
    )
    d = np.asarray(d, np.uint8)
    o = np.asarray(o, np.uint8)
    assert not d[:, comp.qec_detectors].any()   # QEC detectors deterministic-0
    assert not o.any()                          # identity on |00> -> all obs 0
    comp.circuit.detector_error_model(decompose_errors=False)   # DEM builds


@pytest.mark.parametrize("ins", ["0", "1"])
def test_z_basis_inputs_stay_deterministic(ins):
    # the round-trip is identity on any computational-basis input, so the
    # standing observables are detector-style parities -- deterministically 0
    # noiselessly for either input bit (they flip only under a logical fault).
    comp = lower_pbc(_identity_pbc(1), "h6", input_state=ins)
    reference(comp)                              # raises on a non-deterministic build
    _d, o = comp.circuit.compile_detector_sampler(seed=1).sample(
        2000, separate_observables=True
    )
    assert not np.asarray(o, np.uint8).any()


def test_non_z_input_state_is_rejected():
    with pytest.raises(UnsupportedGate, match="Z-basis"):
        lower_pbc(_identity_pbc(2), "h6", input_state="+0")


def test_pi8_rotation_hits_the_magic_stub():
    pbc = PBCCircuit(1, [PPR(("Z",), (0,), "pi/8")], [PPM(("Z",), (0,))])
    with pytest.raises(NotImplementedError, match="magic-state injection"):
        lower_pbc(pbc, "h6", input_state="0")


def test_multiqubit_clifford_ppr_pair_round_trips_to_identity():
    # exp(-i pi/4 Z0 Z1) followed by its inverse == identity: deterministic build
    pbc = PBCCircuit(
        2,
        [PPR(("Z", "Z"), (0, 1), "pi/4"), PPR(("Z", "Z"), (0, 1), "pi/4", sign=-1)],
        [PPM(("Z",), (0,)), PPM(("Z",), (1,))],
    )
    comp = lower_pbc(pbc, "h6", input_state="00")
    reference(comp)
    d, o = comp.circuit.compile_detector_sampler(seed=0).sample(
        3000, separate_observables=True
    )
    assert not np.asarray(d, np.uint8)[:, comp.qec_detectors].any()
    assert not np.asarray(o, np.uint8).any()


def test_unsupported_code_rejected_before_catalyst():
    with pytest.raises(UnsupportedGate, match="transversal H/S/CNOT"):
        lower_pbc(_identity_pbc(1), "rotated_surface", input_state="0")
