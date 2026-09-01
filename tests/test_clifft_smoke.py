"""clifft is the simulation backend -- sanity-check the API this repo relies on."""

import numpy as np
import clifft


def test_real_t_expectation():
    # |0> -H-> |+> -T-> ; <X> = <Y> = cos(pi/4)
    prog = clifft.compile("R 0\nH 0\nT 0\nEXP_VAL X0\nEXP_VAL Y0\nEXP_VAL Z0\nM 0\n")
    r = clifft.sample(prog, shots=1, seed=0)
    x, y, z = r.exp_vals[0]
    assert abs(x - 2 ** -0.5) < 1e-9
    assert abs(y - 2 ** -0.5) < 1e-9
    assert abs(z) < 1e-9


def test_exp_val_pauli_product_star_separator():
    # GHZ: <X0 X1 X2> = +1, <Z0 Z1> = +1
    prog = clifft.compile(
        "R 0 1 2\nH 0\nCX 0 1\nCX 0 2\nEXP_VAL X0*X1*X2\nEXP_VAL Z0*Z1\nM 0 1 2\n"
    )
    ev = clifft.sample(prog, shots=1, seed=0).exp_vals[0]
    assert abs(ev[0] - 1.0) < 1e-9
    assert abs(ev[1] - 1.0) < 1e-9


def test_sample_survivors_fields():
    prog = clifft.compile(
        "R 0 1\nH 0\nCNOT 0 1\nX_ERROR(0.1) 0\nM 0 1\n"
        "DETECTOR rec[-2] rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\n",
        postselection_mask=[1],
    )
    s = clifft.sample_survivors(prog, shots=5000, seed=0)
    assert s.total_shots == 5000
    assert 0 < s.passed_shots <= 5000
    assert s.passed_shots + s.discards == s.total_shots
    assert len(s.observable_ones) == 1


def test_accepts_lightstim_style_noise_instructions():
    for instr in ("DEPOLARIZE1(0.1) 0", "DEPOLARIZE2(0.1) 0 1",
                  "X_ERROR(0.1) 0", "Z_ERROR(0.1) 0", "T_DAG 0", "MR 0"):
        clifft.compile("R 0 1\n" + instr + "\nM 0 1\n")
