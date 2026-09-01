"""The H6 [[6,2,2]] magic factory in isolation."""

import numpy as np
import clifft

from ftsim.qec import build, magic_factory_lines

_STAB = ((0, 1, 2, 3), (2, 3, 4, 5))
_SLOT0 = (0, 2, 4)


def _factory_probe(se_rounds, shots=30_000):
    lay = build(0, 1)
    mp = lay.magic(0)
    d = mp.data
    lines, man = magic_factory_lines(mp.data, mp.xanc, mp.zanc, mp.bare, se_rounds=se_rounds)
    body = "R 0\n" + "\n".join(lines) + "\n"
    for supp in _STAB:                        # stabilizer expectation probes
        body += "EXP_VAL " + "*".join(f"X{d[j]}" for j in supp) + "\n"
        body += "EXP_VAL " + "*".join(f"Z{d[j]}" for j in supp) + "\n"
    for P in "XYZ":
        body += "EXP_VAL " + "*".join(f"{P}{d[j]}" for j in _SLOT0) + "\n"
    body += "M " + " ".join(map(str, mp.data)) + "\n"
    r = clifft.sample(clifft.compile(body), shots=shots, seed=1)
    m = np.asarray(r.measurements, np.uint8)
    keep = np.ones(len(m), bool)
    for grp in man["postselect"]:
        keep &= (m[:, list(grp)].sum(1) % 2 == 0)
    ev = np.asarray(r.exp_vals, float)[keep].mean(axis=0)
    return keep.mean(), ev[:4], ev[4:]        # yield, stab exps, (slot0 X,Y,Z)


def test_factory_yield_is_reasonable():
    y, _, _ = _factory_probe(se_rounds=1)
    assert 0.02 < y < 1.0                     # post-selection discards, but not everything


def test_factory_output_is_in_codespace():
    _, stabs, _ = _factory_probe(se_rounds=1)
    assert np.all(np.abs(stabs - 1.0) < 0.05)  # post-selected -> stabilizers = +1


def test_factory_output_is_a_magic_state():
    """Not a stabilizer state: some logical Pauli expectation is strictly between -1 and 1
    on an axis where a stabilizer state would be +-1 or 0."""
    _, _, (sx, sy, sz) = _factory_probe(se_rounds=1)
    r = (sx * sx + sy * sy + sz * sz) ** 0.5
    assert r > 0.3                             # non-trivial Bloch vector
    assert not any(abs(abs(v) - 1.0) < 1e-3 for v in (sx, sy, sz))  # not a Pauli eigenstate
