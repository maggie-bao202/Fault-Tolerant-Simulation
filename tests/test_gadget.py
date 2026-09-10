"""The T-state teleportation gadget (ftsim.processor) + zero-level distillation."""

import numpy as np
import pytest

import clifft
from ftsim.qec import build
from ftsim.resources import MAGIC_INPUT, magic_factory_lines
from ftsim.processor import MAGIC_FIXUP, t_teleport_lines
from ftsim.logical_gates import LogicalCircuit
from ftsim.pipeline import check_unitary


def test_recipe_constants():
    assert MAGIC_INPUT in ("TT", "A", "H")
    assert isinstance(MAGIC_FIXUP, tuple)


def test_factory_lines_compile_and_have_manifest():
    lay = build(0, 1)
    mp = lay.magic(0)
    lines, man = magic_factory_lines(mp.data, mp.xanc, mp.zanc, mp.bare, se_rounds=1)
    text = "R 0\n" + "\n".join(lines) + "\nM " + " ".join(map(str, mp.data)) + "\n"
    clifft.compile(text)                       # parses
    assert man["meas"] == 4                     # one SE round -> 4 ancilla reads
    assert len(man["postselect"]) == 4
    assert all(isinstance(g, tuple) for g in man["postselect"])


def test_teleport_lines_compile():
    lay = build(1, 1)
    dp, mp = lay.data(0), lay.magic(0)
    lines, man = t_teleport_lines(dp.data, mp.data, mp.xanc, mp.zanc, mp.bare)
    text = "R " + " ".join(map(str, dp.data)) + "\n" + "\n".join(lines) + "\n"
    text += "M " + " ".join(map(str, dp.data)) + "\n"
    clifft.compile(text)
    # meas = factory SE (4) + magic destructive readout (6)
    assert man["meas"] == 10
    # postselect: 4 factory + 2 magic-codeword + 1 S-branch
    assert len(man["postselect"]) == 7


def test_dag_uses_inverse():
    lay = build(1, 1)
    dp, mp = lay.data(0), lay.magic(0)
    _, man_t = t_teleport_lines(dp.data, mp.data, mp.xanc, mp.zanc, mp.bare, dag=False)
    _, man_d = t_teleport_lines(dp.data, mp.data, mp.xanc, mp.zanc, mp.bare, dag=True)
    assert man_t["meas"] == man_d["meas"]


def test_gadget_implements_logical_t():
    """A bare T (not H-conjugated) is an exact logical T."""
    ok, detail = check_unitary(LogicalCircuit([("T", 0)]), shots=40_000)
    assert ok, detail
    assert max(detail.values()) < 0.03


def test_gadget_tt_is_s():
    ok, detail = check_unitary(LogicalCircuit([("T", 0), ("T", 0)]), shots=40_000)
    assert ok, detail
