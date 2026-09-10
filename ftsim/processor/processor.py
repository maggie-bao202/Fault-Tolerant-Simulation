"""Compile a :class:`~ftsim.logical_gates.LogicalCircuit` to one ``[[6,2,2]]`` circuit.

Every step runs through here: per-patch prep, stabilizer-extraction rounds,
transversal ``H/S/CNOT`` layers, one T-state teleport block per ``T`` (see
:mod:`ftsim.processor.t_state_teleport`), and the final destructive readout.

Output is clifft circuit text plus the metadata needed to post-select and score:

* every ``DETECTOR`` is post-selected;
* QEC detectors (stabilizer / H-check / consistency) are deterministic on the
  noiseless circuit -- kept iff they equal the noiseless reference;
* ``S``-branch detectors (one per ``T``) are *random* on the noiseless circuit --
  kept iff their raw value is 0 (post-select the no-``S`` teleportation branch).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .emit import Circ
from .t_state_teleport import t_teleport_lines
from ..logical_gates import LogicalCircuit
from ..qec.layout import Layout, build as build_layout
from ..qec.h6 import encoder_lines

_INPUT_FIXUP = {  # transversal gates to turn |++>_L (get_dist_circ) into the input state
    "+": [],
    "-": ["S", "S"],          # Z_L on both slots  -> |-->_L
    "0": ["H"],               # H_L               -> |00>_L
    "1": ["H", "X"],          # H_L then X (transversal X flips Z_L parity) -> |11>_L
}


@dataclass
class Compiled:
    text: str
    n_qubits: int
    n_logical: int
    input_state: str
    n_detectors: int
    n_observables: int
    sbranch_detectors: List[int] = field(default_factory=list)   # raw==0 post-select
    qec_detectors: List[int] = field(default_factory=list)       # ==reference post-select
    t_count: int = 0


def _encoder_lines(data: Sequence[int]) -> List[str]:
    return encoder_lines(list(data))


def compile(
    lc: LogicalCircuit,
    *,
    input_state: str | None = None,
    se_rounds_per_layer: int = 1,
    factory_se_rounds: int = 1,
) -> Compiled:
    n = lc.n_qubits
    input_state = "0" * n if input_state is None else input_state
    if len(input_state) != n or any(ch not in _INPUT_FIXUP for ch in input_state):
        raise ValueError(f"input_state must be length {n} over {set(_INPUT_FIXUP)}.")

    lay: Layout = build_layout(n, lc.t_count)
    c = Circ()

    all_data = [q for p in lay.data_patches for q in p.data]
    c.gate("R", all_data)

    # -- prep: |++>_L per patch, then per-qubit input fixup -----------------
    for q, p in enumerate(lay.data_patches):
        for ln in _encoder_lines(p.data):
            c.raw(ln)
        for g in _INPUT_FIXUP[input_state[q]]:
            c.gate(g, p.data)

    qec_dets: List[int] = []
    sbranch_dets: List[int] = []

    def emit_se(patches, prev: Dict[str, List[int]] | None) -> Dict[str, List[int]]:
        anc = c.se_round(patches)
        for name, ids in anc.items():
            for i, mid in enumerate(ids):
                if prev is None:
                    c.detector([mid])
                else:
                    c.detector([mid, prev[name][i]])
                qec_dets.append(c._det_count() - 1)
        return anc

    prev = emit_se(lay.data_patches, None)

    # -- gate layers -------------------------------------------------------
    kmagic = 0
    for layer in lc.layers():
        for g in layer:
            if g[0] == "CNOT": #transversal CX on all physical qubits of the two logical qubits
                pc, pt = lay.data(g[1]), lay.data(g[2])
                c.gate("CX", [x for pair in zip(pc.data, pt.data) for x in pair])
            elif g[0] in ("T", "T_DAG"):
                mp = lay.magic(kmagic)
                kmagic += 1
                dp = lay.data(g[1])
                lines, man = t_teleport_lines(
                    dp.data, mp.data, mp.xanc, mp.zanc, mp.bare,
                    se_rounds=factory_se_rounds, dag=(g[0] == "T_DAG"),
                )
                base = c.n_meas
                c.raw_block(lines, int(man["meas"]))
                # every teleport-block check (factory stabilizers, magic codeword
                # consistency, S-branch bit) is random on the noiseless circuit,
                # so post-select all of them on raw 0.
                for grp in man["postselect"]:            # list[tuple[int,...]]
                    c.detector([base + r for r in grp])
                    sbranch_dets.append(c._det_count() - 1)
            else:  # H / S / S_DAG / X  (transversal 1q)
                c.gate(g[0], lay.data(g[1]).data)
        prev = emit_se(lay.data_patches, prev)

    # -- final destructive readout (Z) + observables + consistency --------
    readout: Dict[str, List[int]] = {}
    for p in lay.data_patches:
        readout[p.name] = c.measure(p.data, "Z")
    for q, p in enumerate(lay.data_patches):
        ids = readout[p.name]
        c.observable([ids[j] for j in (0, 2, 4)], idx=q)   # slot-0 Z_L parity
    for p in lay.data_patches:
        ids = readout[p.name]
        for i, supp in enumerate(((0, 1, 2, 3), (2, 3, 4, 5))):
            c.detector([ids[j] for j in supp] + [prev[p.name][2 + i]])
            qec_dets.append(c._det_count() - 1)

    return Compiled(
        text=c.text(),
        n_qubits=len(lay.all_qubits),
        n_logical=n,
        input_state=input_state,
        n_detectors=c._det_count(),
        n_observables=n,
        sbranch_detectors=sbranch_dets,
        qec_detectors=qec_dets,
        t_count=lc.t_count,
    )
