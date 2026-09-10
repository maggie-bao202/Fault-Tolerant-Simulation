"""The compile artefact shared by every protocol: a clean ``stim.Circuit`` plus
the metadata :mod:`ftsim.score` needs to post-select and score it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import stim


@dataclass
class Compiled:
    circuit: stim.Circuit
    n_logical: int
    n_qubits: int
    input_state: str
    t_count: int = 0
    #: deterministic-0 noiseless -> post-select == reference (== 0)
    qec_detectors: List[int] = field(default_factory=list)
    #: random noiseless -> post-select raw == 0 (Magic-H6 no-S branch)
    sbranch_detectors: List[int] = field(default_factory=list)
    #: observable indices scored for a logical error (all, for the round-trip)
    observable_indices: List[int] = field(default_factory=list)
    #: per logical qubit, measurement columns whose XOR is its slot-0 Z parity
    #: (populated only by ``final="slot0_z"``, for ``check_unitary``)
    slot0_meas_cols: List[List[int]] = field(default_factory=list)


__all__ = ["Compiled"]
