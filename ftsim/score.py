"""Sample a compiled circuit, apply post-selection, score logical errors.

Post-selection has two flavours (mirrors the Magic-H6 recipe):

* ``qec_detectors``  -- deterministic-0 noiseless; kept iff they equal the
  noiseless reference (which is all-zero for a correct build).
* ``sbranch_detectors`` -- *random* noiseless (the no-``S`` teleportation
  branch); kept iff their raw value is 0. The noiseless reference for the QEC
  detectors and observables is read off the shots that already pass this filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from ._compiled import Compiled


@dataclass
class RunStats:
    total: int
    passed: int
    errors: int
    per_observable_errors: List[int] = field(default_factory=list)

    @property
    def post_selection_rate(self) -> float:
        return self.passed / self.total if self.total else float("nan")

    @property
    def logical_error_rate(self) -> float:
        return self.errors / self.passed if self.passed else float("nan")


@dataclass
class Reference:
    ref_obs: np.ndarray
    noiseless_yield: float
    error_floor: float = 0.0


def _sample(circuit, shots: int, seed: int):
    d, o = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )
    return np.asarray(d, np.uint8), np.asarray(o, np.uint8)


def _keep_mask(dets: np.ndarray, comp: Compiled, ref_qec: np.ndarray) -> np.ndarray:
    m = np.ones(dets.shape[0], dtype=bool)
    if comp.qec_detectors:
        m &= (dets[:, comp.qec_detectors] == ref_qec).all(axis=1)
    if comp.sbranch_detectors:
        m &= (dets[:, comp.sbranch_detectors] == 0).all(axis=1)
    return m


def reference(comp: Compiled, shots: int = 20_000, seed: int = 12345) -> Reference:
    dets, obs = _sample(comp.circuit, shots, seed)
    if comp.sbranch_detectors:
        sb_ok = (dets[:, comp.sbranch_detectors] == 0).all(axis=1)
    else:
        sb_ok = np.ones(dets.shape[0], dtype=bool)
    if sb_ok.sum() == 0:
        raise RuntimeError("noiseless reference: sbranch post-selection kept 0 shots")

    if comp.qec_detectors:
        qsurv = dets[sb_ok][:, comp.qec_detectors]
        ref_qec = (qsurv.mean(axis=0) > 0.5).astype(np.uint8)
        frac = float((qsurv == ref_qec).all(axis=1).mean())
        if frac < 0.98:
            raise RuntimeError(
                f"noiseless QEC detectors not deterministic ({frac:.3f} match) -- compile bug"
            )
        if ref_qec.any():
            raise RuntimeError("noiseless QEC detector reference is not all-zero -- compile bug")
    else:
        ref_qec = np.zeros(0, np.uint8)

    keep = sb_ok & _keep_mask(dets, comp, ref_qec)
    kobs = obs[keep]
    ref_obs = (
        (kobs.mean(axis=0) > 0.5).astype(np.uint8)
        if kobs.size else np.zeros(obs.shape[1], np.uint8)
    )
    floor = float((kobs != ref_obs).any(axis=1).mean()) if kobs.size else 0.0
    return Reference(ref_obs, float(keep.mean()), floor)


def run(
    comp: Compiled,
    noisy_circuit,
    shots: int,
    *,
    seed: int = 0,
    ref: Optional[Reference] = None,
) -> RunStats:
    if ref is None:
        ref = reference(comp)
    dets, obs = _sample(noisy_circuit, shots, seed)
    ref_qec = np.zeros(len(comp.qec_detectors), np.uint8)
    keep = _keep_mask(dets, comp, ref_qec)
    kobs = obs[keep]
    if kobs.size:
        wrong = kobs != ref.ref_obs
        err_shot = wrong.any(axis=1)
        per_obs = wrong.sum(axis=0).tolist()
        errors = int(err_shot.sum())
    else:
        per_obs = [0] * obs.shape[1]
        errors = 0
    return RunStats(int(shots), int(keep.sum()), errors, [int(x) for x in per_obs])


__all__ = ["RunStats", "Reference", "reference", "run"]
