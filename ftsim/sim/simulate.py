"""Run a compiled circuit through clifft and apply post-selection + scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import clifft

from ..processor import Compiled
from .noise import add_noise


@dataclass
class RunStats:
    total: int
    passed: int
    errors: int
    per_observable_errors: list[int]

    @property
    def post_selection_rate(self) -> float:
        return self.passed / self.total if self.total else float("nan")

    @property
    def logical_error_rate(self) -> float:
        return self.errors / self.passed if self.passed else float("nan")


@dataclass
class Reference:
    det_qec: np.ndarray      # (n_qec,) expected noiseless QEC-detector bits
    obs: np.ndarray          # (n_obs,) expected noiseless observable bits (kept branch)
    noiseless_yield: float   # fraction of noiseless shots on the kept (no-S) branch
    error_floor: float = 0.0  # noiseless kept-shot rate of obs != ref.obs (coherent
                              # gadget imperfection; subtract from noisy LER)


def _keep_mask(dets: np.ndarray, comp: Compiled, ref_det_qec: np.ndarray) -> np.ndarray:
    qec = dets[:, comp.qec_detectors] if comp.qec_detectors else dets[:, :0]
    sb = dets[:, comp.sbranch_detectors] if comp.sbranch_detectors else dets[:, :0]
    m = np.ones(dets.shape[0], dtype=bool)
    if qec.shape[1]:
        m &= (qec == ref_det_qec).all(axis=1)
    if sb.shape[1]:
        m &= (sb == 0).all(axis=1)
    return m


def reference(comp: Compiled, shots: int = 20_000, seed: int = 12345) -> Reference:
    """Noiseless reference: expected QEC-detector pattern and observable values.

    The teleport-block ("sbranch") checks are *random* on the noiseless circuit
    (they encode the distillation / no-S-branch post-selection).  Conditioned on
    them all reading 0, the magic blocks are codewords and every QEC detector
    (data-patch stabilizer comparisons) is deterministic.  So we first apply the
    raw-0 sbranch filter, then read the QEC / observable reference off the
    survivors.
    """
    prog = clifft.compile(comp.text)
    r = clifft.sample(prog, shots=shots, seed=seed)
    dets = np.asarray(r.detectors, dtype=np.uint8)
    obs = np.asarray(r.observables, dtype=np.uint8)

    if comp.sbranch_detectors:
        sb_ok = (dets[:, comp.sbranch_detectors] == 0).all(axis=1)
    else:
        sb_ok = np.ones(dets.shape[0], dtype=bool)
    if sb_ok.sum() == 0:
        raise RuntimeError("Noiseless reference: sbranch post-selection kept 0 shots.")

    qec = dets[:, comp.qec_detectors] if comp.qec_detectors else dets[:, :0]
    if qec.shape[1]:
        qsurv = qec[sb_ok]
        ref_qec = (qsurv.mean(axis=0) > 0.5).astype(np.uint8)
        det_frac = float((qsurv == ref_qec).all(axis=1).mean())
        if det_frac < 0.98:
            raise RuntimeError(
                f"Noiseless QEC detectors not deterministic ({det_frac:.3f} match) "
                "-- compile / teleport bug."
            )
    else:
        ref_qec = np.zeros(0, np.uint8)

    keep = sb_ok & _keep_qec(dets, comp, ref_qec)
    kobs = obs[keep]
    ref_obs = (kobs.mean(axis=0) > 0.5).astype(np.uint8) if kobs.size else np.zeros(comp.n_observables, np.uint8)
    floor = float((kobs != ref_obs).any(axis=1).mean()) if kobs.size else 0.0
    return Reference(ref_qec, ref_obs, float(keep.mean()), floor)


def _keep_qec(dets: np.ndarray, comp: Compiled, ref_det_qec: np.ndarray) -> np.ndarray:
    if not comp.qec_detectors:
        return np.ones(dets.shape[0], dtype=bool)
    return (dets[:, comp.qec_detectors] == ref_det_qec).all(axis=1)


def run(
    comp: Compiled,
    p: float,
    shots: int = 200_000,
    *,
    mode: str = "full",
    seed: int = 0,
    ref: Optional[Reference] = None,
) -> RunStats:
    if ref is None:
        ref = reference(comp)
    text = add_noise(comp.text, p, mode)
    prog = clifft.compile(text)
    r = clifft.sample(prog, shots=shots, seed=seed)
    dets = np.asarray(r.detectors, dtype=np.uint8)
    obs = np.asarray(r.observables, dtype=np.uint8)
    keep = _keep_mask(dets, comp, ref.det_qec)
    kobs = obs[keep]
    wrong = kobs != ref.obs
    err_shot = wrong.any(axis=1)
    per_obs = wrong.sum(axis=0).tolist() if kobs.size else [0] * comp.n_observables
    return RunStats(
        total=int(shots),
        passed=int(keep.sum()),
        errors=int(err_shot.sum()),
        per_observable_errors=[int(x) for x in per_obs],
    )
