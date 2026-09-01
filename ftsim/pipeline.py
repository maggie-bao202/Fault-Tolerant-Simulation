"""Top-level entry point: logical circuit -> FT report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import time

import numpy as np
import clifft

from .compiler.lower import Compiled, compile as compile_lc
from .logical_gates import LogicalCircuit
from .sim.ideal import state_of
from .sim.simulate import Reference, RunStats, reference, run

# input states used for the noiseless logical-unitary check
_CHECK_INPUTS = ["0", "1", "+"]


@dataclass
class FTReport:
    n_qubits: int
    n_logical: int
    t_count: int
    physical_qubits: int
    shots: int
    passed_shots: int
    logical_errors: int
    post_selection_rate: float
    logical_error_rate: float
    seconds: float
    unitary_check_passed: bool
    noiseless_error_floor: float = 0.0
    noiseless_yield: float = float("nan")
    unitary_check_detail: Dict[str, float] = field(default_factory=dict)
    per_observable_errors: List[int] = field(default_factory=list)

    @property
    def logical_error_rate_net(self) -> float:
        """LER with the coherent noiseless floor subtracted (>= 0)."""
        return max(self.logical_error_rate - self.noiseless_error_floor, 0.0)

    def __str__(self) -> str:
        return (
            f"FTReport(n={self.n_logical}, T={self.t_count}, "
            f"phys_qubits={self.physical_qubits}, shots={self.shots}, "
            f"accept={self.post_selection_rate:.4g}, "
            f"LER={self.logical_error_rate:.4g} (floor {self.noiseless_error_floor:.3g}), "
            f"unitary_ok={self.unitary_check_passed}, {self.seconds:.1f}s)"
        )


def _empirical_out_dist(comp: Compiled, ref: Reference, shots: int, seed: int) -> np.ndarray:
    """Kept-branch computational-basis output distribution (Z readout)."""
    prog = clifft.compile(comp.text)
    r = clifft.sample(prog, shots=shots, seed=seed)
    import numpy as _np
    dets = _np.asarray(r.detectors, dtype=_np.uint8)
    obs = _np.asarray(r.observables, dtype=_np.uint8)
    from .sim.simulate import _keep_mask
    keep = _keep_mask(dets, comp, ref.det_qec)
    kobs = obs[keep]
    n = comp.n_logical
    idx = (kobs * (1 << _np.arange(n))).sum(axis=1).astype(int)
    dist = _np.bincount(idx, minlength=1 << n).astype(float)
    return dist / max(dist.sum(), 1)


def _ideal_out_dist(lc: LogicalCircuit, n: int, input_state: str) -> np.ndarray:
    psi = state_of(lc, n, input_state)
    return np.abs(psi) ** 2


# Bare [[6,2,2]] transversal physical T is logical T only up to a small coherent
# rotation (the code has no exact transversal T), so a T conjugated by H carries a
# residual ~0.07 TVD.  Clifford-only and non-H-conjugated-T circuits are exact.
_UNITARY_ATOL = 0.12


def check_unitary(lc: LogicalCircuit, *, shots: int = 20_000, atol: float = _UNITARY_ATOL,
                  seed: int = 7) -> tuple[bool, Dict[str, float]]:
    n = lc.n_qubits
    detail: Dict[str, float] = {}
    ok = True
    for ins in _CHECK_INPUTS:
        comp = compile_lc(lc, input_state=ins * n)
        ref = reference(comp)
        emp = _empirical_out_dist(comp, ref, shots, seed)
        idl = _ideal_out_dist(lc, n, ins * n)
        tvd = 0.5 * float(np.abs(emp - idl).sum())
        detail[f"tvd[{ins}]"] = tvd
        ok = ok and (tvd <= atol)
    return ok, detail


def run_pipeline(
    lc: LogicalCircuit,
    p: float,
    *,
    shots: int = 200_000,
    mode: str = "full",
    input_state: Optional[str] = None,
    seed: int = 0,
    check: bool = True,
    factory_se_rounds: int = 1,
) -> FTReport:
    t0 = time.perf_counter()
    n = lc.n_qubits
    input_state = "0" * n if input_state is None else input_state

    # LER is measured on the round-trip lc ; lc.inverse() so the logical output is
    # deterministic (= input) and a per-shot error rate is well defined.
    bench_lc = LogicalCircuit(list(lc.gates) + list(lc.inverse().gates))
    comp: Compiled = compile_lc(bench_lc, input_state=input_state,
                                factory_se_rounds=factory_se_rounds)
    ref = reference(comp)

    unitary_ok, detail = (True, {})
    if check:
        unitary_ok, detail = check_unitary(lc) #implementation check

    stats: RunStats = run(comp, p, shots, mode=mode, seed=seed, ref=ref)

    return FTReport(
        n_qubits=comp.n_qubits,
        n_logical=n,
        t_count=lc.t_count,
        physical_qubits=comp.n_qubits,
        shots=stats.total,
        passed_shots=stats.passed,
        logical_errors=stats.errors,
        post_selection_rate=stats.post_selection_rate,
        logical_error_rate=stats.logical_error_rate,
        seconds=time.perf_counter() - t0,
        unitary_check_passed=unitary_ok,
        noiseless_error_floor=ref.error_floor,
        noiseless_yield=ref.noiseless_yield,
        unitary_check_detail=detail,
        per_observable_errors=stats.per_observable_errors,
    )
