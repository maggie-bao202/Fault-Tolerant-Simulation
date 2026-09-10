"""Top-level entry point: logical circuit / factory -> FT report.

    run_pipeline(lc, p, code="h6",
                 zero_lvl_distill=None, cultivation=False, distillation=None,
                 protocol="processor" | "memory" | "factory") -> FTReport

The physical circuit is built with ``lightstim.ir.CircuitBuilder`` (detectors
auto-generated), noise via ``lightstim.noise.NoiseInjector``, scoring by
post-selection (:mod:`ftsim.score`).

* **processor** -- run the logical ``{H,S,T,CNOT}`` circuit; each ``T`` is
  supplied by a magic-state factory protocol (:mod:`ftsim.factory`), selected
  with ``zero_lvl_distill=`` / ``cultivation`` / ``distillation=``.
* **memory** -- a d-round memory experiment for the code.
* **factory** -- build *just* the selected magic-state protocol and report its
  yield + output infidelity (``lc`` is ignored / may be ``None``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from lightstim.noise.config import NoiseConfig
from lightstim.noise.injector import NoiseInjector

from ._compiled import Compiled
from .backends import UnsupportedGate, list_codes
from .factory import CrossCodeInject, make_magic_source
from .logical_gates import LogicalCircuit
from .processor import compile_logical_circuit, compile_memory
from .score import Reference, RunStats, reference, run
from .sim.ideal import state_of

_CHECK_INPUTS = ["0", "1", "+"]
_UNITARY_ATOL = 0.06


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
    code: str = "h6"
    protocol: str = "processor"
    magic: str = ""
    noiseless_error_floor: float = 0.0
    noiseless_yield: float = float("nan")
    unitary_check_detail: Dict[str, float] = field(default_factory=dict)
    per_observable_errors: List[int] = field(default_factory=list)

    @property
    def logical_error_rate_net(self) -> float:
        return max(self.logical_error_rate - self.noiseless_error_floor, 0.0)

    def __str__(self) -> str:
        mg = f", magic={self.magic}" if self.magic else ""
        return (
            f"FTReport(code={self.code}, protocol={self.protocol}{mg}, "
            f"n={self.n_logical}, T={self.t_count}, phys={self.physical_qubits}, "
            f"shots={self.shots}, accept={self.post_selection_rate:.4g}, "
            f"LER={self.logical_error_rate:.4g} (floor {self.noiseless_error_floor:.3g}), "
            f"unitary_ok={self.unitary_check_passed}, {self.seconds:.1f}s)"
        )


def _t_free(lc: LogicalCircuit) -> LogicalCircuit:
    return LogicalCircuit([g for g in lc.gates if g[0] not in ("T", "T_DAG")])


def _inject(circuit, p: float, mode: str):
    if p <= 0:
        return circuit
    cfg = (NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p, p_idle=p / 5)
           if mode == "idle" else
           NoiseConfig(p_1q=p, p_2q=p, p_meas=p, p_reset=p))
    return NoiseInjector.from_circuit_level(
        cfg, list(range(circuit.num_qubits))
    ).inject_noise(circuit)


def _empirical_dist(comp: Compiled, shots: int, seed: int) -> np.ndarray:
    """Kept-branch slot-0 Z output distribution from RAW measurements
    (check_unitary runs a T-free circuit -> no sbranch, QEC dets deterministic-0)."""
    n = comp.n_logical
    m = np.asarray(comp.circuit.compile_sampler(seed=seed).sample(shots), np.uint8)
    bits = np.stack(
        [m[:, cols].sum(axis=1) % 2 for cols in comp.slot0_meas_cols], axis=1
    )
    idx = (bits * (1 << np.arange(n))).sum(axis=1).astype(int)
    dist = np.bincount(idx, minlength=1 << n).astype(float)
    return dist / max(dist.sum(), 1.0)


def check_unitary(
    lc: LogicalCircuit,
    *,
    code: str = "h6",
    shots: int = 20_000,
    atol: float = _UNITARY_ATOL,
    seed: int = 7,
    code_kwargs: Optional[dict] = None,
) -> tuple[bool, Dict[str, float]]:
    """Noiseless check that the compiled circuit implements ``lc``.

    Under the Clifford proxy ``T`` acts as identity, so the reference is
    :func:`state_of` of ``lc`` with every ``T`` dropped -- exact for Clifford
    circuits; for non-Clifford circuits this validates the Clifford skeleton only.
    """
    ref_lc = _t_free(lc)
    n = lc.n_qubits
    detail: Dict[str, float] = {}
    ok = True
    if n == 0:
        return True, {"note": "empty circuit"}
    if len(ref_lc) == 0:
        return True, {"note": "Clifford skeleton is identity (T modelled as identity)"}
    tried = 0
    for ins in _CHECK_INPUTS:
        try:
            comp = compile_logical_circuit(
                ref_lc, code, input_state=ins * n, final="slot0_z",
                code_kwargs=code_kwargs,
            )
        except UnsupportedGate:
            continue
        tried += 1
        emp = _empirical_dist(comp, shots, seed)
        idl = np.abs(state_of(ref_lc, n, ins * n)) ** 2
        tvd = 0.5 * float(np.abs(emp - idl).sum())
        detail[f"tvd[{ins}]"] = tvd
        ok = ok and (tvd <= atol)
    if tried == 0:
        detail["note"] = "no preparable check input for this code"
    if lc.t_count:
        detail["note"] = "Clifford skeleton only (T modelled as identity)"
    return ok, detail


def run_pipeline(
    lc: Optional[LogicalCircuit],
    p: float,
    *,
    code: str = "h6",
    factory_code: Optional[str] = None,
    protocol: str = "processor",
    zero_lvl_distill=None,
    cultivation: bool = False,
    distillation=None,
    shots: int = 200_000,
    mode: str = "full",
    input_state: Optional[str] = None,
    seed: int = 0,
    check: bool = True,
    se_rounds_per_layer: int = 1,
    rounds: int = 3,
    code_kwargs: Optional[dict] = None,
) -> FTReport:
    t0 = time.perf_counter()
    magic = make_magic_source(
        zero_lvl_distill=zero_lvl_distill, cultivation=cultivation,
        distillation=distillation,
    )
    cross = factory_code is not None and factory_code != code
    if cross:
        if magic is None:
            raise ValueError("factory_code needs a magic-state flag (e.g. distillation=1)")
        magic = CrossCodeInject(magic)
    magic_name = magic.name if magic is not None else ""

    unitary_ok, detail = True, {}
    t_count = 0

    if protocol == "factory" or lc is None:
        if magic is None:
            raise ValueError(
                "protocol='factory' needs a magic-state protocol: pass "
                "zero_lvl_distill=True or distillation=1"
            )
        comp: Compiled = magic.build_standalone(code, code_kwargs=code_kwargs)
        n_logical = comp.n_logical
        protocol_label = "factory"

    elif protocol == "memory":
        comp = compile_memory(
            code, rounds=rounds, basis=(input_state or "Z"), code_kwargs=code_kwargs,
        )
        n_logical = comp.n_logical
        protocol_label = "memory"

    else:  # processor
        t_count = lc.t_count
        if t_count and magic is None:
            magic = make_magic_source(zero_lvl_distill=True)       # sensible default
            magic_name = magic.name
        n = lc.n_qubits
        istate = "0" * n if input_state is None else input_state
        bench = LogicalCircuit(list(lc.gates) + list(lc.inverse().gates))
        comp = compile_logical_circuit(
            bench, code, magic_source=magic,
            factory_code=(factory_code if cross else None),
            input_state=istate, se_rounds_per_layer=se_rounds_per_layer,
            code_kwargs=code_kwargs,
        )
        n_logical = n
        protocol_label = "processor"
        if check:
            unitary_ok, detail = check_unitary(lc, code=code, code_kwargs=code_kwargs)

    ref: Reference = reference(comp)
    noisy = _inject(comp.circuit, p, mode)
    stats: RunStats = run(comp, noisy, shots, seed=seed, ref=ref)

    return FTReport(
        n_qubits=comp.n_qubits,
        n_logical=n_logical,
        t_count=t_count,
        physical_qubits=comp.n_qubits,
        shots=stats.total,
        passed_shots=stats.passed,
        logical_errors=stats.errors,
        post_selection_rate=stats.post_selection_rate,
        logical_error_rate=stats.logical_error_rate,
        seconds=time.perf_counter() - t0,
        unitary_check_passed=unitary_ok,
        code=code,
        protocol=protocol_label,
        magic=magic_name,
        noiseless_error_floor=ref.error_floor,
        noiseless_yield=ref.noiseless_yield,
        unitary_check_detail=detail,
        per_observable_errors=stats.per_observable_errors,
    )


def list_protocols() -> List[str]:
    return ["processor", "memory", "factory"]


__all__ = ["FTReport", "run_pipeline", "check_unitary", "list_codes", "list_protocols"]
