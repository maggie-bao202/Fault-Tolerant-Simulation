"""The **processor** protocol: run FT logical operations on encoded data.

    initialize -> encode -> {gate layer ; SE round} per depth layer -> readout

Each Clifford gate is dispatched to a LightStim ``LogicalOpSet`` method
(``CodeSpec.gate_methods`` names which; ``lightstim.ir.LogicalExecutor`` routes
by patch type). Each ``T`` / ``T_DAG`` is teleported through a magic patch
supplied by a :class:`ftsim.factory.MagicProtocol` -- pick it with
``run_pipeline``'s ``zero_lvl_distill=`` / ``distillation=`` flags.

:func:`compile_memory` is the trivial companion: a d-round memory experiment,
delegated straight to ``lightstim.protocols.MemoryExperiment``.
"""

from __future__ import annotations

from typing import Optional

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.logical_executor import LogicalExecutor
from lightstim.ir.tracker import SyndromeTracker
from lightstim.protocols import MemoryExperiment

from ._compiled import Compiled
from .backends import CodeSpec, Driver, UnsupportedGate
from .backends import get as get_code
from .logical_gates import LogicalCircuit


def _spec(code) -> CodeSpec:
    return code if isinstance(code, CodeSpec) else get_code(code)


class _GateRunner:
    """Dispatch a LogicalCircuit gate to the code's LightStim op-set method."""

    def __init__(self, driver: Driver, builder: CircuitBuilder):
        self.driver = driver
        self.executor = LogicalExecutor(builder)
        self._registered: set = set()

    def apply(self, gate, patch_names) -> None:
        name = gate[0]
        handler = self.driver.spec.gate_methods.get(name)
        if handler is None:
            raise UnsupportedGate(f"{self.driver.spec.name}: gate {name!r}")
        patches = [self.driver.patch(pn) for pn in patch_names]
        if callable(handler):                         # e.g. an _unsupported marker
            handler(self.driver, self.executor.builder, *patches)
            return
        pt = type(patches[0])
        if pt not in self._registered:
            self.executor.register_op_set(pt, self.driver.spec.op_set)
            self._registered.add(pt)
        self.executor.apply_logical_operation(handler, patches)


# --------------------------------------------------------------------------
# memory  ->  lightstim.protocols.MemoryExperiment
# --------------------------------------------------------------------------

def compile_memory(code, *, rounds: int = 3, basis: str = "Z",
                   code_kwargs: dict | None = None) -> Compiled:
    spec = _spec(code)
    exp = MemoryExperiment(
        qec_patch=spec.patch_factory(**(code_kwargs or {})),
        extraction_block_class=spec.extraction_block,
        rounds=rounds,
        basis=basis,
        noise_params=None,
    )
    circ = exp.build()
    return Compiled(
        circuit=circ,
        n_logical=int(exp.system.num_logicals),
        n_qubits=circ.num_qubits,
        input_state=basis,
        qec_detectors=list(range(circ.num_detectors)),
        observable_indices=list(range(circ.num_observables)),
    )


# --------------------------------------------------------------------------
# logical_circuit
# --------------------------------------------------------------------------

def compile_logical_circuit(
    lc: LogicalCircuit,
    code,
    *,
    magic_source=None,
    factory_code=None,
    input_state: str | None = None,
    se_rounds_per_layer: int = 1,
    final: str = "auto",
    code_kwargs: dict | None = None,
) -> Compiled:
    spec = _spec(code)
    factory_spec = _spec(factory_code) if factory_code is not None else None
    n = lc.n_qubits
    input_state = "0" * n if input_state is None else input_state
    if len(input_state) != n:
        raise ValueError(f"input_state must have length {n}")

    n_t = lc.t_count
    if n_t and magic_source is None:
        raise ValueError(
            "circuit has T gates but no magic-state protocol selected; pass "
            "zero_lvl_distill=True (or distillation=1) to run_pipeline"
        )

    driver = Driver(spec, n, magic=n_t, factory_spec=factory_spec, **(code_kwargs or {}))
    system = driver.system

    tracker = SyndromeTracker(
        num_qubits=system.num_qubits,
        expected_num_logicals=system.num_logicals,
    )
    builder = CircuitBuilder(tracker=tracker, system_config=system, if_detector=True)
    builder.write_coordinates()

    all_data = sorted(system.data_indices)
    builder.initialize({q: "Z" for q in all_data}, n=system.num_qubits)

    for q, name in enumerate(driver.data_names):
        driver.prepare(builder, name, input_state[q])

    se = driver.extraction_block()
    builder.apply_syndrome_extraction(se.circuit, rounds=1)          # baseline

    qec_dets: list[int] = list(range(builder.circuit.num_detectors))
    sbranch_dets: list[int] = []
    gates = _GateRunner(driver, builder)

    kmagic = 0
    for layer in lc.layers():
        for g in layer:
            if g[0] in ("T", "T_DAG"):
                added = magic_source.gadget(
                    driver, builder, driver.data_names[g[1]], kmagic,
                    dag=(g[0] == "T_DAG"),
                )
                kmagic += 1
                qec_dets.extend(added["qec"])
                sbranch_dets.extend(added["sbranch"])
            else:
                gates.apply(g, [driver.data_names[i] for i in g[1:]])
        lo = builder.circuit.num_detectors
        builder.apply_syndrome_extraction(se.circuit, rounds=se_rounds_per_layer)
        qec_dets.extend(range(lo, builder.circuit.num_detectors))

    lo = builder.circuit.num_detectors
    slot0_cols: list[list[int]] = []
    if final == "slot0_z":
        c = builder.circuit
        m0 = c.num_measurements
        c.append("M", all_data)
        col = {q: m0 + i for i, q in enumerate(all_data)}
        for name in driver.data_names:
            slot0_cols.append([col[q] for q in driver.logical_z_support(name)])
        obs_idx: list[int] = []
    else:
        builder.apply_data_readout(final_measurements={q: "Z" for q in all_data})
        obs_idx = list(range(builder.circuit.num_observables))
    qec_dets.extend(range(lo, builder.circuit.num_detectors))

    circ = builder.circuit
    qec_dets = sorted(set(qec_dets) - set(sbranch_dets))
    return Compiled(
        circuit=circ,
        n_logical=n,
        n_qubits=circ.num_qubits,
        input_state=input_state,
        t_count=lc.t_count,
        qec_detectors=qec_dets,
        sbranch_detectors=sorted(set(sbranch_dets)),
        observable_indices=obs_idx,
        slot0_meas_cols=slot0_cols,
    )


__all__ = ["compile_logical_circuit", "compile_memory"]
