"""Lower a :class:`~ftsim.ppm.ir.PBCCircuit` to an encoded ``stim.Circuit``.

Only the **Clifford part** is lowered:

* each Clifford Pauli Product Rotation (``pi/4`` / ``pi/2``) -> transversal
  ``{H, S, CNOT}`` via :func:`_apply_clifford_ppr` (weight-k ``pi/4``: basis-change
  every factor onto ``Z``, CNOT-ladder the word onto a pivot patch, transversal
  ``S`` / ``S_dag``, uncompute).  Consecutive commuting PPRs are grouped into
  layers (:func:`_commuting_layers`) with one SE round per layer -- the PBC
  analogue of ``LogicalCircuit.layers()``;
* each terminal measurement: a bare single-qubit ``Z`` is covered by the final
  data readout; anything else goes through the ancilla-assisted encoded
  Pauli-product-measurement primitive :func:`_encoded_ppm` (the
  :func:`ftsim.factory._teleport` transversal pattern: fresh ``|0>_L`` ancilla
  patch, one transversal CNOT per factor, destructive ``Z_L`` readout, one
  ``OBSERVABLE_INCLUDE``);
* the first non-Clifford ``pi/8`` rotation hands off to
  :func:`ftsim.ppm.magic.magic_injection`, which raises ``NotImplementedError``.

``ftsim.pipeline.run_pipeline`` always compiles the round-trip
``bench = lc + lc.inverse()``, so the lowered Clifford PPR sequence composes to
the identity and every emitted detector / observable is deterministic
noiselessly -- the invariant :mod:`ftsim.score` requires.

Supported data codes: ``h6`` and ``steane`` (they expose transversal ``H`` / ``S``
/ ``CNOT``).  Everything else raises :class:`~ftsim.ppm._errors.PPMUnsupported`.
"""

from __future__ import annotations

from typing import List, Optional

import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.tracker import SyndromeTracker

from .._compiled import Compiled
from ..backends import CodeSpec, Driver
from ..backends import get as get_code
from ._errors import PPMUnsupported
from .ir import PPM, PPR, PBCCircuit
from .magic import magic_injection

PPM_SUPPORTED_CODES = ("h6", "steane")

_REQUIRED_OPS = (
    "transversal_hadamard",
    "transversal_s",
    "transversal_s_dag",
    "transversal_cnot",
    "transversal_x",
    "transversal_z",
)


def _spec(code) -> CodeSpec:
    return code if isinstance(code, CodeSpec) else get_code(code)


def _check_code_supported(spec: CodeSpec) -> None:
    if spec.name not in PPM_SUPPORTED_CODES:
        raise PPMUnsupported(
            f"code {spec.name!r}: the PPM frontend needs transversal H/S/CNOT "
            f"(supported: {', '.join(PPM_SUPPORTED_CODES)}); use frontend='gate'"
        )
    ops = spec.op_set
    missing = [m for m in _REQUIRED_OPS if not callable(getattr(ops, m, None))]
    if ops is None or missing:
        raise PPMUnsupported(
            f"code {spec.name!r}: op-set is missing {missing or ['op_set']}; "
            "use frontend='gate'"
        )


# --------------------------------------------------------------------------
# the transversal encoded Pauli-product-measurement primitive
# --------------------------------------------------------------------------

def _basis_change_to_z(ops, builder, patch, pauli: str) -> None:
    """Conjugate ``patch`` so its logical ``pauli`` factor becomes ``Z``."""
    if pauli == "Z":
        return
    if pauli == "X":
        ops.transversal_hadamard(builder, patch)
    elif pauli == "Y":
        ops.transversal_s_dag(builder, patch)     # (H S_dag) Y (S H) = Z
        ops.transversal_hadamard(builder, patch)
    else:  # pragma: no cover - PPR/PPM normalise to X/Y/Z
        raise PPMUnsupported(f"unsupported Pauli factor {pauli!r}")


def _basis_change_from_z(ops, builder, patch, pauli: str) -> None:
    """Undo :func:`_basis_change_to_z`."""
    if pauli == "Z":
        return
    if pauli == "X":
        ops.transversal_hadamard(builder, patch)
    elif pauli == "Y":
        ops.transversal_hadamard(builder, patch)
        ops.transversal_s(builder, patch)


def _encoded_ppm(driver: Driver, builder: CircuitBuilder, ppm: PPM,
                 *, ancilla: str) -> int:
    """Measure the logical Pauli product ``ppm`` into a fresh ``|0>_L`` ancilla
    patch by transversal CNOT + destructive ``Z_L`` readout.  Emits one
    ``OBSERVABLE_INCLUDE`` and returns its index."""
    ops = driver.spec.op_set
    targets = list(zip(ppm.qubits, ppm.paulis))

    for qi, p in targets:
        _basis_change_to_z(ops, builder, driver.patch(driver.data_names[qi]), p)

    for qi, _ in targets:
        ops.transversal_cnot(
            builder,
            driver.patch(driver.data_names[qi]),   # control = data patch
            driver.patch(ancilla),                 # target  = ancilla patch
        )

    a_data = driver.data(ancilla)
    builder.apply_mid_data_readout({q: "Z" for q in a_data})

    c = builder.circuit
    n_a = len(a_data)
    support = set(driver.logical_z_support(ancilla))
    recs = [stim.target_rec(-n_a + i) for i, q in enumerate(a_data) if q in support]
    oidx = builder.tracker.allocate_observable()
    c.append("OBSERVABLE_INCLUDE", recs, oidx)

    for qi, p in targets:
        _basis_change_from_z(ops, builder, driver.patch(driver.data_names[qi]), p)

    return oidx


# --------------------------------------------------------------------------
# Clifford PPR synthesis + SE-round layering
# --------------------------------------------------------------------------

def _pprs_commute(a: PPR, b: PPR) -> bool:
    """Whether two Pauli-product rotations commute (their Pauli words do)."""
    da, db = dict(zip(a.qubits, a.paulis)), dict(zip(b.qubits, b.paulis))
    anti = sum(1 for q in da.keys() & db.keys() if da[q] != db[q])
    return anti % 2 == 0


def _commuting_layers(pprs) -> "list[list[PPR]]":
    """Greedily group consecutive PPRs into layers of mutually commuting
    rotations -- the PBC analogue of ``LogicalCircuit.layers()``.  One SE round
    per layer (instead of per PPR) keeps the fault-tolerant SE cadence close to
    the gate path's while cutting round count several-fold."""
    layers: list[list[PPR]] = []
    for ppr in pprs:
        if layers and all(_pprs_commute(ppr, x) for x in layers[-1]):
            layers[-1].append(ppr)
        else:
            layers.append([ppr])
    return layers


def _apply_clifford_ppr(driver: Driver, builder: CircuitBuilder, ppr: PPR) -> None:
    """Apply a Clifford Pauli Product Rotation ``exp(-i * sign * angle * P)``
    (``angle`` in ``{pi/2, pi/4}``) as transversal logical gates.

    ``pi/4`` weight-k: conjugate every factor onto ``Z``, run a CNOT ladder into
    a pivot patch so its ``Z_L`` carries the whole word, apply transversal
    ``S`` / ``S_dag`` on the pivot, then uncompute -- ``{H, S, CNOT}`` only.
    ``pi/2`` = ``-i * sign * P`` up to global phase -> transversal Paulis.
    """
    ops = driver.spec.op_set
    factors = list(zip(ppr.qubits, ppr.paulis))

    if ppr.angle == "pi/2":
        for q, p in factors:
            patch = driver.patch(driver.data_names[q])
            if p == "X":
                ops.transversal_x(builder, patch)
            elif p == "Z":
                ops.transversal_z(builder, patch)
            elif p == "Y":
                ops.transversal_x(builder, patch)
                ops.transversal_z(builder, patch)          # Y up to global phase
            else:  # pragma: no cover
                raise PPMUnsupported(f"pi/2 PPR on Pauli {p!r} unsupported")
        return

    # pi/4
    dag = ppr.sign < 0
    for q, p in factors:
        _basis_change_to_z(ops, builder, driver.patch(driver.data_names[q]), p)

    pivot = driver.patch(driver.data_names[factors[0][0]])
    rest = [driver.patch(driver.data_names[q]) for q, _ in factors[1:]]
    for patch in rest:
        ops.transversal_cnot(builder, patch, pivot)         # Z_pivot <- Z_pivot * Z_patch
    (ops.transversal_s_dag if dag else ops.transversal_s)(builder, pivot)
    for patch in reversed(rest):
        ops.transversal_cnot(builder, patch, pivot)

    for q, p in factors:
        _basis_change_from_z(ops, builder, driver.patch(driver.data_names[q]), p)


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------

def lower_pbc(
    pbc: PBCCircuit,
    code,
    *,
    input_state: str,
    se_rounds_per_layer: int = 1,
    t_count: int = 0,
    code_kwargs: Optional[dict] = None,
) -> Compiled:
    """Lower ``pbc`` (its rotations already the round-trip's) to a
    :class:`~ftsim._compiled.Compiled`.  Catalyst-free: drive it directly with a
    hand-built :class:`PBCCircuit` in tests."""
    spec = _spec(code)
    _check_code_supported(spec)

    n = pbc.n_qubits
    if len(input_state) != n:
        raise ValueError(f"input_state must have length {n}; got {input_state!r}")
    if any(c not in "01" for c in input_state):
        raise PPMUnsupported(
            "PPM frontend v1 supports Z-basis ('0'/'1') input states only; "
            f"got {input_state!r} (the net-identity round-trip keeps Z-product "
            "PPMs deterministic)"
        )

    cliff_pprs, pi8_pprs = pbc.partition()
    if pi8_pprs:
        # empty stub: raises NotImplementedError, propagates out of run_pipeline
        magic_injection(driver=None, builder=None, ppr=pi8_pprs[0])

    # a terminal PPM that is just single-qubit Z is covered by the final data
    # readout; anything else needs an ancilla-assisted encoded measurement.
    def _trivial(m: PPM) -> bool:
        return m.paulis == ("Z",)

    nontrivial = [m for m in pbc.measurements if not _trivial(m)]
    driver = Driver(spec, n, ppm_ancillas=len(nontrivial), **(code_kwargs or {}))
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
    for name in driver.ancilla_names:
        driver.prepare(builder, name, "0")                        # |0>_L ancilla

    se = driver.extraction_block()
    builder.apply_syndrome_extraction(se.circuit, rounds=1)       # baseline
    qec_dets: List[int] = list(range(builder.circuit.num_detectors))

    # cliff_pprs is program-ordered and (past the pi/8 guard above) is the whole
    # rotation list; when magic injection lands, this loop must interleave the
    # pi/8 rotations at their program positions rather than batch around them.
    for layer in _commuting_layers(cliff_pprs):
        for ppr in layer:
            _apply_clifford_ppr(driver, builder, ppr)
        lo = builder.circuit.num_detectors
        builder.apply_syndrome_extraction(se.circuit, rounds=se_rounds_per_layer)
        qec_dets.extend(range(lo, builder.circuit.num_detectors))

    obs_idx: List[int] = []
    for k, ppm in enumerate(nontrivial):
        obs_idx.append(
            _encoded_ppm(driver, builder, ppm, ancilla=driver.ancilla_names[k])
        )
        lo = builder.circuit.num_detectors
        builder.apply_syndrome_extraction(se.circuit, rounds=se_rounds_per_layer)
        qec_dets.extend(range(lo, builder.circuit.num_detectors))

    lo = builder.circuit.num_detectors
    # mirror ftsim.processor.compile_logical_circuit: read out every data index
    # (the ancilla data qubits already left the active set via
    # apply_mid_data_readout, exactly as the magic patches do in the gate path).
    builder.apply_data_readout(final_measurements={q: "Z" for q in all_data})
    qec_dets.extend(range(lo, builder.circuit.num_detectors))
    obs_idx.extend(
        i for i in range(builder.circuit.num_observables) if i not in obs_idx
    )

    circ = builder.circuit
    return Compiled(
        circuit=circ,
        n_logical=n,
        n_qubits=circ.num_qubits,
        input_state=input_state,
        t_count=t_count,
        qec_detectors=sorted(set(qec_dets)),
        sbranch_detectors=[],
        observable_indices=sorted(set(obs_idx)),
        slot0_meas_cols=[],
    )


def compile_ppm(
    lc,
    code,
    *,
    input_state: Optional[str] = None,
    se_rounds_per_layer: int = 1,
    avoid_y: bool = True,
    device: str = "lightning.qubit",
    code_kwargs: Optional[dict] = None,
) -> Compiled:
    """``LogicalCircuit`` (already the round-trip ``bench``) -> PBC IR via
    Catalyst -> encoded :class:`~ftsim._compiled.Compiled`.

    Raises :class:`ImportError` if ``ftsim[ppm]`` is not installed,
    :class:`~ftsim.ppm._errors.PPMUnsupported` for an unsupported code / IR
    shape, and :class:`NotImplementedError` if the circuit contains a
    non-Clifford (``pi/8``) rotation.
    """
    spec = _spec(code)
    _check_code_supported(spec)                                   # fail fast

    from .catalyst_frontend import logical_circuit_to_mlir
    from .mlir_parse import parse_pbc

    mlir = logical_circuit_to_mlir(lc, avoid_y=avoid_y, device=device)
    pbc = parse_pbc(mlir, lc.n_qubits)

    istate = "0" * lc.n_qubits if input_state is None else input_state
    return lower_pbc(
        pbc, spec,
        input_state=istate,
        se_rounds_per_layer=se_rounds_per_layer,
        t_count=lc.t_count,
        code_kwargs=code_kwargs,
    )


__all__ = ["compile_ppm", "lower_pbc", "PPM_SUPPORTED_CODES", "PPMUnsupported"]
