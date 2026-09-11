"""``LogicalCircuit`` -> Catalyst PPM passes -> ``pbc``-dialect MLIR text.

The only module in :mod:`ftsim.ppm` that needs the optional ``ftsim[ppm]``
dependency (PennyLane + ``pennylane-catalyst``).  Imported lazily by
:func:`ftsim.ppm.encoded.compile_ppm`; a missing dependency raises
:class:`ImportError` with an install hint.

Pipeline: ``to_ppr -> commute_ppr -> merge_ppr_ppm`` -- the Clifford-preserving
prefix of :func:`catalyst.passes.ppm_compilation`.  Stopping before
``ppr_to_ppm`` keeps the ``pi/8`` Pauli Product Rotations intact (for
:func:`ftsim.ppm.magic.magic_injection`) and avoids the auxiliary-qubit /
Y-measurement / feed-forward expansion that full ``ppm_compilation`` emits.

For a Clifford circuit the prefix yields a sequence of Clifford ``pbc.ppr``
(``pi/4`` / ``pi/2``) with the terminal ``qml.expval(qml.Z(i))`` left as
``quantum.namedobs`` -- so :mod:`ftsim.ppm.encoded` re-synthesises each Clifford
PPR from transversal ``{H, S, CNOT}`` and reads out ``Z`` at the end.
"""

from __future__ import annotations

import glob
import os
import shutil
import tempfile
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from ..logical_gates import LogicalCircuit

#: Catalyst version the MLIR grammar in :mod:`ftsim.ppm.mlir_parse` was written
#: against.
TESTED_CATALYST = "0.15"

_INSTALL_HINT = (
    "the PPM frontend (frontend='ppm') needs PennyLane + Catalyst; install with:\n"
    '    pip install -e ".[ppm]"'
)

_STAGE = "QuantumCompilationStage"


def require_catalyst():
    """Import PennyLane + Catalyst or raise :class:`ImportError` with a hint.

    Returns ``(pennylane, catalyst.passes, catalyst.debug.get_compilation_stage)``.
    """
    try:
        import pennylane as qml
        import catalyst  # noqa: F401
        from catalyst import passes as catalyst_passes
        from catalyst.debug import get_compilation_stage

        catalyst_passes.to_ppr  # noqa: B018
        catalyst_passes.commute_ppr  # noqa: B018
        catalyst_passes.merge_ppr_ppm  # noqa: B018
    except Exception as exc:  # ImportError or a Catalyst build/runtime error
        raise ImportError(_INSTALL_HINT) from exc
    return qml, catalyst_passes, get_compilation_stage


_GATE_EMIT = {
    "H": lambda qml, q: qml.Hadamard(wires=q),
    "S": lambda qml, q: qml.S(wires=q),
    "S_DAG": lambda qml, q: qml.adjoint(qml.S)(wires=q),
    "T": lambda qml, q: qml.T(wires=q),
    "T_DAG": lambda qml, q: qml.adjoint(qml.T)(wires=q),
    "X": lambda qml, q: qml.PauliX(wires=q),
}


def _retrieve_mlir(compiled, get_compilation_stage, workdir: Optional[str]) -> str:
    """The post-``to_ppr`` MLIR text (contains ``pbc.ppr`` / ``pbc.ppm``)."""
    candidates = []
    try:
        s = get_compilation_stage(compiled, _STAGE)
        if isinstance(s, str) and s:
            candidates.append(s)
    except Exception:
        pass
    for attr in ("mlir_opt", "mlir"):
        try:
            v = getattr(compiled, attr)
        except Exception:
            continue
        if isinstance(v, str) and v:
            candidates.append(v)
    if workdir and os.path.isdir(workdir):
        for path in sorted(glob.glob(os.path.join(workdir, "**", "*.mlir"),
                                     recursive=True)):
            try:
                candidates.append(open(path, "r", encoding="utf-8").read())
            except OSError:
                pass

    for text in candidates:
        if "pbc.ppr" in text or "pbc.ppm" in text:
            return text
    if candidates:
        return max(candidates, key=len)
    raise RuntimeError(
        "ftsim.ppm.catalyst_frontend: could not obtain post-transform MLIR "
        f"(tried get_compilation_stage({_STAGE!r}), .mlir_opt, .mlir, "
        f"keep_intermediate). Installed Catalyst tested against {TESTED_CATALYST}."
    )


def logical_circuit_to_mlir(
    lc: "LogicalCircuit",
    *,
    avoid_y: bool = True,
    device: str = "lightning.qubit",
    keep_dir: Optional[str] = None,
) -> str:
    """Compile ``lc`` through ``to_ppr -> commute_ppr -> merge_ppr_ppm`` and
    return the emitted ``pbc``-dialect MLIR text.

    ``avoid_y`` is advisory (the Y-avoidance knob lives on the skipped
    ``ppr_to_ppm`` step); the downstream lowering rejects Y factors explicitly.
    """
    qml, catalyst_passes, get_compilation_stage = require_catalyst()
    from catalyst import qjit

    n = lc.n_qubits
    if n == 0:
        raise ValueError("logical_circuit_to_mlir: empty circuit")

    dev = qml.device(device, wires=n)
    gates = list(lc.gates)

    @qml.qnode(dev)
    def _base():
        for g in gates:
            if g[0] == "CNOT":
                qml.CNOT(wires=[g[1], g[2]])
            else:
                _GATE_EMIT[g[0]](qml, g[1])
        return [qml.expval(qml.Z(i)) for i in range(n)]

    transformed = catalyst_passes.merge_ppr_ppm(
        catalyst_passes.commute_ppr(catalyst_passes.to_ppr(_base))
    )

    # Catalyst's keep_intermediate + get_compilation_stage read/write files in
    # the CWD; run the compile inside a throwaway directory so the repo stays
    # clean, then recover the MLIR text before deleting it.
    fn_name = getattr(transformed, "__name__", "_base")
    tmp = keep_dir or tempfile.mkdtemp(prefix="ftsim_ppm_")
    cwd0 = os.getcwd()
    try:
        os.chdir(tmp)
        compiled = qjit(transformed, keep_intermediate=True, capture=True)
        for attr in ("mlir",):  # force AOT lowering
            try:
                getattr(compiled, attr)
            except Exception:
                pass
        return _retrieve_mlir(
            compiled, get_compilation_stage, os.path.join(tmp, fn_name)
        )
    finally:
        os.chdir(cwd0)
        if keep_dir is None:
            shutil.rmtree(tmp, ignore_errors=True)


__all__ = ["logical_circuit_to_mlir", "require_catalyst", "TESTED_CATALYST"]
