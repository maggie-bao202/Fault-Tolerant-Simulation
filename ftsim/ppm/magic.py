"""Non-Clifford ``pi/8`` PPR handling -- the magic-state seam.

For now this is an **empty stub**: a Pauli-based-computation circuit with any
``pi/8`` Pauli Product Rotation cannot be lowered to stim, because that rotation
needs a distilled ``|T>_L`` magic resource gate-teleported onto the data
patch(es).

The eventual implementation will:

* borrow / distil a ``|T>_L`` patch (reuse :mod:`ftsim.factory`),
* gate-teleport the ``pi/8`` PPR onto its target patch(es) via a transversal
  CNOT + a Pauli-product correction conditioned on the teleportation outcome,
* emit the acceptance detector(s) so :mod:`ftsim.score` can post-select.

Until then, :func:`ftsim.ppm.encoded.lower_pbc` calls :func:`magic_injection`
on the first ``pi/8`` PPR it sees and the ``NotImplementedError`` propagates all
the way out of ``run_pipeline(..., frontend="ppm")``.
"""

from __future__ import annotations

from typing import Optional

from .ir import PPR


def magic_injection(
    driver=None,
    builder=None,
    ppr: Optional[PPR] = None,
    *,
    k: Optional[int] = None,
) -> None:
    """Placeholder for ``pi/8`` magic-state creation + injection.

    Always raises :class:`NotImplementedError`.  ``driver`` / ``builder`` / ``k``
    are accepted so the eventual real implementation is a drop-in.
    """
    detail = ""
    if ppr is not None:
        detail = f" (paulis={ppr.paulis} qubits={ppr.qubits} sign={ppr.sign})"
    raise NotImplementedError(
        "PPM frontend: non-Clifford pi/8 PPR encountered" + detail + "; "
        "magic-state injection is not implemented yet. Use frontend='gate' for "
        "T circuits (Magic-H6 Clifford proxy), or run a Clifford circuit with "
        "frontend='ppm'."
    )


__all__ = ["magic_injection"]
