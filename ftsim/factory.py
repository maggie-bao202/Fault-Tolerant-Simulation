"""Magic-state **factory** protocols -- how a code produces a ``|T>``-type resource.

A :class:`MagicProtocol` is a *protocol* (an experiment / FT operation), distinct
from the QEC :mod:`~ftsim.backends` it runs on. The factory has a few possible
stages -- not a mandatory sequence:

    Preparation / injection
    0-level distillation      -- H6ZeroLevelDistillation  (state verification + post-select)
    Cultivation               -- Cultivation              (grow / protect in a larger code)
    Distillation (L1, L2, ..) -- H6Distillation(level=)    (n -> 1 magic distillation)

``run_pipeline`` picks one with ``zero_lvl_distill=`` / ``cultivation`` /
``distillation=``. With ``factory_code != code`` it is wrapped in
:class:`CrossCodeInject`: the factory runs in one code and the checked resource
is **injected** into the processor's code (no inter-code coupler -- grow the
verified small-code state into the larger code).

This build is the **Clifford proxy** (stim only): the magic block is a
stabilizer stand-in for ``|T>_L``, so error propagation and post-selection yield
are exact but the ``T`` rotation itself is not modelled. Across a cross-code
inject the stages are structurally composed and their yields multiply, but the
resource is a fresh ``|+>_L`` stand-in at each stage -- carrying the true ``|T>``
across the unencode/inject boundary needs the non-Clifford backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import stim

from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.tracker import SyndromeTracker

from ._compiled import Compiled
from .backends import H6_SLOT0, CodeSpec, Driver, UnsupportedGate
from .backends import get as get_code


class MagicProtocol(ABC):
    name: str = "abstract"
    n_magic_patches_per_t: int = 1

    @abstractmethod
    def gadget(self, driver: Driver, builder: CircuitBuilder,
               data_name: str, k: int, *, dag: bool = False) -> Dict[str, List[int]]:
        """Teleport one logical ``T``/``T_DAG`` onto ``data_name``, consuming the
        ``k``-th magic patch(es). Return ``{"qec": [...], "sbranch": [...]}``."""

    @abstractmethod
    def build_standalone(self, code, *, code_kwargs: Optional[dict] = None) -> Compiled:
        """Build just the factory block for a yield / output-infidelity benchmark."""


def _require_h6(spec_or_driver, who: str) -> None:
    name = getattr(spec_or_driver, "spec", spec_or_driver)
    name = getattr(name, "name", name)
    if name != "h6":
        raise UnsupportedGate(f"{who} requires an H6 factory (got {name!r})")


def _teleport(driver, builder, src_name, dst_name, slot0):
    """Transversal CNOT (data control -> magic target) + MZ(magic) + no-S branch
    detector. ``src_name`` is the data patch, ``dst_name`` the magic patch."""
    ops = driver.op_set_of(dst_name)
    ops.transversal_cnot(builder, driver.patch(src_name), driver.patch(dst_name))
    m_data = driver.data(dst_name)
    builder.apply_mid_data_readout({q: "Z" for q in m_data})
    c = builder.circuit
    n_m = len(m_data)
    recs = [stim.target_rec(-n_m + i) for i, q in enumerate(m_data) if q in
            {m_data[j] for j in slot0}]
    c.append("DETECTOR", recs, [0.0, 0.0, 0.0])
    return builder.circuit.num_detectors - 1                      # the no-S detector


# --------------------------------------------------------------------------
# 0-level distillation
# --------------------------------------------------------------------------

class H6ZeroLevelDistillation(MagicProtocol):
    """``[[6,2,2]]`` zero-level distillation: encode a magic block, one
    post-selected SE round (this *is* the zero-level check -- it projects the
    block back into the codespace), then use it."""

    name = "h6_zero_level_distillation"

    def __init__(self, recipe: str = "TT"):
        if recipe not in ("TT", "A", "H"):
            raise ValueError(f"recipe must be 'TT' | 'A' | 'H'; got {recipe!r}")
        self.recipe = recipe

    def _encode_and_check(self, driver, builder, name) -> List[int]:
        """|+>_L magic block + one post-selected SE round. Returns qec dets."""
        driver.encode_zero_l(builder, name)
        driver.op_set_of(name).transversal_hadamard(builder, driver.patch(name))
        lo = builder.circuit.num_detectors
        builder.apply_syndrome_extraction(
            circuit_chunk=driver.extraction_block_of(name).circuit, rounds=1
        )
        return list(range(lo, builder.circuit.num_detectors))

    def gadget(self, driver, builder, data_name, k, *, dag=False):
        _require_h6(driver, self.name)
        m_name = driver.magic_names[k]
        qec = self._encode_and_check(driver, builder, m_name)
        sb = _teleport(driver, builder, data_name, m_name, H6_SLOT0)
        return {"qec": qec, "sbranch": [sb]}

    def build_standalone(self, code, *, code_kwargs=None):
        driver, builder, all_data = _one_block(code, code_kwargs)
        _require_h6(driver.spec, self.name)
        self._encode_and_check(driver, builder, "q0")
        builder.apply_data_readout(final_measurements={q: "X" for q in all_data})
        return _standalone_compiled(driver, builder)


# --------------------------------------------------------------------------
# multi-level distillation
# --------------------------------------------------------------------------

class H6Distillation(MagicProtocol):
    """Magic-H6 ``n -> 1`` distillation.

    ``level=1`` -- a ``[[6,2,2]]`` block + the Bell-pair X-logical H-check
    (``HSixLogicalXCheckBlock``), post-selected. ``level>=2`` -- concatenated
    ``[[36,4,4]]`` -- **not implemented** (LightStim roadmap stage 4).
    """

    name = "h6_distillation"

    def __init__(self, level: int = 1):
        level = int(level)
        if level < 1:
            raise ValueError("distillation level must be >= 1")
        if level >= 2:
            raise NotImplementedError(
                "level-2 concatenated [[36,4,4]] distillation is not implemented "
                "(LightStim roadmap stage 4 -- feat/magic-h6-protocol); "
                "use distillation=1 or zero_lvl_distill=True."
            )
        self.level = level
        self._zero = H6ZeroLevelDistillation("TT")

    def gadget(self, driver, builder, data_name, k, *, dag=False):
        # A single teleported T consumes the block immediately, so level-1 ==
        # zero-level here; the Bell-pair H-check matters when the block is kept
        # (build_standalone).
        return self._zero.gadget(driver, builder, data_name, k, dag=dag)

    def build_standalone(self, code, *, code_kwargs=None):
        from lightstim.qec_code.H_six import (
            HSixExtractionBlock, HSixLogicalOpSet, HSixLogicalXCheckBlock,
        )
        spec = code if isinstance(code, CodeSpec) else get_code(code)
        _require_h6(spec, self.name)
        driver = Driver(spec, 1, h_check_ancillas=2, **(code_kwargs or {}))
        system = driver.system
        patch = driver.patch("q0")
        tracker = SyndromeTracker(system.num_qubits,
                                  expected_num_logicals=system.num_logicals)
        builder = CircuitBuilder(tracker, system, if_detector=True)
        builder.write_coordinates()
        data = sorted(patch.data_indices)
        builder.initialize({q: "Z" for q in data}, n=system.num_qubits)
        driver.encode_zero_l(builder, "q0")
        HSixLogicalOpSet().transversal_hadamard(builder, patch)          # |+>_L
        builder.apply_syndrome_extraction(HSixExtractionBlock(system).circuit, rounds=1)
        builder.apply_syndrome_extraction(
            HSixLogicalXCheckBlock(system).circuit, rounds=1             # the distillation check
        )
        builder.apply_data_readout(final_measurements={q: "X" for q in data})
        return _standalone_compiled(driver, builder)


# --------------------------------------------------------------------------
# cross-code injection (factory in one code -> resource injected into another)
# --------------------------------------------------------------------------

class CrossCodeInject(MagicProtocol):
    """Wrap an H6 factory so its checked resource is **injected** into the
    processor's (different) code, then gate-teleported onto the data patch.

    Proxy composition: H6 factory acceptance gates the shot; a fresh ``|+>_L``
    is prepared + SE-checked in the target code (the "grow"); then a same-code
    transversal-CNOT teleport onto the data patch.
    """

    def __init__(self, inner: MagicProtocol):
        self.inner = inner
        self.name = f"{inner.name}+inject"

    def gadget(self, driver, builder, data_name, k, *, dag=False):
        inner = self.inner._zero if hasattr(self.inner, "_zero") else self.inner

        # 1. H6 factory block: encode |+>_L + zero-level (or L1) SE check. The
        #    block stays encoded -- its check detectors are the factory's
        #    acceptance gate; it is read out with everything at the end.
        h6_name = driver.magic_names[k]
        qec = inner._encode_and_check(driver, builder, h6_name)

        # 2. grow a fresh resource in the target (processor) code: |+>_L on the
        #    inject patch + one post-selected SE round (the injection / growth).
        inj_name = driver.inject_names[k]
        driver.op_set_of(inj_name).transversal_hadamard(builder, driver.patch(inj_name))
        lo = builder.circuit.num_detectors
        builder.apply_syndrome_extraction(
            circuit_chunk=driver.extraction_block_of(inj_name).circuit, rounds=1
        )
        qec += list(range(lo, builder.circuit.num_detectors))

        # 3. same-code teleport: inject patch -> data patch
        slot0 = _z_support_labels(driver, inj_name)
        sb = _teleport(driver, builder, data_name, inj_name, slot0)
        return {"qec": qec, "sbranch": [sb]}

    def build_standalone(self, code, *, code_kwargs=None):
        # benchmark = the H6 factory in isolation (the target-code growth is
        # code-dependent; benchmark it via protocol="memory" on that code)
        return self.inner.build_standalone("h6")


def _z_support_labels(driver, name):
    """Positions within ``driver.data(name)`` whose Z-parity is Z_L."""
    data = driver.data(name)
    support = set(driver.logical_z_support(name))
    return tuple(i for i, q in enumerate(data) if q in support)


# --------------------------------------------------------------------------
# cultivation
# --------------------------------------------------------------------------

class Cultivation(MagicProtocol):
    """Magic-state cultivation (grow / protect a magic state in a larger QEC
    code, then expand) -- **not implemented**. arXiv:2409.17595."""

    name = "cultivation"

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "magic state cultivation (arXiv:2409.17595) is not implemented; "
            "use zero_lvl_distill=True or distillation=1"
        )

    def gadget(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def build_standalone(self, *a, **k):  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def make_magic_source(*, zero_lvl_distill=None, cultivation=False,
                      distillation=None) -> Optional[MagicProtocol]:
    """Build the one magic-state protocol selected by the flags (or ``None``).
    Exactly one of the three may be set."""
    chosen = [n for n, v in (("zero_lvl_distill", zero_lvl_distill),
                             ("cultivation", cultivation),
                             ("distillation", distillation)) if v]
    if len(chosen) > 1:
        raise ValueError(f"set at most one magic-state protocol, got {chosen}")
    if zero_lvl_distill:
        recipe = "TT" if zero_lvl_distill is True else str(zero_lvl_distill)
        return H6ZeroLevelDistillation(recipe)
    if cultivation:
        return Cultivation()
    if distillation:
        level = 1 if distillation is True else int(distillation)
        return H6Distillation(level=level)
    return None


# --------------------------------------------------------------------------
# shared helpers for build_standalone
# --------------------------------------------------------------------------

def _one_block(code, code_kwargs):
    spec: CodeSpec = code if isinstance(code, CodeSpec) else get_code(code)
    driver = Driver(spec, 1, **(code_kwargs or {}))
    system = driver.system
    tracker = SyndromeTracker(system.num_qubits,
                              expected_num_logicals=system.num_logicals)
    builder = CircuitBuilder(tracker, system, if_detector=True)
    builder.write_coordinates()
    all_data = sorted(system.data_indices)
    builder.initialize({q: "Z" for q in all_data}, n=system.num_qubits)
    return driver, builder, all_data


def _standalone_compiled(driver: Driver, builder: CircuitBuilder) -> Compiled:
    circ = builder.circuit
    return Compiled(
        circuit=circ,
        n_logical=int(driver.system.num_logicals),
        n_qubits=circ.num_qubits,
        input_state="magic",
        qec_detectors=list(range(circ.num_detectors)),
        observable_indices=list(range(circ.num_observables)),
    )


__all__ = [
    "MagicProtocol", "H6ZeroLevelDistillation", "H6Distillation",
    "CrossCodeInject", "Cultivation", "make_magic_source",
]
