"""Pluggable QEC-code backends -- thin descriptors over ``lightstim``.

A code is a :class:`CodeSpec`: it names LightStim's patch class, extraction
block and :class:`~lightstim.ir.operation.LogicalOpSet`, declares which logical
gates it supports (``gate_methods`` -- a capability map + name bridge, *not*
gate logic), and supplies small state-prep callbacks. :class:`Driver` is the
per-compilation system layout (multi-patch ``QECSystem``, patch lookup,
readout). Gate *dispatch* lives with the caller -- :mod:`ftsim.processor`.

A ``CodeSpec`` says nothing about magic states: *what to do* with the code
(process a logical circuit, distil a magic state, cultivate one) is a
**protocol** -- see :mod:`ftsim.processor` and :mod:`ftsim.factory`.

``code="steane"`` is LightStim's ``ColorCode(distance=3)`` (== ``[[7,1,3]]``)
with ``ColorCodeLogicalOpSet``.

Register a code with :func:`register`; run it with
``run_pipeline(lc, p, code=<name>)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional


from lightstim.ir.builder import CircuitBuilder
from lightstim.ir.operation import LogicalOpSet
from lightstim.ir.qec_system import QECSystem


class UnsupportedGate(RuntimeError):
    """The code has no fault-tolerant realisation of this logical gate / state."""


@dataclass(frozen=True)
class CodeSpec:
    name: str
    patch_factory: Callable[..., object]          # e.g. HSixCode  /  lambda d: RotatedSurfaceCode(distance=d)
    extraction_block: type                         # LightStim SE-block class
    op_set: Optional[LogicalOpSet] = None
    gate_methods: Mapping[str, object] = field(default_factory=dict)
    #: (driver, builder, patch_name, state) -> None   -- prepare a logical state
    prepare: Optional["Callable"] = None
    #: (driver, builder, data_indices) -> None        -- encode |0>_L from |0>^n
    #: (None means the Z-basis reset already *is* |0>_L, e.g. surface code)
    encode_zero_l: Optional["Callable"] = None
    layout_pitch: float = 24.0
    logicals_per_patch: int = 1


_CODES: Dict[str, CodeSpec] = {}


def register(spec: CodeSpec) -> CodeSpec:
    _CODES[spec.name] = spec
    return spec


def get(name: str) -> CodeSpec:
    try:
        return _CODES[name]
    except KeyError:
        raise KeyError(f"unknown code {name!r}; available: {sorted(_CODES)}") from None


def list_codes() -> List[str]:
    return sorted(_CODES)


# --------------------------------------------------------------------------
# generic driver
# --------------------------------------------------------------------------

class Driver:
    """One compilation's worth of state for a :class:`CodeSpec`."""

    def __init__(self, spec: CodeSpec, n_patches: int, *, magic: int = 0,
                 factory_spec: Optional[CodeSpec] = None, **code_kwargs):
        self.spec = spec
        self.factory_spec = factory_spec or spec
        self.code_kwargs = code_kwargs
        self.system = QECSystem()
        self.data_names = [f"q{i}" for i in range(n_patches)]
        self.magic_names = [f"m{i}" for i in range(magic)]        # factory_spec patches
        #: cross-code injection targets (data-code patches), only when the
        #: factory runs in a *different* code
        self.inject_names = (
            [f"inj{i}" for i in range(magic)]
            if self.factory_spec is not spec else []
        )
        self._pmap: Dict[str, object] = {}
        self._spec_of: Dict[str, CodeSpec] = {}

        def _add(name, s, slot, dy):
            self._spec_of[name] = s
            self._pmap[name] = self.system.add_patch(
                s.patch_factory(**(code_kwargs if s is spec else {})),
                offset=(s.layout_pitch * (slot % 6), s.layout_pitch * (slot // 6) + dy),
                name=name,
            )

        for slot, name in enumerate(self.data_names):
            _add(name, spec, slot, 0)
        for slot, name in enumerate(self.magic_names):
            _add(name, self.factory_spec, slot, 40)
        for slot, name in enumerate(self.inject_names):
            _add(name, spec, slot, 80)

    # -- geometry -----------------------------------------------------

    def patch(self, name: str):
        return self._pmap[name]

    def data(self, name: str) -> List[int]:
        return sorted(self._pmap[name].data_indices)

    def extraction_block(self):
        return self.spec.extraction_block(self.system)

    # -- construction ----------------------------------------------

    def prepare(self, builder: CircuitBuilder, patch_name: str, state: str) -> None:
        if self.spec.prepare is None:
            if state not in ("0",):
                raise UnsupportedGate(f"{self.spec.name}: only |0>_L prep")
            return
        self.spec.prepare(self, builder, patch_name, state)

    def spec_of(self, name: str) -> CodeSpec:
        return self._spec_of[name]

    def op_set_of(self, name: str):
        return self._spec_of[name].op_set

    def extraction_block_of(self, name: str):
        return self._spec_of[name].extraction_block(self.system)

    def encode_zero_l(self, builder: CircuitBuilder, patch_name: str) -> None:
        """Encode ``|0>_L`` on ``patch_name`` (no-op if the Z-reset already is)."""
        enc = self._spec_of[patch_name].encode_zero_l
        if enc is not None:
            enc(self, builder, self.data(patch_name))

    # -- readout ---------------------------------------------------

    def logical_z_support(self, patch_name: str) -> List[int]:
        for op in getattr(self._pmap[patch_name], "logical_ops", []):
            if op.get("type") == "Z":
                return sorted(op["data_indices"])
        return self.data(patch_name)


# --------------------------------------------------------------------------
# per-code prep helpers  (small; the encoders genuinely differ)
# --------------------------------------------------------------------------

#: data-qubit labels of the used logical slot (slot 0) of the [[6,2,2]] code
H6_SLOT0 = (0, 2, 4)


def _h6_encode_zero_l(driver: Driver, builder: CircuitBuilder, data: List[int]) -> None:
    # get_zero_l_encoder is TICK-separated, so one apply_unitary_block still gets
    # per-gate circuit-level noise placement.
    from lightstim.qec_code.H_six import get_zero_l_encoder
    builder.apply_unitary_block(get_zero_l_encoder(data))


def _h6_prepare(driver: Driver, builder: CircuitBuilder, name: str, state: str) -> None:
    patch = driver.patch(name)
    driver.encode_zero_l(builder, name)                           # |0>_L
    ops = driver.spec.op_set
    if state == "0":
        return
    if state == "1":
        ops.transversal_x(builder, patch)
    elif state == "+":
        ops.transversal_hadamard(builder, patch)
    elif state == "-":
        ops.transversal_x(builder, patch)
        ops.transversal_hadamard(builder, patch)
    else:
        raise ValueError(f"unknown prep state {state!r}")


def _plain_prepare(driver: Driver, builder: CircuitBuilder, name: str, state: str) -> None:
    """State prep for codes whose Z-reset already *is* ``|0>_L`` (surface, Steane):
    the first SE round projects into the codespace."""
    if state == "0":
        return
    patch = driver.patch(name)
    ops = driver.op_set_of(name)
    if state == "1":
        ops.transversal_x(builder, patch)                     # inherited: X on X_L support
    elif state == "+":
        ops.transversal_hadamard(builder, patch)
    elif state == "-":
        ops.transversal_x(builder, patch)
        ops.transversal_hadamard(builder, patch)
    else:
        raise ValueError(f"unknown prep state {state!r}")


def _surface_prepare(driver: Driver, builder: CircuitBuilder, name: str, state: str) -> None:
    if state in ("+", "-"):
        raise UnsupportedGate(
            f"{driver.spec.name}: |+>_L prep needs fold-transversal H (follow-up)"
        )
    _plain_prepare(driver, builder, name, state)


def _unsupported(msg):
    def _h(*_a, **_k):
        raise UnsupportedGate(msg)
    return _h


# --------------------------------------------------------------------------
# built-in codes
# --------------------------------------------------------------------------

def _register_builtins() -> None:
    from lightstim.qec_code.generic_css import GenericCSSColorationExtractionBlock
    from lightstim.qec_code.H_six import (
        HSixCode, HSixExtractionBlock, HSixLogicalOpSet,
    )
    register(CodeSpec(
        name="h6",
        patch_factory=HSixCode,
        extraction_block=HSixExtractionBlock,
        op_set=HSixLogicalOpSet(),
        gate_methods={
            "H": "transversal_hadamard", "S": "transversal_s",
            "S_DAG": "transversal_s_dag", "X": "transversal_x",
            "CNOT": "transversal_cnot",
        },
        prepare=_h6_prepare,
        encode_zero_l=_h6_encode_zero_l,
        layout_pitch=20.0,
        logicals_per_patch=2,
    ))

    from lightstim.qec_code.surface_code.rotated import (
        RotatedSurfaceCode, RotatedSurfaceCodeExtractionBlock,
        RotatedSurfaceCodeLogicalOpSet,
    )
    register(CodeSpec(
        name="rotated_surface",
        patch_factory=lambda distance=3: RotatedSurfaceCode(distance=distance),
        extraction_block=RotatedSurfaceCodeExtractionBlock,
        op_set=RotatedSurfaceCodeLogicalOpSet(RotatedSurfaceCodeExtractionBlock),
        gate_methods={
            "X": "transversal_x",
            "CNOT": "transversal_cnot",
            "H": _unsupported("rotated_surface: fold-transversal H is a follow-up; use code='h6'"),
            "S": _unsupported("rotated_surface: fold-transversal S is a follow-up; use code='h6'"),
        },
        prepare=_surface_prepare,
        layout_pitch=14.0,
    ))

    from lightstim.qec_code.color_code import ColorCode, ColorCodeLogicalOpSet
    register(CodeSpec(
        name="steane",                       # ColorCode(distance=3) == [[7,1,3]]
        patch_factory=lambda: ColorCode(distance=3),
        # generic bipartite-coloring SE (multi-patch friendly; the color code's
        # own middle-out plan is single-patch); no locality gain at d=3.
        extraction_block=GenericCSSColorationExtractionBlock,
        op_set=ColorCodeLogicalOpSet(),
        gate_methods={
            "H": "transversal_hadamard", "S": "transversal_s",
            "S_DAG": "transversal_s_dag", "X": "transversal_x",
            "CNOT": "transversal_cnot",
        },
        prepare=_plain_prepare,
        layout_pitch=18.0,
    ))

    from lightstim.qec_code.repetition import (
        RepetitionCode, RepetitionCodeExtractionBlock,
    )
    register(CodeSpec(
        name="repetition",
        patch_factory=lambda distance=3: RepetitionCode(distance=distance),
        extraction_block=RepetitionCodeExtractionBlock,
    ))
    try:
        from lightstim.qec_code.color_code import ColorCode, ColorCodeExtractionBlock
        register(CodeSpec(
            name="color",
            patch_factory=lambda distance=3: ColorCode(distance=distance),
            extraction_block=ColorCodeExtractionBlock,
        ))
    except Exception:
        pass


_register_builtins()

__all__ = [
    "CodeSpec", "Driver", "UnsupportedGate", "H6_SLOT0",
    "register", "get", "list_codes",
]
