"""Placement / index bookkeeping for a bank of ``[[6,2,2]]`` patches.

One ``[[6,2,2]]`` patch per logical qubit (we use only logical slot 0, support
``{d0, d2, d4}``; slot 1 rides along as a spectator in ``|0>``).  Plus a pool of
transient "magic" patches, one consumed per ``T`` gate.

Each patch owns 12 consecutive global qubit indices
(``12*patch .. 12*patch + 11``): ``0..5`` data, ``6..7`` X-anc, ``8..9`` Z-anc,
``10..11`` bare (H-check) anc.  All indices returned are global.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .h6 import (
    LOCAL_BARE,
    LOCAL_DATA,
    LOCAL_XANC,
    LOCAL_ZANC,
    PATCH_QUBITS,
)


@dataclass
class Patch:
    name: str
    data: List[int]          # 6 globals, code-label order 0..5
    xanc: List[int]          # 2 globals
    zanc: List[int]          # 2 globals
    bare: List[int]          # 2 globals (H-check ancillas)

    @property
    def slot0(self) -> List[int]:
        return [self.data[0], self.data[2], self.data[4]]

    @property
    def slot1(self) -> List[int]:
        return [self.data[1], self.data[3], self.data[5]]

    @property
    def anc(self) -> List[int]:
        """SE-round ancilla read order for this patch: X-anc pair then Z-anc pair."""
        return [*self.xanc, *self.zanc]


@dataclass
class Layout:
    data_patches: List[Patch]
    magic_patches: List[Patch]
    n_logical: int
    n_magic: int
    all_qubits: List[int] = field(default_factory=list)

    def data(self, q: int) -> Patch:
        return self.data_patches[q]

    def magic(self, k: int) -> Patch:
        return self.magic_patches[k]

#index arithmatic to get the global qubit indices for a given patch
def _mk_patch(name: str, slot: int) -> Patch:
    base = PATCH_QUBITS * slot
    return Patch(
        name=name,
        data=[base + i for i in LOCAL_DATA],
        xanc=[base + i for i in LOCAL_XANC],
        zanc=[base + i for i in LOCAL_ZANC],
        bare=[base + i for i in LOCAL_BARE],
    )

#only the addressing scheme
def build(n_logical: int, n_magic: int) -> Layout:
    """A layout with ``n_logical`` data patches then ``n_magic`` magic patches."""
    data_patches = [_mk_patch(f"q{q}", q) for q in range(n_logical)]
    magic_patches = [_mk_patch(f"m{k}", n_logical + k) for k in range(n_magic)]
    total = PATCH_QUBITS * (n_logical + n_magic)
    return Layout(data_patches, magic_patches, n_logical, n_magic,
                  list(range(total)))
