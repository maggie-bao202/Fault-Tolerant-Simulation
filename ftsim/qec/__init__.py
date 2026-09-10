"""The [[6,2,2]] code layer: code data and patch layout.

Magic-state prep and the logical-``T`` gadget moved out of ``qec`` --
see :mod:`ftsim.resources` (zero-level distillation) and
:mod:`ftsim.processor` (T-state teleportation).
"""

from .layout import Layout, Patch, build
from .h6 import encoder_lines, hcheck_lines, se_round_lines

__all__ = [
    "Layout", "Patch", "build",
    "encoder_lines", "hcheck_lines", "se_round_lines",
]
