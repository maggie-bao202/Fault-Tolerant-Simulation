"""The [[6,2,2]] code layer: code data, patch layout, magic factory + teleport."""

from .layout import Layout, Patch, build
from .h6 import encoder_lines, hcheck_lines, se_round_lines
from .tfactory import (
    MAGIC_FIXUP,
    MAGIC_INPUT,
    magic_factory_lines,
    t_teleport_lines,
)

__all__ = [
    "Layout", "Patch", "build",
    "encoder_lines", "hcheck_lines", "se_round_lines",
    "MAGIC_FIXUP", "MAGIC_INPUT", "magic_factory_lines", "t_teleport_lines",
]
