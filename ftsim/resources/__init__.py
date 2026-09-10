"""Resource states for the ``[[6,2,2]]`` / Magic-H6 code.

Currently just the encoded magic block produced by **zero-level distillation**
(:mod:`ftsim.resources.zero_level_distillation`).
"""

from .zero_level_distillation import MAGIC_INPUT, magic_factory_lines

__all__ = ["MAGIC_INPUT", "magic_factory_lines"]
