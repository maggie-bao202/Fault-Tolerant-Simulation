"""The ``[[6,2,2]]`` processor: run a ``LogicalCircuit`` through every step.

* :mod:`ftsim.processor.processor` -- ``compile(lc, ...) -> Compiled``: prep,
  stabilizer-extraction rounds, transversal gate layers, one T-state teleport
  block per ``T``, and the final readout, all lowered into one clifft circuit.
* :mod:`ftsim.processor.t_state_teleport` -- the logical-``T`` gadget that
  consumes a distilled magic block (see :mod:`ftsim.resources`).
* :mod:`ftsim.processor.emit` -- the circuit-text builder with ``rec[-k]``
  tracking used by the compile step.
"""

from .emit import Circ
from .t_state_teleport import MAGIC_FIXUP, t_teleport_lines
from .processor import Compiled, compile

__all__ = ["Circ", "MAGIC_FIXUP", "t_teleport_lines", "Compiled", "compile"]
