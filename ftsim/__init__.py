"""ftsim -- self-contained fault-tolerant [[6,2,2]] simulation pipeline with real T gates."""

from __future__ import annotations

from .logical_gates import LogicalCircuit
from .pipeline import FTReport, run_pipeline

__all__ = ["LogicalCircuit", "FTReport", "run_pipeline"]
