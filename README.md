# Fault-Tolerant-Simulation

Pluggable end-to-end fault-tolerant simulation of a logical `{H, S, T, CNOT}`
circuit, built on **[LightStim](../LightStim)**. The QEC **code** and the
**protocol** are separate axes; magic-state supply is chosen by flag.

```python
from ftsim import LogicalCircuit, run_pipeline

lc = LogicalCircuit([("H", 0), ("T", 0), ("CNOT", 0, 1), ("S", 1), ("T", 1)])

run_pipeline(lc, p=1e-3)                              # processor, T via zero-level distillation
run_pipeline(lc, p=1e-3, distillation=1)             # T via level-1 Magic-H6 distillation
run_pipeline(lc, p=1e-3, code="rotated_surface")    # swap the code
run_pipeline(lc, p=1e-3, protocol="memory")         # d-round memory experiment
run_pipeline(None, p=1e-3, protocol="factory",      # benchmark the magic factory alone
             zero_lvl_distill=True)
```

## Code ≠ protocol

* **`code=`** — the encoded structure only (`lightstim.qec_code`): `"h6"`,
  `"steane"` (= `ColorCode(distance=3)`, `[[7,1,3]]`), `"rotated_surface"`,
  `"repetition"`, `"color"`, … A `CodeSpec` names LightStim's patch class, SE
  block and `LogicalOpSet` + a gate→method map. Codes carry **no** magic-state
  logic.
  * **`factory_code=`** — run the magic-state factory in a *different* code from
    the processor. The checked resource is **injected** into the processor's
    code (grow the verified small-code state into the larger one — no inter-code
    coupler), then gate-teleported. e.g.
    `run_pipeline(lc, code="steane", factory_code="h6", distillation=1)`.
    Proxy note: the factory acceptance and the target-code growth compose and
    their yields multiply, but the resource is a fresh `|+>_L` stand-in at each
    stage until the non-Clifford backend lands.
* **`protocol=`** — what to *do* with the code:
  * `"processor"` (default) — run the logical circuit: per-patch encode →
    `{transversal gate layer ; SE round}` → readout. Clifford gates driven by
    LightStim `LogicalOpSet` methods (routed by `lightstim.ir.LogicalExecutor`).
  * `"memory"` — a d-round memory experiment, delegated to
    `lightstim.protocols.MemoryExperiment`.
  * `"factory"` — build *just* a magic-state factory block and report its yield +
    output infidelity.

## Magic-state factory

A magic-state factory has several possible protocols (not a mandatory sequence).
Pick one with a flag; it is the `T` source for the processor, or the standalone
subject of `protocol="factory"`:

| flag | protocol | status |
|---|---|---|
| `zero_lvl_distill=` | `[[6,2,2]]` **0-level distillation** — encode + one SE round (post-selected) + teleport. `True` / `"TT"` / `"A"` / `"H"`. | ✅ (Clifford proxy; recipe `"TT"` = the reference; `"A"`/`"H"` are the non-Clifford path) |
| `distillation=` | Magic-H6 **n→1 distillation**. `1` = one `[[6,2,2]]` block + the Bell-pair X-logical H-check. `>=2` = concatenated `[[36,4,4]]`. | `1` ✅, `>=2` → `NotImplementedError` (LightStim roadmap stage 4) |
| `cultivation=` | **Magic-state cultivation** (grow/protect in a larger code; arXiv:2409.17595). | → `NotImplementedError` |

At most one may be set. `T` is the **Magic-H6 Clifford proxy** (stim-only): the
magic block is a stabilizer stand-in for `|T>_L`, so error propagation and
post-selection yield are exact but the `T` rotation is not modelled
(`noiseless_error_floor == 0`; `check_unitary` says so for non-Clifford circuits).
Real-`T` fidelity and the `p_in → p_in²` gain are the follow-up (LightStim
`feat/magic-h6-protocol` + a `clifft` backend).

## Layout

```
ftsim/
  logical_gates.py   LogicalCircuit -- the input gate-list type          (public)
  pipeline.py        run_pipeline / check_unitary / FTReport             (public)
  backends.py        CodeSpec (descriptor over LightStim classes) + Driver
  processor.py       the processor protocol (LogicalCircuit -> stim.Circuit)
                     + compile_memory (-> lightstim.protocols.MemoryExperiment)
  factory.py         MagicProtocol base + H6ZeroLevelDistillation /
                     H6Distillation(level=) / Cultivation + make_magic_source
  score.py           sample -> post-select -> logical-error rate
  sim/ideal.py       dense state-vector reference for check_unitary
  ppm/               opt-in Pauli-based-computation frontend (frontend="ppm")
                     LogicalCircuit -> Catalyst PPM passes -> PBC IR ->
                     encoded stim (Clifford PPMs only; pi/8 -> magic stub)
tests/
notebooks/ft_pipeline_demo.ipynb
```

Add a code: one `ftsim.backends.register(CodeSpec(...))`. Add a factory protocol:
subclass `ftsim.factory.MagicProtocol` and wire it into `make_magic_source`.

## PPM frontend (opt-in)

`run_pipeline(lc, p, frontend="ppm")` routes the **processor** protocol through a
Pauli-based-computation layer: the logical circuit is lowered by
PennyLane/Catalyst (`to_ppr -> commute_ppr -> merge_ppr_ppm`) to a sequence of
Pauli Product Rotations (`pi/4`/`pi/2` Clifford, `pi/8` non-Clifford) plus
terminal `Z` measurements, and **only the Clifford part** is compiled to an
encoded `stim.Circuit` -- each Clifford PPR re-synthesised from transversal
`{H, S, CNOT}` on `code="h6"` or `"steane"`. Every non-Clifford `pi/8` rotation
raises in `ftsim.ppm.magic.magic_injection` (the seam where magic-state creation
+ injection will land). The default `frontend="gate"` path is unchanged.

Needs the optional dependency group: `pip install -e ".[ppm]"`.

## Requirements

Python 3.12; `lightstim` (editable, `../LightStim`), `stim`, `numpy`, `pandas`.
Optional: `ftsim[ppm]` = `pennylane` + `pennylane-catalyst` for `frontend="ppm"`.

```
uv pip install -e ../LightStim -e .      # or: uv sync
```
