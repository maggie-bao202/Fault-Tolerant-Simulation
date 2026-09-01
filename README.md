# Fault_Tolerant_Sim

End-to-end fault-tolerant simulation pipeline for a logical `{H, S, T, CNOT}` circuit,
run entirely in the `[[6,2,2]]` / Magic-H6 code with **real** `T` gates via the
[clifft](https://github.com/unitaryfoundation/clifft) near-Clifford simulator.

## What it does

Given a logical circuit as a Python gate list, `ftsim`:

1. **Compiles** it to a single `[[6,2,2]]` circuit — one code patch per logical qubit
   (logical slot 0 used; slot 1 is a spectator), `H/S/CNOT` transversal, and each `T`
   realised by distilling a magic block in the H6 `[[6,2,2]]` factory
   (Magic-H6 encoder + transversal physical `T` + stabilizer post-selection)
   and gate-teleporting it — post-selecting the no-`S` branch.
2. **Checks** the noiseless compiled circuit against the intended logical unitary
   (compiled output distribution vs a dense state-vector reference, per input state).
3. **Simulates** the round-trip `lc ; lc†` under circuit-level depolarising noise
   (real `T` gates via clifft) and reports post-selection yield and logical error rate.

Scoring is by **post-selection**: `[[6,2,2]]` has distance 2, so any fired stabilizer /
factory / teleportation-branch detector discards the shot.

### Fidelity notes

- **Clifford gates (`H/S/CNOT`)** are exact and fault-tolerant: LER = 0 at `p = 0`,
  compiled unitary matches exactly.
- **`T` gates** are exact when *not* immediately conjugated by `H` on the same qubit
  (a `T` that is first-on-its-qubit, or follows only `S`/`CNOT`). Bare `[[6,2,2]]`
  has **no exact transversal `T`**, so a `T` sandwiched by `H` carries a residual
  coherent infidelity (~7% TVD per such `T`). This is surfaced per circuit as
  `FTReport.noiseless_error_floor`; `FTReport.logical_error_rate_net` subtracts it.
  Exact arbitrary-`T` fault tolerance needs the concatenated Magic-H6 protocol
  (`lightstim.protocols.magic_h6_benchmark` level 2) — out of scope here.
- clifft cost grows with simultaneously-live magic; keep the `T`-count small.

## Layout

```
ftsim/
  logical_gates.py     LogicalCircuit -- the input gate-list type      (public)
  pipeline.py          run_pipeline(lc, p, ...) -> FTReport            (public)
  qec/                  the [[6,2,2]] code layer
    h6.py               code data + circuit fragments (encoder, SE round, H-check)
    layout.py            patch placement / global-index bookkeeping
    tfactory.py          H6 magic factory + T-teleportation gadget
  compiler/            LogicalCircuit -> one [[6,2,2]] clifft circuit
    emit.py              circuit-text builder with rec[-k] tracking
    lower.py             compile(lc, ...) -> Compiled  (text + detector/observable metadata)
  sim/                 noise, execution, reference
    noise.py             circuit-level depolarising noise (text level)
    simulate.py          run through clifft + post-selected scoring
    ideal.py             dense state-vector reference for the unitary check
tests/                  ideal / clifft-smoke / compile-noiseless / factory / gadget / pipeline
notebooks/ft_pipeline_demo.ipynb
```

**Self-contained**: no dependency on any file outside this repo. The `[[6,2,2]]`
encoder / extraction round and the T-teleportation gadget are vendored into
`ftsim/qec/h6.py` and `ftsim/qec/tfactory.py` (ported from CQCL Magic-H6
`Code614.py` / LightStim `lightstim.qec_code.six_two_two`).

## Requirements

- Python 3.12 (matches the shared `Infleqtion/.venv`)
- `clifft`, `stim`, `numpy`, `pandas` (+ `matplotlib` for the demo notebook)

```
pip install -e .            # from this directory
```

## Quick start

```python
from ftsim.logical_gates import LogicalCircuit
from ftsim.pipeline import run_pipeline

lc = LogicalCircuit([("H", 0), ("T", 0), ("CNOT", 0, 1), ("S", 1), ("T", 1)])
report = run_pipeline(lc, p=1e-3, shots=200_000)
print(report)
```
