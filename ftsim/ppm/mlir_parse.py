"""Parse Catalyst's ``pbc``-dialect MLIR into a :class:`~ftsim.ppm.ir.PBCCircuit`.

The Pauli products produced by ``to_ppr -> commute_ppr -> merge_ppr_ppm`` live
only in the compiled MLIR (they are MLIR transform passes, not PennyLane-level
rewrites) and Catalyst exposes no structured Python view of them -- only aggregate
counts via ``catalyst.passes.ppm_specs``.  So this module reads the text emitted
by ``catalyst.debug.get_compilation_stage(..., "QuantumCompilationStage")``.

Rather than line-oriented regex, it runs a small **MLIR tokenizer**
(:func:`_tokenize`) and a statement matcher over the flat QNode body.  It
recognises exactly the handful of operations it needs::

    %c   = arith.constant 1 : i64
    %q   = quantum.extract %reg[ 0 ] : !quantum.reg -> !quantum.bit
    %q   = quantum.extract %reg[%c] : ...
    %r   = pbc.ppr ["Z"](4) %q : !quantum.bit           # (4)->pi/4, (-8)->pi/8 sign -1
    %r:2 = pbc.ppr ["Z", "X"](4) %a, %b : ...           # grouped result
    %m,%o= pbc.ppm ["Z"](-) %q : i1, !quantum.bit       # (-) -> measurement sign -1
    %obs = quantum.namedobs %r[ PauliZ ] : !quantum.obs # terminal measurement

A qubit SSA value's *wire index* is the integer index of the ``quantum.extract``
that produced it; ``pbc.ppr`` threads its qubits 1:1 (result i <- operand i) and
``pbc.ppm`` additionally returns a leading ``i1`` result which is skipped.
Register threading through ``quantum.insert`` never changes a wire, so it is
ignored.  Everything else (``quantum.gphase``, ``catalyst.list_*``,
``quantum.expval``, ``tensor.*`` ...) is ignored.  A classically-conditioned
``pbc.ppr`` (``cond(...)``) or any unresolved qubit SSA raises
:class:`~ftsim.ppm._errors.PPMUnsupported` -- the parser never guesses a wire.

**Version-sensitive.**  Regenerate ``tests/fixtures/ppm/*.mlir`` from real
``qml.qjit`` output and re-check when bumping ``pennylane-catalyst`` (see
``ftsim.ppm.catalyst_frontend.TESTED_CATALYST``).
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ._errors import PPMUnsupported
from .ir import PPM, PPR, PBCCircuit

_DENOM_TO_ANGLE = {2: "pi/2", 4: "pi/4", 8: "pi/8"}
_NAMEDOBS_PAULI = {"PAULIX": "X", "PAULIY": "Y", "PAULIZ": "Z"}

# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------

_TOKEN = re.compile(
    r"""
      (?P<WS>        [ \t]+                      )
    | (?P<COMMENT>   //[^\n]*                    )
    | (?P<NL>        \n                          )
    | (?P<STR>       "(?:[^"\\]|\\.)*"           )
    | (?P<SSA>       %[A-Za-z0-9_.$#-]+          )
    | (?P<SYM>       @[A-Za-z0-9_.$-]+           )
    | (?P<INT>       -?\d+                       )
    | (?P<TYPE>      ![A-Za-z0-9_.]+             )
    | (?P<NAME>      [A-Za-z_][A-Za-z0-9_.]*     )
    | (?P<ARROW>     ->                          )
    | (?P<PUNCT>     [=(){}\[\]<>:,]             )
    """,
    re.VERBOSE,
)

# only these group a statement across newlines; "{ }" region/attr braces do not
# (region bodies are newline-separated; multi-line attribute dicts do not occur
# in the flat QNode body this parser consumes).
_OPEN = {"(": ")", "[": "]", "<": ">"}
_CLOSE = set(_OPEN.values())


def _tokenize(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    pos = 0
    n = len(text)
    while pos < n:
        m = _TOKEN.match(text, pos)
        if m is None:
            pos += 1  # skip anything unrecognised (e.g. stray unicode)
            continue
        pos = m.end()
        kind = m.lastgroup or "PUNCT"
        if kind in ("WS", "COMMENT"):
            continue
        out.append((kind, m.group()))
    return out


def _statements(tokens: List[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
    """Split the token stream into statements at newlines that sit outside any
    ``() [] {} <>`` group."""
    stmts: List[List[Tuple[str, str]]] = []
    cur: List[Tuple[str, str]] = []
    depth = 0
    for kind, val in tokens:
        if kind == "NL":
            if depth == 0 and cur:
                stmts.append(cur)
                cur = []
            continue
        if kind == "PUNCT" and val in _OPEN:
            depth += 1
        elif kind == "PUNCT" and val in _CLOSE:
            depth = max(0, depth - 1)
        cur.append((kind, val))
    if cur:
        stmts.append(cur)
    return stmts


# ---------------------------------------------------------------------------
# statement helpers
# ---------------------------------------------------------------------------

def _split_results(stmt):
    """``([result SSAs], op-name, [rest tokens])`` for ``%a, %b:2 = op ...``;
    ``([], op-name, rest)`` when there is no ``=``."""
    eq = next((i for i, (k, v) in enumerate(stmt) if k == "PUNCT" and v == "="), -1)
    if eq == -1:
        name = stmt[0][1] if stmt and stmt[0][0] == "NAME" else None
        return [], name, stmt[1:]
    results: List[str] = []
    i = 0
    while i < eq:
        k, v = stmt[i]
        if k == "SSA":
            if i + 2 < eq and stmt[i + 1] == ("PUNCT", ":") and stmt[i + 2][0] == "INT":
                results.extend(f"{v}#{j}" for j in range(int(stmt[i + 2][1])))
                i += 3
            else:
                results.append(v)
                i += 1
        else:
            i += 1
    rest = stmt[eq + 1 :]
    name = rest[0][1] if rest and rest[0][0] == "NAME" else None
    return results, name, rest[1:]


def _bracket_strings(rest):
    """The quoted words inside the first ``[ ... ]`` group, and the token index
    just past its closing ``]``."""
    try:
        lo = next(i for i, (k, v) in enumerate(rest) if v == "[")
    except StopIteration:
        return None, 0
    depth = 0
    words: List[str] = []
    for i in range(lo, len(rest)):
        k, v = rest[i]
        if v == "[":
            depth += 1
        elif v == "]":
            depth -= 1
            if depth == 0:
                return words, i + 1
        elif k == "STR":
            words.append(v.strip('"').upper())
        elif k == "NAME":
            words.append(v.upper())
    return words, len(rest)


def _paren_token(rest, start):
    """A ``( <int|-> )`` immediately at ``rest[start]`` -> its inner text, else
    ``None``."""
    if start < len(rest) and rest[start] == ("PUNCT", "("):
        inner = rest[start + 1]
        if inner[0] in ("INT", "PUNCT") and inner[1] not in (")",):
            return inner[1]
    return None


def _operand_ssas(rest, start):
    """SSA operands from ``rest[start]`` up to ``:`` / ``->`` / ``cond`` / end."""
    ssas: List[str] = []
    for k, v in rest[start:]:
        if v in (":", "->") or (k == "NAME" and v == "cond"):
            break
        if k == "SSA":
            ssas.append(v)
    return ssas


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def parse_pbc(mlir_text: str, n_qubits: int) -> PBCCircuit:
    """Parse ``mlir_text`` into a :class:`PBCCircuit` over ``n_qubits`` logical
    qubits.  Raises :class:`PPMUnsupported` on any construct outside the small
    grammar this module recognises."""
    stmts = _statements(_tokenize(mlir_text))

    const_of: Dict[str, int] = {}
    wire_of: Dict[str, int] = {}
    rotations: List[PPR] = []
    measurements: List[PPM] = []
    seen_terminal = False

    def _wire(ssa: str, stmt) -> int:
        try:
            return wire_of[ssa]
        except KeyError:
            raise PPMUnsupported(
                f"could not resolve qubit SSA {ssa!r} to a wire index in: "
                f"{_render(stmt)!r}"
            ) from None

    for stmt in stmts:
        results, op, rest = _split_results(stmt)
        if op is None:
            continue

        if op == "arith.constant":
            val = next((v for k, v in rest if k == "INT"), None)
            if results and val is not None:
                const_of[results[0]] = int(val)
            continue

        if op == "quantum.extract":
            # %r = quantum.extract %reg [ <idx> ] : ...
            try:
                lo = next(i for i, (k, v) in enumerate(rest) if v == "[")
                idx_tok = rest[lo + 1]
            except (StopIteration, IndexError):
                continue
            if not results:
                continue
            if idx_tok[0] == "INT":
                wire_of[results[0]] = int(idx_tok[1])
            elif idx_tok[0] == "SSA":
                if idx_tok[1] not in const_of:
                    raise PPMUnsupported(
                        f"quantum.extract index {idx_tok[1]!r} is not a known "
                        f"integer constant: {_render(stmt)!r}"
                    )
                wire_of[results[0]] = const_of[idx_tok[1]]
            continue

        if op == "quantum.namedobs":
            # %obs = quantum.namedobs %q [ PauliZ ] : !quantum.obs
            q = next((v for k, v in rest if k == "SSA"), None)
            words, _ = _bracket_strings(rest)
            basis = _NAMEDOBS_PAULI.get((words or [""])[0])
            if q is None or basis is None:
                raise PPMUnsupported(f"unparseable namedobs: {_render(stmt)!r}")
            seen_terminal = True
            measurements.append(PPM(paulis=(basis,), qubits=(_wire(q, stmt),)))
            continue

        if op not in ("pbc.ppr", "pbc.ppm"):
            continue

        if any(k == "NAME" and v == "cond" for k, v in rest):
            raise PPMUnsupported(
                "classically-conditioned pbc.ppr (feed-forward) is not "
                f"supported by the PPM frontend: {_render(stmt)!r}"
            )

        letters, after = _bracket_strings(rest)
        if not letters or any(t not in ("X", "Y", "Z", "I") for t in letters):
            raise PPMUnsupported(f"unparseable Pauli list: {_render(stmt)!r}")
        tok = _paren_token(rest, after)
        opnd_start = after + (3 if tok is not None else 0)
        operands = _operand_ssas(rest, opnd_start)
        if len(letters) != len(operands):
            raise PPMUnsupported(
                f"{op}: {len(letters)} Pauli letters vs {len(operands)} "
                f"operands: {_render(stmt)!r}"
            )
        wires = [_wire(s, stmt) for s in operands]

        if op == "pbc.ppr":
            if seen_terminal:
                raise PPMUnsupported(
                    "pbc.ppr after a terminal measurement: the pass stack or "
                    f"MLIR grammar has changed: {_render(stmt)!r}"
                )
            if tok in (None, "-"):
                raise PPMUnsupported(
                    f"pbc.ppr without a rotation denominator: {_render(stmt)!r}"
                )
            denom = abs(int(tok))
            if denom not in _DENOM_TO_ANGLE:
                raise PPMUnsupported(
                    f"pbc.ppr denominator {denom} is not 2/4/8: {_render(stmt)!r}"
                )
            rotations.append(
                PPR(
                    paulis=letters,
                    qubits=wires,
                    angle=_DENOM_TO_ANGLE[denom],
                    sign=-1 if int(tok) < 0 else 1,
                )
            )
            for r, w in zip(results[: len(wires)], wires):
                wire_of[r] = w
        else:  # pbc.ppm -- explicit terminal measurement
            seen_terminal = True
            sign = -1 if tok == "-" or (tok not in (None, "-") and int(tok) < 0) else 1
            measurements.append(PPM(paulis=letters, qubits=wires, sign=sign))
            qres = results[1:] if len(results) == len(wires) + 1 else results
            for r, w in zip(qres, wires):
                wire_of[r] = w

    if not measurements:
        raise PPMUnsupported(
            "no terminal measurements (quantum.namedobs / pbc.ppm) found in the "
            "compiled MLIR"
        )
    return PBCCircuit(n_qubits=n_qubits, rotations=rotations,
                      measurements=measurements)


def _render(stmt) -> str:
    return " ".join(v for _, v in stmt)


__all__ = ["parse_pbc"]
