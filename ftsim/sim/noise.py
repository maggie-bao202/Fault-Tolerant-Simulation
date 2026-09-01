"""Circuit-level depolarising noise, injected at the text level.

clifft accepts the full Stim noise set (``DEPOLARIZE1/2``, ``X_ERROR``,
``Z_ERROR`` ...), so we insert error instructions line-by-line without needing a
``stim.Circuit``.  ``mode='full'`` puts rate ``p`` on every gate / measurement /
reset; ``mode='idle'`` additionally depolarises the targets of every gate at
``p/5`` beforehand (a coarse memory term).
"""

from __future__ import annotations

from typing import List

_1Q = {"H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_Y", "T", "T_DAG"}
_2Q = {"CX", "CNOT", "CZ", "CY"}
_MEAS_Z = {"M", "MZ", "MR", "MRZ"}
_MEAS_X = {"MX", "MRX"}
_RESET = {"R", "RX", "RY"}
_SKIP = {"QUBIT_COORDS", "SHIFT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE",
         "EXP_VAL", "TICK", "REPEAT", "}", ""}


def _targets(line: str) -> List[str]:
    return line.split()[1:]


def add_noise(text: str, p: float, mode: str = "full") -> str:
    if p <= 0:
        return text
    if mode not in ("full", "idle"):
        raise ValueError("mode must be 'full' or 'idle'.")
    out: List[str] = []
    for line in text.splitlines():
        head = line.split("(", 1)[0].split()[0] if line.strip() else ""
        if head in _SKIP:
            out.append(line)
            continue
        tg = _targets(line)
        if mode == "idle" and head in (_1Q | _2Q):
            out.append(f"DEPOLARIZE1({p/5}) " + " ".join(tg))
        if head in _RESET:
            out.append(line)
            out.append(f"X_ERROR({p}) " + " ".join(tg))
        elif head in _MEAS_Z:
            out.append(f"X_ERROR({p}) " + " ".join(tg))
            out.append(line)
        elif head in _MEAS_X:
            out.append(f"Z_ERROR({p}) " + " ".join(tg))
            out.append(line)
        elif head in _1Q:
            out.append(line)
            out.append(f"DEPOLARIZE1({p}) " + " ".join(tg))
        elif head in _2Q:
            out.append(line)
            out.append(f"DEPOLARIZE2({p}) " + " ".join(tg))
        else:
            out.append(line)
    return "\n".join(out) + "\n"
