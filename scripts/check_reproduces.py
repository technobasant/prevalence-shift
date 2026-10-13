"""Fail unless a fresh `sim.py --reps 1000` reproduces the committed results.

Compares out/results.json (just written) with the committed copy in git, every
number, ignoring only the environment block and the CPU time. Floats get a
relative tolerance of 1e-6: on the machine that produced the figures the rerun
is bit-identical, and the tolerance only absorbs last-bit libm differences on
other platforms — far below the four decimals MEASUREMENTS.md prints.

    uv run python sim.py --reps 1000 && uv run python scripts/check_reproduces.py
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {"env", "cpu_seconds"}
RTOL, ATOL = 1e-6, 1e-12


def committed() -> dict:
    blob = subprocess.run(["git", "show", "HEAD:out/results.json"], cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout
    return json.loads(blob)


def compare(a, b, path: str, problems: list[str]) -> None:
    if isinstance(a, dict):
        if set(a) != set(b):
            problems.append(f"{path}: keys differ")
            return
        for k in a:
            if path == "" and k in IGNORED:
                continue
            compare(a[k], b[k], f"{path}/{k}", problems)
    elif isinstance(a, list):
        if len(a) != len(b):
            problems.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", problems)
    elif isinstance(a, float) or isinstance(b, float):
        if not math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL):
            problems.append(f"{path}: committed {a!r}, rerun {b!r}")
    elif a != b:
        problems.append(f"{path}: committed {a!r}, rerun {b!r}")


def main() -> int:
    rerun = json.loads((ROOT / "out" / "results.json").read_text())
    problems: list[str] = []
    compare(committed(), rerun, "", problems)
    if problems:
        print(f"{len(problems)} figure(s) did not reproduce:")
        print("\n".join(problems[:50]))
        return 1
    print("results.json reproduces: every committed figure matches the rerun")
    return 0


if __name__ == "__main__":
    sys.exit(main())
