"""Drive the abstraction benchmark: one fresh process per data point.

Memory per gate is the slope between two circuit sizes, not a single reading.
Taking the slope cancels the interpreter, the imports and any allocator slack,
none of which scale with the number of gates.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCRIPT = HERE / "bench_abstraction.py"

#: Which interpreter has which abstraction installed.
VARIANTS = {
    "qiskit-backed": HERE / "venv-wt-develop" / "Scripts" / "python.exe",
    "gate-objects": HERE / "venv-wt-daniel" / "Scripts" / "python.exe",
    "columnar-ir": Path(sys.executable),
}

SMALL, LARGE = 20_000, 220_000
QUBITS = 16


def run(python: Path, args: list[str]) -> dict:
    """Run one measurement in its own process and return the parsed result."""
    completed = subprocess.run(
        [str(python), str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in completed.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"no result from {python}:\n{completed.stdout}\n{completed.stderr}")


def main() -> None:
    """Run every variant at two sizes and print the bytes-per-gate slope."""
    rows = []
    for variant, python in VARIANTS.items():
        for symbolic in (False, True):
            flags = ["--symbolic"] if symbolic else []
            small = run(
                python,
                ["--variant", variant, "--gates", str(SMALL), "--qubits", str(QUBITS), *flags],
            )["memory"][0]
            large = run(
                python,
                ["--variant", variant, "--gates", str(LARGE), "--qubits", str(QUBITS), *flags],
            )["memory"][0]

            bytes_per_gate = (large["working_set_bytes"] - small["working_set_bytes"]) / (
                LARGE - SMALL
            )
            rows.append(
                {
                    "variant": variant,
                    "symbolic": symbolic,
                    "bytes_per_gate": bytes_per_gate,
                    "build_us_per_gate": large["build_us_per_gate"],
                }
            )

    width = max(len(r["variant"]) for r in rows)
    print(f"\n{'abstraction':<{width}}  {'angles':<9} {'bytes/gate':>11} {'build us/gate':>14}")
    print("-" * (width + 40))
    for row in rows:
        angles = "symbolic" if row["symbolic"] else "numeric"
        print(
            f"{row['variant']:<{width}}  {angles:<9} "
            f"{row['bytes_per_gate']:>11.1f} {row['build_us_per_gate']:>14.3f}"
        )
    print()
    (HERE / "bench_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
