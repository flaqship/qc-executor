"""Measure what one gate costs in each circuit abstraction.

Run under an environment that has exactly one of the three versions installed;
the variant is chosen with --variant.  Emits one JSON object so the runs can be
compared side by side.

Three abstractions:

  qiskit-backed  the original: every gate is a Qiskit CircuitInstruction
  gate-objects   branch ba_daniel: every gate is a typed Python object
  columnar-ir    this branch: gates live in packed arrays

Memory is process working set, not ``tracemalloc``.  Qiskit 2.x keeps circuit
data in Rust, which the Python allocator never sees -- tracemalloc reports
~0.5 bytes per gate for it, which is an artefact of where the bytes live rather
than a measurement.  Working set counts them wherever they are.
"""

# The gate-objects variant lives on another branch, so its imports cannot
# resolve in this checkout; the ctypes struct is a plain record by nature.
# pylint: disable=import-error,no-name-in-module,too-few-public-methods
# pylint: disable=attribute-defined-outside-init

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import time


def _working_set_bytes() -> int:
    """Return the process working set, the closest thing to RSS on Windows."""

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    # The pseudo-handle is -1; without an explicit pointer restype ctypes
    # truncates it and the call fails silently, reporting zero bytes.
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
        ctypes.c_void_p(kernel32.GetCurrentProcess()), ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise OSError("GetProcessMemoryInfo failed")
    return int(counters.WorkingSetSize)


def _make_builder(variant: str):
    """Return ``(circuit_class, parameters_class)`` for the installed variant."""
    if variant == "gate-objects":
        from qc_executor.abstraction.abstract_parameter import ParameterVector
        from qc_executor.abstraction.abstract_quantum_circuit import AbstractQuantumCircuit

        return AbstractQuantumCircuit, ParameterVector

    from qc_executor import Parameters, QuantumCircuit

    return QuantumCircuit, Parameters


def build(circuit_class, parameters_class, num_qubits: int, num_gates: int, symbolic: bool):
    """Build a circuit of roughly ``num_gates`` instructions.

    A repeating layer of one- and two-qubit gates, a third of them rotations,
    so the mix is representative rather than best case for any storage scheme.
    """
    circuit = circuit_class(num_qubits)
    angles = parameters_class("x", num_qubits) if symbolic else None
    placed = 0
    while placed < num_gates:
        for qubit in range(num_qubits):
            if placed >= num_gates:
                break
            circuit.h(qubit)
            placed += 1
            if placed >= num_gates:
                break
            circuit.rx(qubit, angles[qubit] if symbolic else 0.1 * (qubit + 1))
            placed += 1
            if placed >= num_gates:
                break
            circuit.cx(qubit, (qubit + 1) % num_qubits)
            placed += 1
    return circuit


def measure(variant: str, num_qubits: int, num_gates: int, symbolic: bool) -> dict:
    """Measure absolute memory and build time for one size, in this process.

    Reported as an absolute working set, not a delta: the caller runs one
    process per size and takes the slope, so the interpreter, the imports and
    any allocator slack cancel out.
    """
    circuit_class, parameters_class = _make_builder(variant)
    gc.collect()

    start = time.perf_counter()
    circuit = build(circuit_class, parameters_class, num_qubits, num_gates, symbolic)
    elapsed = time.perf_counter() - start
    gc.collect()

    return {
        "variant": variant,
        "num_qubits": num_qubits,
        "num_gates": num_gates,
        "symbolic": symbolic,
        "working_set_bytes": _working_set_bytes(),
        "build_us_per_gate": elapsed / num_gates * 1e6 if num_gates else 0.0,
        "held": len(circuit.parameters) >= 0,
    }


def _force_compiled(native) -> object:
    """Touch whatever a backend builds lazily, so timings include the work.

    Without this the columnar variant appears ~1000x faster than it is: it
    only wraps the instruction store, and the compilation happens on first
    access to the native artifact.
    """
    for attribute in ("native", "qiskit_circuit", "pennylane_circuit", "gates"):
        value = getattr(native, attribute, None)
        if value is not None:
            return value
    return native


def translate(variant: str, backend: str, num_qubits: int, num_gates: int, symbolic: bool) -> dict:
    """Measure build-plus-translate, the cost of getting to a runnable circuit.

    Measured together because the designs put the work in different places:
    the Qiskit-backed circuit builds its native form gate by gate, so
    translation is free; the columnar one defers everything to compilation.
    """
    from qc_executor import Executor

    circuit_class, parameters_class = _make_builder(variant)
    executor = Executor.create(backend)

    # Warm up caches and any first-call machinery.
    _force_compiled(
        executor.transpile_circuit(
            build(circuit_class, parameters_class, num_qubits, 2000, symbolic)
        )
    )

    start = time.perf_counter()
    circuit = build(circuit_class, parameters_class, num_qubits, num_gates, symbolic)
    build_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    _force_compiled(executor.transpile_circuit(circuit))
    translate_elapsed = time.perf_counter() - start

    return {
        "variant": variant,
        "backend": backend,
        "num_gates": num_gates,
        "symbolic": symbolic,
        "build_us_per_gate": build_elapsed / num_gates * 1e6,
        "translate_us_per_gate": translate_elapsed / num_gates * 1e6,
        "total_us_per_gate": (build_elapsed + translate_elapsed) / num_gates * 1e6,
    }


def main() -> None:
    """Measure one configuration, or the translate sweep, and print JSON."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--backends", default="")
    parser.add_argument("--gates", type=int, default=None)
    parser.add_argument("--qubits", type=int, default=16)
    parser.add_argument("--symbolic", action="store_true")
    args = parser.parse_args()

    results = {"memory": [], "translate": []}
    if args.gates is not None:
        results["memory"].append(measure(args.variant, args.qubits, args.gates, args.symbolic))
        print(json.dumps(results))
        return

    for backend in filter(None, args.backends.split(",")):
        for symbolic in (False, True):
            try:
                results["translate"].append(translate(args.variant, backend, 16, 20_000, symbolic))
            except Exception as error:  # noqa: BLE001 - reported, not raised
                results["translate"].append(
                    {
                        "variant": args.variant,
                        "backend": backend,
                        "symbolic": symbolic,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )

    print(json.dumps(results))


if __name__ == "__main__":
    main()
