"""Benchmark Qulacs expectation batches with configurable worker counts."""

import argparse
import json
import resource
import time

import numpy as np

from qc_executor import Executor, Parameters, QuantumCircuit, QuantumOperator


def _cpu_time():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def main():
    """Run a benchmark using a single circuit and a batch of rotation angles."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--qubits", type=int, default=14)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable results.")
    args = parser.parse_args()
    if min(args.workers, args.batch_size, args.qubits, args.layers, args.repeats) < 1:
        parser.error("All benchmark arguments must be positive.")
    if args.warmups < 0 or (args.grid_size is not None and args.grid_size < 1):
        parser.error("warmups must be non-negative and grid-size must be positive.")

    p = Parameters("p", 1)
    x = Parameters("x", 1)
    angle = Parameters("angle", 1)
    gate_angle = p[0] * x[0] if args.grid_size is not None else angle[0]
    circuit = QuantumCircuit(args.qubits)
    for _ in range(args.layers):
        for wire in range(args.qubits):
            circuit.ry(wire, gate_angle)
        for wire in range(args.qubits - 1):
            circuit.cx(wire, wire + 1)
    observable = QuantumOperator(["Z" * args.qubits], [1.0])
    executor = Executor.create("qulacs", max_workers=args.workers)
    if args.grid_size is None:
        values = {"angle": np.linspace(0.0, 1.0, args.batch_size).reshape(-1, 1)}
    else:
        rows = np.linspace(0.0, 1.0, args.grid_size).reshape(-1, 1)
        values = {"p": rows, "x": rows}

    for _ in range(args.warmups):
        executor.expectation_value(circuit, observable, **values)

    wall_start = time.perf_counter()
    cpu_start = _cpu_time()
    wall_samples = []
    for _ in range(args.repeats):
        sample_start = time.perf_counter()
        result = executor.expectation_value(circuit, observable, **values)
        wall_samples.append(time.perf_counter() - sample_start)
    wall = time.perf_counter() - wall_start
    cpu = _cpu_time() - cpu_start
    own_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if args.json:
        print(
            json.dumps(
                {
                    "batch_size": (
                        args.grid_size**2 if args.grid_size is not None else args.batch_size
                    ),
                    "workers": args.workers,
                    "qubits": args.qubits,
                    "layers": args.layers,
                    "repeats": args.repeats,
                    "warmups": args.warmups,
                    "grid_size": args.grid_size,
                    "shape": np.shape(result),
                    "values": np.asarray(result).reshape(-1).tolist(),
                    "wall_seconds": wall,
                    "wall_samples_seconds": wall_samples,
                    "cpu_seconds": cpu,
                    "average_cpus": cpu / wall,
                    "peak_rss_self": own_rss,
                    "peak_rss_largest_child": child_rss,
                }
            )
        )
        return
    print(f"shape={result.shape}, workers={args.workers}, repeats={args.repeats}")
    print(f"wall_seconds={wall:.3f}, cpu_seconds={cpu:.3f}, average_cpus={cpu / wall:.2f}")
    print(f"peak_rss_self={own_rss}, peak_rss_largest_child={child_rss} (KiB on Linux)")


if __name__ == "__main__":
    main()
