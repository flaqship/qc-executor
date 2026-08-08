# Circuit abstraction benchmark

What one gate costs in each of the three circuit representations this project
has had:

| name | where | how a gate is stored |
| --- | --- | --- |
| `qiskit-backed` | `develop` | a Qiskit `CircuitInstruction`, inside a `qiskit.QuantumCircuit` |
| `gate-objects` | branch `ba_daniel` | a typed Python object in a list (`CliffordGate`, `RotationGate`, ...) |
| `columnar-ir` | this branch | packed `array` columns, one entry per gate, with a sparse overlay for symbolic angles |

## Running it

Each variant needs its own environment, because they are three versions of the
same package. Create a worktree and a venv per comparison branch:

```bash
git worktree add --detach ../wt-develop origin/develop
git worktree add --detach ../wt-daniel  origin/ba_daniel

uv venv ../venv-develop --python 3.12
uv pip install --python ../venv-develop/Scripts/python.exe -e ../wt-develop packaging pennylane qulacs
# ... same for ba_daniel
```

Then point `VARIANTS` in `run_abstraction_comparison.py` at those interpreters
and run it. `circuit_abstraction.py` measures one point; the driver runs one
process per point and takes the slope.

## Method

**Memory is the working set, not `tracemalloc`.** Qiskit 2.x keeps circuit data
in Rust, which the Python allocator never sees — `tracemalloc` reports about
**0.5 bytes per gate** for it, which measures where the bytes live rather than
how many there are.

**Bytes per gate is the slope between two circuit sizes** (20k and 220k gates),
each built in a fresh process. Taking the slope cancels the interpreter, the
imports and allocator slack, none of which scale with gate count. A single
absolute reading cannot separate those.

**Build and translate are timed together.** The designs put the work in
different places: the Qiskit-backed circuit builds its native form gate by gate
as you call `h()`/`cx()`, so translating is nearly free, while the columnar one
defers everything to compilation. Timing translation alone would flatter
whichever design front-loads less; the benchmark forces the native artifact so
the total is comparable.

The circuit is a repeating `h`, `rx`, `cx` layer on 16 qubits — a third of the
gates carry an angle, so neither the numeric nor the symbolic path dominates by
construction.

## Results

Measured on Windows 11, Python 3.12, qiskit 2.5.1, 16 qubits.

### Memory

Bytes of working set per gate:

| abstraction | numeric angles | symbolic angles |
| --- | ---: | ---: |
| `qiskit-backed` | 51.2 | 252.1 |
| `gate-objects` | 169.4 | 156.0 |
| **`columnar-ir`** | **21.7** | **49.7** |

The columnar store is **2.4x** smaller than the Qiskit-backed circuit for
numeric angles and **5.1x** smaller for symbolic ones; against the typed
gate-object list it is **7.8x** and **3.1x** smaller.

Two things worth reading off this rather than assuming:

- **The Qiskit baseline is much better than it used to be.** The refactor plan
  quoted ~130 bytes/gate numeric and ~646 symbolic, measured against an older
  Qiskit. Qiskit 2.x moved circuit storage into Rust and is far leaner than
  that. The columnar representation still wins, but by less than the plan
  assumed, and the honest numbers are the ones above.
- **`gate-objects` costs about the same either way** (169 vs 156). A Python
  object per gate dominates, and symbolic angles are interned SymPy symbols
  shared across gates, so making the angles symbolic adds almost nothing. For
  the columnar store the symbolic overlay is a real, visible cost: one dict
  entry per symbolic angle, which is the 21.7 -> 49.7 step for a circuit where
  a third of the gates carry one.

### Build and translate

Microseconds per gate, 20k gates:

| abstraction | backend | angles | build | translate | total |
| --- | --- | --- | ---: | ---: | ---: |
| `qiskit-backed` | pennylane | numeric | 1.91 | 5.19 | 7.10 |
| `gate-objects` | pennylane | numeric | 1.82 | 0.46 | 2.28 |
| `columnar-ir` | pennylane | numeric | 1.08 | 2.46 | **3.54** |
| `qiskit-backed` | qulacs | numeric | 1.93 | 8.20 | 10.13 |
| `gate-objects` | qulacs | numeric | — | — | *unsupported* |
| `columnar-ir` | qulacs | numeric | 0.98 | 9.31 | 10.28 |
| `qiskit-backed` | pennylane | symbolic | 1.99 | 232.6 | 234.6 |
| `gate-objects` | pennylane | symbolic | 0.24 | 231.2 | 231.4 |
| `columnar-ir` | pennylane | symbolic | 1.19 | 238.4 | 239.6 |
| `qiskit-backed` | qulacs | symbolic | 2.07 | 305.1 | 307.2 |
| `columnar-ir` | qulacs | symbolic | 1.18 | 331.7 | 332.9 |

On `ba_daniel` the Qulacs backend still reaches for `_qiskit_circuit`, so that
combination raises `AttributeError` rather than producing a number.

**Symbolic translation is ~230-330 us/gate on all three**, roughly 100x the
numeric cost, and the spread between the abstractions is a few percent. That
time is `sympy.lambdify`, called once per symbolic angle — it is not a property
of how gates are stored, and no change to the representation will move it. If
symbolic circuits need to get faster, caching or batching the lambdification is
the thing to attack, not the container.

For numeric circuits the columnar store builds fastest (about 1 us/gate against
1.9), which is where its lack of per-gate Python objects shows.
