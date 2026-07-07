"""Performance-Benchmark des Executors (Abstraktionsschicht), backend-umschaltbar.

Eigenständiges Gegenstück zum reinen PennyLane-/Qiskit-Benchmark: hier läuft
dieselbe Arbeit (creation / execution / gradient) über den ``Executor`` – einmal
mit dem PennyLane-, einmal mit dem Qiskit-Backend (``BACKEND_CHOICE``).

Der Aufbau ist bewusst IDENTISCH zum Roh-Framework-Benchmark (gleiche Gate-Sets,
gleiche GATE_CONFIGS/QUBIT_CONFIGS, gleiche Mess-Methodik, gleiche CSV-Spalten
``qnc_/qc_``, gleiches Plot-Layout). Dadurch lassen sich die beiden Benchmarks
fair vergleichen: der Overhead deiner Abstraktion ergibt sich als spaltenweise
Differenz  Executor − Roh-Framework  für dieselbe Zelle.

Zwei Linien – sie entsprechen genau der einen echten Executor-Option ``caching``
(``Executor.create(backend, caching=...)``) und sind 1:1 auf die passenden Linien
des Roh-Benchmarks gemappt:

    qnc  (QNode no cache)→ Executor mit caching=False (Standardfall): rechnet bei
                           jedem Aufruf neu (abstrakter Circuit wird je Aufruf
                           ins native Format übersetzt und simuliert).
    qc   (QNode cached)  → Executor mit caching=True: der Ergebnis-Cache greift nach
                           dem Warm-up, jeder weitere identische Aufruf ist ein
                           Cache-Treffer.

Hinweis: Es werden nur Gate-Sets verwendet, die die Abstraktionsschicht
unterstützt (kein Rot/Toffoli), damit derselbe GATE_SET_CHOICE im Roh-Benchmark
denselben Schaltkreis erzeugt.
"""

import gc
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from qc_executor import Executor
from qc_executor.abstraction import (
    AbstractQuantumCircuit,
    AbstractQuantumOperator,
    ParameterVector,
)

# Windows-Konsole auf UTF-8 stellen (sonst UnicodeEncodeError bei → und Umlauten)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =========================================================
# ➤  BACKEND-AUSWAHL  ←  hier anpassen
# =========================================================
#
#  "pennylane"  → Executor.create("pennylane")
#  "qiskit"     → Executor.create("qiskit")
#
BACKEND_CHOICE = "pennylane"

# =========================================================
# ➤  GATE-SET AUSWAHL  ←  hier anpassen
# =========================================================
#
#  "clifford"               → h, s, cx, x, y, z
#  "clifford_t"             → h, t, cx
#  "single_qubit_plus_cnot" → rx, ry, rz, cx
#
# Namen und Gatter decken sich mit den gleichnamigen Sets des Roh-Benchmarks
# (nur die von der Abstraktion unterstützten – Rot/Toffoli fehlen dort).
#
GATE_SET_CHOICE = "clifford"

# =========================================================
# ➤  BENCHMARK-MODUS  ←  hier anpassen
# =========================================================
#
#  "creation"  → Zeit, aus der Gatter-Sequenz einen lauffähigen Zustand
#                herzustellen (Aufbau + Transpile bzw. erster Aufruf).
#  "execution" → Reine Ausführungszeit (statevector) nach dem Aufbau.
#  "gradient"  → Gradienten-Berechnung (⟨Z₀⟩ nach trainierbarer RY-Schicht).
#
BENCHMARK_MODE = "execution"

# Plots am Ende interaktiv anzeigen? Für Batch-/Headless-Läufe auf False setzen
# (die PNGs werden unabhängig davon immer gespeichert).
SHOW_PLOTS = True

# =========================================================
# Gate-Definitionen  (kanonische Namen, kleingeschrieben)
# =========================================================

GATE_SETS = {
    "clifford":               ["h", "s", "cx", "x", "y", "z"],
    "clifford_t":             ["h", "t", "cx"],
    "single_qubit_plus_cnot": ["rx", "ry", "rz", "cx"],
}

VALID_MODES = {"creation", "execution", "gradient"}
VALID_BACKENDS = {"pennylane", "qiskit"}


def resolve_gate_set(choice: str) -> list:
    if choice not in GATE_SETS:
        valid = ", ".join(f'"{k}"' for k in GATE_SETS)
        raise ValueError(f"Unbekanntes Gate-Set '{choice}'. Gültig: {valid}")
    return GATE_SETS[choice]


if BENCHMARK_MODE not in VALID_MODES:
    raise ValueError(f"Unbekannter Modus '{BENCHMARK_MODE}'. Gültig: {VALID_MODES}")
if BACKEND_CHOICE not in VALID_BACKENDS:
    raise ValueError(f"Unbekanntes Backend '{BACKEND_CHOICE}'. Gültig: {VALID_BACKENDS}")

ACTIVE_GATE_SET = resolve_gate_set(GATE_SET_CHOICE)
print(f"Backend  : {BACKEND_CHOICE!r}")
print(f"Gate-Set : {GATE_SET_CHOICE!r}  →  {ACTIVE_GATE_SET}")
print(f"Modus    : {BENCHMARK_MODE!r}\n")

# =========================================================
# Konfiguration  (identisch zum Roh-Benchmark)
# =========================================================

RESULT_DIR = Path(__file__).parent.parent / "Results" / "Executor"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED    = 42
REPEATS = 4

QUBIT_CONFIGS = [5]

GATE_CONFIGS = np.unique(
    np.round(np.logspace(np.log10(10), np.log10(1000), 20)).astype(int)
)

#: Präfixe der zwei Linien – identisch zum Roh-Benchmark, damit die CSVs
#: spaltenweise vergleichbar sind (caching=False vs caching=True).
METHODS = ["qnc", "qc"]

# =========================================================
# Versioned CSV  (Backend im Dateinamen → kein Überschreiben)
# =========================================================

version = 1
while True:
    CSV_FILE = (
        RESULT_DIR
        / f"benchmark_executor_{BACKEND_CHOICE}_{GATE_SET_CHOICE}_{BENCHMARK_MODE}_V{version}.csv"
    )
    if not CSV_FILE.exists():
        break
    version += 1

# =========================================================
# Gate-Eigenschaften  (kanonische Namen)
# =========================================================

_SINGLE_0 = {"h", "s", "t", "x", "y", "z", "sdag", "tdag"}  # 1 Qubit, kein Winkel
_ANGLE_1  = {"rx", "ry", "rz", "p"}                          # 1 Qubit, 1 Winkel
_TWO_Q_0  = {"cx", "cy", "cz", "swap"}                       # 2 Qubit, kein Winkel
_ANGLE_2  = {"crx", "cry", "crz", "cp"}                      # 2 Qubit, 1 Winkel

# =========================================================
# Gate-Sequenz generieren  (identische Logik/Seed wie Roh-Benchmark)
# =========================================================
# Jedes Element: (gate_name, wires, params)

def generate_gate_sequence(
    num_qubits: int,
    total_gates: int,
    gate_set: list,
    seed: int = 42,
) -> list:
    rng = np.random.default_rng(seed)
    sequence = []

    for _ in range(total_gates):
        gate = str(rng.choice(gate_set))
        w    = int(rng.integers(0, num_qubits))

        if gate in _TWO_Q_0 or gate in _ANGLE_2:
            if num_qubits >= 2:
                target = int((w + 1) % num_qubits)
                params = [float(rng.uniform(0, 2 * np.pi))] if gate in _ANGLE_2 else []
                sequence.append((gate, [w, target], params))
            else:
                sequence.append(("h", [w], []))
        elif gate in _ANGLE_1:
            sequence.append((gate, [w], [float(rng.uniform(0, 2 * np.pi))]))
        else:
            sequence.append((gate, [w], []))

    return sequence

# =========================================================
# Abstrakten Schaltkreis bauen
# =========================================================

def apply_abstract(qc: AbstractQuantumCircuit, gate_name: str, wires: list, params: list) -> None:
    if   gate_name in _SINGLE_0: getattr(qc, gate_name)(wires[0])
    elif gate_name in _ANGLE_1:  getattr(qc, gate_name)(wires[0], params[0])
    elif gate_name in _TWO_Q_0:  getattr(qc, gate_name)(wires[0], wires[1])
    elif gate_name in _ANGLE_2:  getattr(qc, gate_name)(wires[0], wires[1], params[0])
    else:
        raise ValueError(f"Unbekanntes Gatter: {gate_name!r}")


def build_abstract(num_qubits: int, sequence: list) -> AbstractQuantumCircuit:
    qc = AbstractQuantumCircuit(num_qubits)
    for gn, ws, ps in sequence:
        apply_abstract(qc, gn, ws, ps)
    return qc


def build_abstract_trainable(
    num_qubits: int, sequence: list, theta: ParameterVector
) -> AbstractQuantumCircuit:
    """Trainierbare RY-Schicht (ein Parameter je Qubit) + Gate-Block."""
    qc = AbstractQuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(i, theta[i])
    for gn, ws, ps in sequence:
        apply_abstract(qc, gn, ws, ps)
    return qc

# =========================================================
# Runner-Builder je Modus
# =========================================================
# Liefert ein dict {methoden_key: run_callable}. Der Aufbau von Executor und
# (abstraktem) Circuit passiert HIER (Setup, nicht gemessen); nur ``run()`` wird
# getimt. Beide Linien bekommen exakt dieselbe (abstrakte) Eingabe – der EINZIGE
# Unterschied ist die Executor-Option caching:
#
#   qnc → caching=False (Standardfall): rechnet bei jedem Aufruf neu
#   qc  → caching=True : Cache-Treffer nach dem Warm-up

def build_runners(mode: str, num_qubits: int, sequence: list) -> dict:
    ex = Executor.create(BACKEND_CHOICE)                     # ohne Ergebnis-Cache
    ex_cached = Executor.create(BACKEND_CHOICE, caching=True)

    # ------------------------------------------------ creation
    if mode == "creation":
        def r_qnc():
            ex.statevector(build_abstract(num_qubits, sequence))

        def r_qc():
            ex_cached.statevector(build_abstract(num_qubits, sequence))

        return {"qnc": r_qnc, "qc": r_qc}

    # ------------------------------------------------ execution
    if mode == "execution":
        qc_abs = build_abstract(num_qubits, sequence)

        def r_qnc():
            ex.statevector(qc_abs)

        def r_qc():
            ex_cached.statevector(qc_abs)

        return {"qnc": r_qnc, "qc": r_qc}

    # ------------------------------------------------ gradient
    if mode == "gradient":
        z0 = "I" * (num_qubits - 1) + "Z"          # ⟨Z₀⟩ (little-endian)
        obs = AbstractQuantumOperator(paulis=[z0], coeffs=[1.0])
        values = np.zeros(num_qubits).tolist()

        theta = ParameterVector("x", num_qubits)
        qc_abs = build_abstract_trainable(num_qubits, sequence, theta)
        op = ex.transpile_operator(obs)

        def r_qnc():
            ex.expectation_value_derivatives(qc_abs, op, "x", **{"x": values})

        def r_qc():
            ex_cached.expectation_value_derivatives(qc_abs, op, "x", **{"x": values})

        return {"qnc": r_qnc, "qc": r_qc}

    raise ValueError(f"Unbekannter Modus '{mode}'.")


MODE_LABELS = {
    "creation":  "Erstellungszeit",
    "execution": "Ausführungszeit",
    "gradient":  "Gradienten-Berechnungszeit",
}

# =========================================================
# Timing  (identisch zum Roh-Benchmark)
# =========================================================

def measure_runtime(func, repeats: int = 5) -> dict:
    func()      # Warm-up
    times = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return {
        "avg": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
    }

# =========================================================
# Speicher-Messung (Peak via tracemalloc)
# =========================================================

def measure_memory(func, repeats: int = 5) -> dict:
    func()      # Warm-up (lazy Imports/Caches nicht mitmessen)
    peaks = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        func()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak / 1024**2)   # MiB
    return {
        "avg": float(np.mean(peaks)),
        "std": float(np.std(peaks)),
        "min": float(np.min(peaks)),
        "max": float(np.max(peaks)),
    }

# =========================================================
# Benchmark-Schleife
# =========================================================

results = []
first_write = True   # Header nur beim allerersten Schreiben

for num_qubits in QUBIT_CONFIGS:
    for total_gates in GATE_CONFIGS:

        print(f"Qubits={num_qubits}  Gates={total_gates}")

        sequence = generate_gate_sequence(
            num_qubits, total_gates, ACTIVE_GATE_SET, seed=SEED
        )

        runners = build_runners(BENCHMARK_MODE, num_qubits, sequence)

        stats = {m: measure_runtime(runners[m], REPEATS) for m in METHODS}
        mem   = {m: measure_memory(runners[m], REPEATS) for m in METHODS}

        print(
            f"  qnc={stats['qnc']['avg']:.5f}s  qc={stats['qc']['avg']:.5f}s"
        )

        row = {
            "timestamp":      datetime.now().isoformat(),
            "backend":        BACKEND_CHOICE,
            "gate_set":       GATE_SET_CHOICE,
            "benchmark_mode": BENCHMARK_MODE,
            "num_qubits":     num_qubits,
            "total_gates":    total_gates,
        }
        for m in METHODS:
            row.update({f"{m}_{k}": v for k, v in stats[m].items()})
            row.update({f"{m}_mem_{k}": v for k, v in mem[m].items()})
        results.append(row)

        # Zwischenergebnis sofort anhängen → überlebt einen Absturz
        pd.DataFrame([row]).to_csv(CSV_FILE, mode="a", header=first_write, index=False)
        first_write = False

# =========================================================
# CSV speichern
# =========================================================

df = pd.DataFrame(results)
print(f"\nErgebnisse gespeichert (inkrementell während des Laufs): {CSV_FILE}")

# =========================================================
# Plot  (2 Linien: caching=False vs caching=True; Layout wie Roh-Benchmark)
# =========================================================

def plot_metric(
    df: pd.DataFrame,
    avg_suffix: str,
    std_suffix: str,
    ylabel: str,
    title: str,
    save_path: Path | None = None,
) -> None:
    qubit_vals = sorted(df["num_qubits"].unique())
    gate_set   = df["gate_set"].iloc[0]

    method_cfg = [
        ("Executor (no cache)", "qnc", "tab:orange"),
        ("Executor (cached)",   "qc",  "tab:green"),
    ]

    n_qubits = len(qubit_vals)
    fig, axes = plt.subplots(1, n_qubits, figsize=(6 * n_qubits, 5), squeeze=False)
    axes = axes[0]

    fig.suptitle(
        f"{title}  |  Backend: {BACKEND_CHOICE!r}  |  Gate-Set: {gate_set!r} "
        f"({', '.join(ACTIVE_GATE_SET)})",
        fontsize=13, fontweight="bold", y=1.02,
    )

    for ax, nq in zip(axes, qubit_vals):
        sub = df[df["num_qubits"] == nq].sort_values("total_gates")
        x   = sub["total_gates"].values

        for method_label, prefix, color in method_cfg:
            y = sub[f"{prefix}_{avg_suffix}"].values
            s = sub[f"{prefix}_{std_suffix}"].values

            ax.plot(x, y, marker="o", linewidth=1.8, markersize=4,
                    label=method_label, color=color)
            ax.fill_between(x, np.maximum(y - s, 1e-9), y + s,
                            alpha=0.18, color=color)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Anzahl Gatter", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{nq} Qubits", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot gespeichert: {save_path}")

    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


mode_label = MODE_LABELS[BENCHMARK_MODE]
_stub = f"benchmark_executor_{BACKEND_CHOICE}_{GATE_SET_CHOICE}_{BENCHMARK_MODE}_V{version}"

# --- Laufzeit ---
plot_metric(
    df, avg_suffix="avg", std_suffix="std",
    ylabel="Zeit (s)", title=mode_label,
    save_path=RESULT_DIR / f"{_stub}.png",
)

# --- Speicherverbrauch (Peak, tracemalloc) ---
plot_metric(
    df, avg_suffix="mem_avg", std_suffix="mem_std",
    ylabel="Peak-Speicher (MiB)", title=f"Speicherverbrauch (Peak) – {mode_label}",
    save_path=RESULT_DIR / f"{_stub}_mem.png",
)