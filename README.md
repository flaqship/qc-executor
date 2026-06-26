# QC Executor

This library provides an abstraction for quantum circuits and operators that can be run on different backends using an `Executor` object.

## Installation

### Core Installation

QC Executor can be installed with only the backends you need. By default, only Qiskit (used as the common intermediate representation) is required:

```bash
pip install git+https://github.com/flaqship/qc-executor.git
```

### Backend-Specific Installation

Install QC Executor with specific backends:

```bash
# Install with PennyLane backend
pip install git+https://github.com/flaqship/qc-executor.git#egg=qc-executor[pennylane]

# Install with Qulacs backend
pip install git+https://github.com/flaqship/qc-executor.git#egg=qc-executor[qulacs]

# Install with full Qiskit support (Aer simulator and IBM Runtime)
pip install git+https://github.com/flaqship/qc-executor.git#egg=qc-executor[qiskit-full]

# Install with Pauli Propagation backend
pip install git+https://github.com/flaqship/qc-executor.git#egg=qc-executor[pauli_propagation]

# Install with all backends
pip install git+https://github.com/flaqship/qc-executor.git#egg=qc-executor[all]

# Install multiple specific backends
pip install git+https://github.com/flaqship/qc-executor.git#egg=qc-executor[pennylane,qulacs,pauli_propagation]
```

## Usage

### Factory API (Recommended)

The `Executor` factory class provides a plugin-based interface for creating executors:

```python
from qc_executor import Executor

# Create an executor for a specific backend
executor = Executor.create("qiskit", shots=1024, seed=42)

# List available backends
backends = Executor.available_backends()
print(backends)  # ['qiskit', 'pennylane', 'qulacs', 'pauli_propagation']

# Create circuit and observable
from qc_executor import QuantumCircuit, QuantumOperator

circuit = QuantumCircuit(2)
circuit.h(0)
circuit.cx(0, 1)

observable = QuantumOperator.from_pauli_string("ZZ")

# Run computation
result = executor.expectation_value(circuit, observable)
print(result)
```

### Backend Switching

You can easily switch between backends while preserving configuration:

```python
# Create executor with specific configuration
qiskit_executor = Executor.create("qiskit", shots=1024, seed=42, caching=True)

# Switch to PennyLane backend with same configuration
pennylane_executor = qiskit_executor.switch_backend("pennylane")

# Switch with configuration overrides
qulacs_executor = qiskit_executor.switch_backend("qulacs", shots=2048)
```

### Direct Import API (Alternative)

For backward compatibility, you can still import executors directly:

```python
from qc_executor.qiskit import QiskitExecutor
from qc_executor.pennylane import PennyLaneExecutor
from qc_executor.qulacs import QulacsExecutor

# Create executor directly
executor = QiskitExecutor(shots=1024, seed=42)
```

### Configuration Options

All executors support the following configuration parameters:

- `shots` (int, optional): Number of measurement shots
- `seed` (int, optional): Random seed for reproducibility
- `log_file` (str, optional): Path to log file
- `log_level` (str, optional): Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
- `caching` (bool, optional): Enable result caching
- `cache_dir` (str, optional): Directory for caching (default: "cache")
- `max_cache_size` (int, optional): Maximum number of cached results (default: unlimited)

### Backend Object Convention

When an executor accepts a backend-specific object (for example a Qiskit backend
instance or a PennyLane `qml.devices.Device`), the constructor parameter name is
`backend`.

```python
from qc_executor.pennylane import PennyLaneExecutor

executor = PennyLaneExecutor(backend="default.mixed", shots=1000, seed=42)
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
