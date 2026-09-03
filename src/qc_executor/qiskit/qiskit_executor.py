"""Qiskit backend executor supporting local simulation and IBM Quantum hardware."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Literal, Tuple

import numpy as np
from qiskit.primitives import (
    StatevectorEstimator,
    StatevectorSampler,
)
from qiskit.providers import Backend
from qiskit.quantum_info import Statevector

from qc_executor.base.circuit_base import QuantumCircuitBase
from qc_executor.base.executor_base import ExecutorBase
from qc_executor.base.operator_base import QuantumOperatorBase
from qc_executor.qiskit.optree import OpTreeDerivative, OpTreeEvaluate
from qc_executor.qiskit.optree.optree import (
    OpTreeCircuit,
    OpTreeList,
    OpTreeNodeBase,
    OpTreeOperator,
)
from qc_executor.qiskit.qiskit_circuit import QiskitCircuit
from qc_executor.qiskit.qiskit_operator import QiskitOperator
from qc_executor.utils.qiskit_compat import (
    QISKIT_RUNTIME_AVAILABLE,
    QISKIT_RUNTIME_SMALLER_0_21,
    QISKIT_RUNTIME_SMALLER_0_23,
    QISKIT_RUNTIME_SMALLER_0_28,
    QISKIT_SMALLER_1_2,
    QISKIT_SMALLER_2_0,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Version-gated Qiskit primitive imports (local simulators + base classes)
# ---------------------------------------------------------------------------
# pylint: disable=import-error,no-name-in-module,ungrouped-imports,too-few-public-methods

if QISKIT_SMALLER_1_2:
    from qiskit.circuit import ParameterExpression as ParameterVectorElement
    from qiskit.primitives import (
        BackendEstimator,
        BackendSampler,
    )
    from qiskit.primitives import BaseEstimator as BaseEstimatorV1
    from qiskit.primitives import BaseSampler as BaseSamplerV1

    class BaseEstimatorV2:
        """Dummy BaseEstimatorV2 for Qiskit < 1.0 compat."""

    class BaseSamplerV2:
        """Dummy BaseSamplerV2 for Qiskit < 1.0 compat."""

elif QISKIT_SMALLER_2_0:
    from qiskit.circuit import ParameterExpression as ParameterVectorElement
    from qiskit.primitives import BackendEstimatorV2 as BackendEstimator
    from qiskit.primitives import BackendSamplerV2 as BackendSampler
    from qiskit.primitives import (
        BaseEstimatorV1,
        BaseEstimatorV2,
        BaseSamplerV1,
        BaseSamplerV2,
    )
else:
    from qiskit.circuit import ParameterVectorElement
    from qiskit.primitives import BackendEstimatorV2 as BackendEstimator
    from qiskit.primitives import BackendSamplerV2 as BackendSampler
    from qiskit.primitives import (
        BaseEstimatorV1,
        BaseEstimatorV2,
        BaseSamplerV1,
        BaseSamplerV2,
    )


# Runtime primitive base classes — used for type annotation and isinstance checks.
# Defined as dummies when qiskit-ibm-runtime is not installed.
if QISKIT_RUNTIME_AVAILABLE:
    if QISKIT_RUNTIME_SMALLER_0_21:
        from qiskit_ibm_runtime import Estimator as RuntimeEstimatorV1
        from qiskit_ibm_runtime import Sampler as RuntimeSamplerV1

        class RuntimeEstimatorV2:
            """Dummy RuntimeEstimatorV2 for runtime < 0.21."""

        class RuntimeSamplerV2:
            """Dummy RuntimeSamplerV2 for runtime < 0.21."""

    elif QISKIT_RUNTIME_SMALLER_0_28:
        from qiskit_ibm_runtime import EstimatorV1 as RuntimeEstimatorV1
        from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2
        from qiskit_ibm_runtime import SamplerV1 as RuntimeSamplerV1
        from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2
    else:
        from qiskit_ibm_runtime import Estimator as RuntimeEstimatorV2
        from qiskit_ibm_runtime import Sampler as RuntimeSamplerV2

        class RuntimeEstimatorV1:
            """Dummy RuntimeEstimatorV1 for runtime >= 0.28."""

        class RuntimeSamplerV1:
            """Dummy RuntimeSamplerV1 for runtime >= 0.28."""

else:

    class RuntimeEstimatorV1:
        """Dummy RuntimeEstimatorV1 — qiskit-ibm-runtime not installed."""

    class RuntimeEstimatorV2:
        """Dummy RuntimeEstimatorV2 — qiskit-ibm-runtime not installed."""

    class RuntimeSamplerV1:
        """Dummy RuntimeSamplerV1 — qiskit-ibm-runtime not installed."""

    class RuntimeSamplerV2:
        """Dummy RuntimeSamplerV2 — qiskit-ibm-runtime not installed."""


# Session and Batch: real imports when runtime is available, dummies otherwise.
if QISKIT_RUNTIME_AVAILABLE:
    from qiskit_ibm_runtime import Batch, Session
else:

    class Session:
        """Dummy Session — qiskit-ibm-runtime not installed."""

        def close(self) -> None:
            """No-op stub."""

        def status(self) -> str:
            """No-op stub."""

    class Batch:
        """Dummy Batch — qiskit-ibm-runtime not installed."""

        def close(self) -> None:
            """No-op stub."""


# pylint: enable=import-error,no-name-in-module,ungrouped-imports,too-few-public-methods


def _reverse_qubit_ordering(statevector: np.ndarray) -> np.ndarray:
    """Reorder statevector amplitudes from Qiskit little-endian to big-endian.

    Qiskit indexes amplitude *i* with q[0] as the LSB, while PennyLane and
    Qulacs index it with q[0] as the MSB.  The conversion is a permutation
    that reverses the bit-pattern of each index.

    Example for 2 qubits::

        Qiskit index 1 = 0b01 → reversed = 0b10 = 2  (q[0]=1, q[1]=0)
        Qiskit index 2 = 0b10 → reversed = 0b01 = 1  (q[0]=0, q[1]=1)
    """
    n = int(np.log2(len(statevector)))
    perm = [int(format(i, f"0{n}b")[::-1], 2) for i in range(len(statevector))]
    return statevector[perm]


def _reverse_bitstring(bitstring: str) -> str:
    """Reverse a Qiskit measurement bitstring to big-endian qubit ordering.

    Qiskit writes bitstrings as ``q[n-1]...q[1]q[0]`` (q[0] rightmost).
    PennyLane / Qulacs write ``q[0]q[1]...q[n-1]`` (q[0] leftmost).
    """
    return bitstring[::-1]


def _convert_counts_endianness(counts: dict) -> dict:
    """Return a new counts dict with all bitstrings converted to big-endian."""
    result: dict = {}
    for bitstring, count in counts.items():
        key = _reverse_bitstring(bitstring)
        result[key] = result.get(key, 0) + count
    return result


# ---------------------------------------------------------------------------
# Lazy-loaded helpers for optional dependencies
# ---------------------------------------------------------------------------


def _load_aer_simulator():
    try:
        from qiskit_aer import AerSimulator  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError(
            "qiskit-aer is required for 'backend=\"aer\"' and for shot-based "
            "sampling with 'backend=\"statevector\"'. Install with: "
            "pip install qc-executor[qiskit-full]"
        ) from e
    return AerSimulator


def _check_runtime_available():
    """Guard: raise :class:`ImportError` when *qiskit-ibm-runtime* is missing."""
    if not QISKIT_RUNTIME_AVAILABLE:
        raise ImportError(
            "qiskit-ibm-runtime is required for IBM backend support. "
            "Install with: pip install qc-executor[qiskit-full]"
        )


# pylint: disable=import-outside-toplevel,import-error,no-name-in-module,redefined-outer-name,reimported


def _load_runtime_primitives_v1():
    """Return ``(RuntimeEstimatorV1, RuntimeSamplerV1)`` for *qiskit-ibm-runtime < 0.21*."""
    _check_runtime_available()
    from qiskit_ibm_runtime import Estimator as RuntimeEstimatorV1
    from qiskit_ibm_runtime import Sampler as RuntimeSamplerV1

    return RuntimeEstimatorV1, RuntimeSamplerV1


def _load_runtime_primitives_v2():
    """Return ``(RuntimeEstimatorV2, RuntimeSamplerV2)`` from *qiskit-ibm-runtime*.

    * ``0.21 – 0.27``: explicit ``EstimatorV2`` / ``SamplerV2`` exports.
    * ``>= 0.28``: ``Estimator`` / ``Sampler`` **are** the V2 primitives.
    """
    _check_runtime_available()
    if QISKIT_RUNTIME_SMALLER_0_28:
        from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2
        from qiskit_ibm_runtime import SamplerV2 as RuntimeSamplerV2
    else:
        from qiskit_ibm_runtime import Estimator as RuntimeEstimatorV2
        from qiskit_ibm_runtime import Sampler as RuntimeSamplerV2
    return RuntimeEstimatorV2, RuntimeSamplerV2


def _load_runtime_options_v1():
    """Return the ``Options`` class for V1 runtime primitives (< 0.21)."""
    _check_runtime_available()
    from qiskit_ibm_runtime.options import Options

    return Options


def _load_runtime_session():
    """Return the ``Session`` class from *qiskit-ibm-runtime*."""
    _check_runtime_available()
    from qiskit_ibm_runtime import Session

    return Session


def _load_runtime_batch():
    """Return the ``Batch`` class from *qiskit-ibm-runtime*."""
    _check_runtime_available()
    from qiskit_ibm_runtime import Batch

    return Batch


def _is_backend_instance(obj) -> bool:
    """Return *True* if *obj* is a Qiskit ``Backend`` (V1 or V2) instance."""
    try:
        from qiskit.providers import Backend

        return isinstance(obj, Backend)
    except ImportError:
        return False


def _is_session_or_batch_instance(obj) -> bool:
    """Return *True* if *obj* is a ``qiskit_ibm_runtime.Session`` or ``Batch``."""
    if not QISKIT_RUNTIME_AVAILABLE:
        return False
    try:
        from qiskit_ibm_runtime import Session

        if isinstance(obj, Session):
            return True
    except ImportError:
        pass
    try:
        from qiskit_ibm_runtime import Batch

        if isinstance(obj, Batch):
            return True
    except ImportError:
        pass
    return False


# pylint: enable=import-outside-toplevel,import-error,no-name-in-module,redefined-outer-name,reimported


def _resolve_backend_from_session_or_batch(session_or_batch):
    """Best-effort backend extraction from runtime Session/Batch.

    Supports both direct backend objects and backend-name strings returned by
    some runtime versions via ``backend()``.
    """
    backend = getattr(session_or_batch, "_backend", None)
    if _is_backend_instance(backend):
        return backend

    backend_value = None
    if hasattr(session_or_batch, "backend") and callable(getattr(session_or_batch, "backend")):
        try:
            backend_value = session_or_batch.backend()
        except Exception:  # pylint: disable=broad-exception-caught
            backend_value = None

    if _is_backend_instance(backend_value):
        return backend_value

    if isinstance(backend_value, str):
        service = getattr(session_or_batch, "service", None)
        if service is not None and hasattr(service, "backend"):
            try:
                resolved = service.backend(backend_value)
                if _is_backend_instance(resolved):
                    return resolved
            except Exception:  # pylint: disable=broad-exception-caught
                return None

    return None


def _is_primitive_instance(obj) -> Tuple[bool, bool]:
    """Detect whether *obj* is a Qiskit primitive."""
    # Qiskit base classes (covers BackendEstimator/Sampler and StatevectorEstimator/Sampler)
    if isinstance(obj, (BaseSamplerV1, BaseSamplerV2)):
        return (True, True)
    if isinstance(obj, (BaseEstimatorV1, BaseEstimatorV2)):
        return (True, False)

    # Qiskit concrete statevector primitives
    if isinstance(obj, StatevectorSampler):
        return (True, True)
    if isinstance(obj, StatevectorEstimator):
        return (True, False)

    # IBM Runtime primitives — may NOT inherit from Qiskit base classes
    # (especially runtime >= 0.28 where Estimator/Sampler are the V2 primitives)
    if isinstance(obj, (RuntimeSamplerV1, RuntimeSamplerV2)):
        return (True, True)
    if isinstance(obj, (RuntimeEstimatorV1, RuntimeEstimatorV2)):
        return (True, False)

    return (False, False)


def _classify_backend(backend) -> Tuple[bool, bool, bool]:
    """Classify backend properties in a single pass.

    Returns
    -------
    (remote, ibm_quantum, ibm_fake) : Tuple[bool, bool, bool]
        ``remote`` is *True* for real remote backends.
        ``ibm_quantum`` is *True* only for real IBM Quantum hardware.
        ``ibm_fake`` is *True* only for IBM fake backends.
    """
    # 1. Real IBM hardware
    try:
        from qiskit_ibm_runtime import (  # pylint: disable=import-outside-toplevel,import-error
            IBMBackend,
        )

        if isinstance(backend, IBMBackend):
            return (True, True, False)
    except ImportError:
        pass

    # 2. IBM fake backend
    try:
        from qiskit_ibm_runtime.fake_provider.fake_backend import (  # pylint: disable=import-outside-toplevel,import-error
            FakeBackendV2,
        )

        if isinstance(backend, FakeBackendV2):
            return (False, False, True)
    except ImportError:
        pass

    # 3. Generic local fake backend
    try:
        from qiskit.providers.fake_provider import (  # pylint: disable=import-outside-toplevel
            GenericBackendV2,
        )

        if isinstance(backend, GenericBackendV2):
            return (False, False, False)
    except ImportError:
        pass

    # 4. Fallback: string matching for unrecognised third-party backends
    backend_str = str(backend).lower()
    if "ibm" in backend_str:
        is_fake = "fake" in backend_str
        logger.warning(
            "Backend type %r not recognised via isinstance checks; "
            "falling back to string matching. Consider filing a bug report.",
            type(backend).__name__,
        )
        return (not is_fake, not is_fake, is_fake)

    return (False, False, False)


class QiskitExecutor(ExecutorBase):
    """Class for executing Qiskit circuits.

    Supports local simulation (``"statevector"``, ``"aer_statevector"``,
    ``"aer"``) **and** execution on real IBM Quantum hardware or noise-aware
    fake backends via
    `qiskit-ibm-runtime <https://github.com/Qiskit/qiskit-ibm-runtime>`_.

    The ``backend`` parameter is the single entry point and accepts all
    supported configurations:

    * ``"statevector"`` — Qiskit's reference primitives
      (``StatevectorEstimator``/``StatevectorSampler``): exact for
      ``shots=None``, otherwise an analytic Gaussian noise model on top of
      the exact statevector.
    * ``"aer_statevector"`` — real shot-based sampling of the statevector via
      ``AerSimulator(method="statevector")``; a different, not necessarily
      numerically equivalent estimator/sampler than ``"statevector"`` with
      shots set, despite both targeting the same exact state.
    * ``"aer"`` — the general-purpose ``AerSimulator``.
    * A :class:`~qiskit.providers.Backend` / ``BackendV2`` instance (e.g.
      from ``QiskitRuntimeService`` or ``fake_provider``).
    * A ``qiskit_ibm_runtime.Session`` or ``Batch`` — ownership is transferred
      to the executor, which closes it on exit.
    * A pre-configured Qiskit primitive (``BaseSamplerV2`` / ``BaseEstimatorV2``
      or their V1 equivalents) — injected directly; the missing counterpart
      primitive is created automatically when possible.

    Context-manager use is **strongly recommended** for real IBM backends
    to guarantee that sessions are properly closed::

        with QiskitExecutor(backend=ibm_backend, execution_mode="session") as exe:
            result = exe.expectation_value(circuit, observable, theta=params)

    Args:
        backend: Backend to use for execution.  Accepts:
            ``"statevector"`` (default), ``"aer_statevector"`` or ``"aer"``
            string shortcuts, a Qiskit
            :class:`~qiskit.providers.Backend` instance (IBM hardware or fake),
            a ``qiskit_ibm_runtime.Session`` / ``Batch``, or a pre-configured
            Qiskit primitive (``BaseSamplerV1/V2`` / ``BaseEstimatorV1/V2``).
        shots (int | None, optional): Number of shots for sampling.
        seed (int | None, optional): Random seed for reproducibility.
        log_file (str | None, optional): Path to the log file.
        log_level (str, optional): Logging level.
        caching (bool | None, optional): Whether to use in-memory caching.
        cache_dir (str, optional): Directory for caching.
        max_cache_size (int | None, optional): Maximum number of entries kept
            in each in-memory cache; ``None`` makes them unbounded.
        execution_mode (str, optional): ``"job"`` (default), ``"session"``, or
            ``"batch"``.  Only relevant for real IBM Quantum backends.
            Use ``"session"`` for iterative algorithms (VQE, QAOA) and
            ``"batch"`` for independent parallel jobs.
        options (dict | None, optional): Options forwarded to IBM Runtime primitives
            (e.g. ``{"resilience_level": 1}``).  Ignored for local backends.
        primitive_wrapper (callable | None, optional): ``wrapper(primitive, kind)
            -> primitive``, with ``kind`` either ``"estimator"`` or ``"sampler"``.
            Applied to every primitive this executor (re)builds - on initial
            construction, after a runtime session renewal, and for lazily
            deferred session primitives - so a host application can layer
            cross-cutting concerns (retry, disk caching, seeding, ...) on top
            of execution without ever holding a primitive itself. The
            wrapper's return value must be a genuine instance of the same
            base class as the primitive it was given (``BaseEstimatorV1``/
            ``BaseEstimatorV2`` for an estimator, ``BaseSamplerV1``/
            ``BaseSamplerV2`` for a sampler) - qc_executor's own OpTree
            evaluation dispatches on ``isinstance()`` of exactly those
            classes, so duck-typing ``run()`` alone is not enough. See
            :attr:`raw_estimator`/:attr:`raw_sampler` for the underlying,
            undecorated primitive.
    """

    _native_circuit_class = QiskitCircuit
    _native_operator_class = QiskitOperator

    def __init__(
        self,
        backend: (
            str
            | Backend
            | Session
            | Batch
            | BaseEstimatorV1
            | BaseSamplerV1
            | BaseEstimatorV2
            | BaseSamplerV2
        ) = "statevector",
        shots: int | None = None,
        seed: int | None = None,
        log_file: str | None = None,
        log_level: str = "WARNING",
        caching: bool | None = None,
        cache_dir: str = "cache",
        max_cache_size: int | None = 4096,
        execution_mode: Literal["job", "session", "batch"] = "job",
        options: dict | None = None,
        primitive_wrapper: Callable[[Any, str], Any] | None = None,
    ):
        super().__init__(
            shots=shots,
            seed=seed,
            log_file=log_file,
            log_level=log_level,
            caching=caching,
            cache_dir=cache_dir,
            max_cache_size=max_cache_size,
        )

        # Internal state for IBM backend support
        self._session = None
        self._inside_context_manager: bool = False
        self._remote_backend: bool = False
        self._ibm_quantum_backend: bool = False
        self._execution_mode = execution_mode
        self._options = options
        self._primitive_wrapper = primitive_wrapper
        # The raw (undecorated) primitives this executor built - always the
        # genuine Qiskit object, regardless of what primitive_wrapper turns
        # self._estimator/self._sampler (the ones actually used for
        # execution) into. See _decorate_primitive().
        self._raw_estimator = None
        self._raw_sampler = None
        # Safe defaults — overwritten in the relevant branches below
        self._runtime_primitives_version: str = "v2"
        self._sampler_uses_v1_api: bool = QISKIT_SMALLER_1_2
        self._isa_transpile: bool = False

        # ── 1. Direct primitive injection via backend ──────────────────────
        # User passes a pre-configured Sampler or Estimator directly as the
        # backend.  The missing counterpart primitive is auto-created.
        is_primitive, is_sampler = _is_primitive_instance(backend)
        if is_primitive and options is not None:
            raise ValueError(
                "Ambiguous initialization: 'options' cannot be combined with injected "
                "primitives. Configure the primitive objects directly."
            )
        if (is_primitive or _is_session_or_batch_instance(backend)) and execution_mode != "job":
            raise ValueError(
                "Ambiguous initialization: 'execution_mode' applies only to backend-based "
                "initialization and must remain 'job' when passing a Session/Batch/primitive."
            )

        if is_primitive:
            self._backend = None
            self._session = None

            mode = getattr(backend, "_mode", None)
            if _is_backend_instance(mode):
                self._backend = mode
            elif _is_session_or_batch_instance(mode):
                self._session = mode

            if is_sampler:
                self._sampler = backend
                logger.info(
                    "QiskitExecutor initialised with user-provided Sampler (%s).",
                    type(backend).__name__,
                )
            else:
                self._estimator = backend
                logger.info(
                    "QiskitExecutor initialised with user-provided Estimator (%s).",
                    type(backend).__name__,
                )

            # Extract backend and session from runtime primitives so that
            # _classify_backend, ISA transpilation, and session awareness
            # work correctly even when primitives are injected directly.
            # This mirrors the approach used in the sQUlearn Executor.
            if isinstance(backend, (RuntimeEstimatorV2, RuntimeSamplerV2)):
                self._backend = self._backend or getattr(backend, "_backend", None)
                # Runtime >= 0.23 uses _mode; older versions use _session.
                self._session = self._session or getattr(backend, "_session", None)
            elif isinstance(backend, (RuntimeEstimatorV1, RuntimeSamplerV1)):
                self._backend = self._backend or getattr(backend, "_backend", None)
                self._session = self._session or getattr(backend, "_session", None)
            elif hasattr(backend, "backend"):
                # BackendEstimatorV2 / BackendSamplerV2 expose .backend as a property
                self._backend = self._backend or backend.backend
            elif hasattr(backend, "_backend"):
                # BackendEstimatorV1 / BackendSamplerV1 use ._backend
                self._backend = self._backend or backend._backend

            if self._backend is not None:
                self._remote_backend, self._ibm_quantum_backend, _ = _classify_backend(
                    self._backend
                )
                self._isa_transpile = self._ibm_quantum_backend

            # Determine runtime primitive generation for counterpart creation.
            self._runtime_primitives_version = (
                "v1" if isinstance(backend, (BaseEstimatorV1, BaseSamplerV1)) else "v2"
            )

            # Auto-create missing counterpart primitive from same context.
            if is_sampler:
                if isinstance(backend, StatevectorSampler):
                    self._estimator = StatevectorEstimator()
                elif isinstance(backend, (RuntimeSamplerV1, RuntimeSamplerV2)):
                    self._estimator = self._create_runtime_estimator()
                elif self._backend is not None:
                    self._estimator = BackendEstimator(backend=self._backend)
                else:
                    # No context available — leave estimator unset; the user
                    # may only need the sampler.
                    self._estimator = None
            else:
                if isinstance(backend, StatevectorEstimator):
                    self._sampler = StatevectorSampler()
                elif isinstance(backend, (RuntimeEstimatorV1, RuntimeEstimatorV2)):
                    self._sampler = self._create_runtime_sampler()
                elif self._backend is not None:
                    self._sampler = BackendSampler(backend=self._backend)
                else:
                    # No context available — leave sampler unset; the user
                    # may only need the estimator.
                    self._sampler = None

            # V1 vs V2 sampler API detection should be based on sampler instance.
            self._sampler_uses_v1_api = isinstance(self._sampler, BaseSamplerV1)

        # ── 2. Injected Session / Batch ────────────────────────────────────
        # Ownership is intentionally transferred to the executor: close_session()
        # will close even externally created objects.
        elif _is_session_or_batch_instance(backend):
            _check_runtime_available()
            self._session = backend
            # Retrieve the target backend from the session for ISA transpilation
            self._backend = _resolve_backend_from_session_or_batch(backend)
            if self._backend is None:
                self._backend = None
                logger.warning(
                    "Could not retrieve backend from %s; ISA transpilation will be skipped.",
                    type(backend).__name__,
                )
            if self._backend is not None:
                self._remote_backend, self._ibm_quantum_backend, _ = _classify_backend(
                    self._backend
                )
            else:
                self._remote_backend = False
                self._ibm_quantum_backend = False
            self._isa_transpile = self._backend is not None
            self._runtime_primitives_version = "v1" if QISKIT_RUNTIME_SMALLER_0_21 else "v2"
            self._estimator = self._create_runtime_estimator()
            self._sampler = self._create_runtime_sampler()
            self._sampler_uses_v1_api = self._runtime_primitives_version == "v1"
            logger.info(
                "QiskitExecutor attached to injected %s (ownership transferred).",
                type(backend).__name__,
            )

        # ── 3. Local simulator backends (string shortcuts) ────────────────
        elif isinstance(backend, str):
            if backend == "statevector":
                # Qiskit's reference primitives - exact for shots=None, else an
                # analytic Gaussian noise model on top of the exact statevector
                # (default_precision / default_shots). No Aer required.
                self._estimator = StatevectorEstimator()
                self._sampler = StatevectorSampler()
                self._backend = None
            elif backend == "aer_statevector":
                # Real shot-based sampling of the statevector via Aer, as
                # opposed to "statevector"'s analytic noise model.
                aer_simulator_cls = _load_aer_simulator()
                self._backend = aer_simulator_cls(method="statevector")
                self._estimator = BackendEstimator(backend=self._backend)
                self._sampler = BackendSampler(backend=self._backend)
            elif backend == "aer":
                aer_simulator_cls = _load_aer_simulator()
                self._backend = aer_simulator_cls()
                self._estimator = BackendEstimator(backend=self._backend)
                self._sampler = BackendSampler(backend=self._backend)
            else:
                raise ValueError(
                    f"Unknown backend string: {backend!r}. "
                    "Use 'statevector', 'aer_statevector', 'aer', or pass a "
                    "Backend / Session instance."
                )

        # ── 4. Backend object (IBMBackend / FakeBackend / any BackendV2) ──
        elif _is_backend_instance(backend):
            self._backend = backend
            (
                self._remote_backend,
                self._ibm_quantum_backend,
                is_ibm_fake,
            ) = _classify_backend(backend)
            self._isa_transpile = True

            # Primitive creation strategy:
            # - Real IBM hardware:  always use runtime primitives (V1 or V2).
            # - IBM fake backends:  use runtime primitives if available (>= 0.21);
            #   fall back to local BackendEstimator/BackendSampler for runtime < 0.21
            #   because V1 runtime primitives require an active IBM account.
            # - Non-IBM backends (AerSimulator, etc.): use local primitives only;
            #   qiskit-ibm-runtime is not required.
            needs_runtime = self._ibm_quantum_backend or is_ibm_fake
            if needs_runtime and QISKIT_RUNTIME_AVAILABLE:
                self._runtime_primitives_version = "v1" if QISKIT_RUNTIME_SMALLER_0_21 else "v2"
                if self._runtime_primitives_version == "v1" and not self._ibm_quantum_backend:
                    # V1 runtime requires IBM account even for fakes — use local fallback
                    self._estimator = BackendEstimator(backend=self._backend)
                    self._sampler = BackendSampler(backend=self._backend)
                    self._sampler_uses_v1_api = QISKIT_SMALLER_1_2
                else:
                    _check_runtime_available()
                    if self._uses_managed_session():
                        # Delay session + primitive creation until the first
                        # real execution or context-manager entry. This avoids
                        # noisy warnings for recommended ``with`` usage.
                        self._estimator = None
                        self._sampler = None
                    else:
                        self._estimator = self._create_runtime_estimator()
                        self._sampler = self._create_runtime_sampler()
                    self._sampler_uses_v1_api = self._runtime_primitives_version == "v1"
            elif needs_runtime and not QISKIT_RUNTIME_AVAILABLE and self._ibm_quantum_backend:
                # Real IBM hardware always needs runtime
                _check_runtime_available()
            else:
                # Generic / non-IBM backend — no runtime required
                self._estimator = BackendEstimator(backend=self._backend)
                self._sampler = BackendSampler(backend=self._backend)
                self._sampler_uses_v1_api = QISKIT_SMALLER_1_2

            logger.info(
                "Initialised QiskitExecutor with %s (remote=%s, mode=%s).",
                backend,
                self._remote_backend,
                execution_mode,
            )

        else:
            raise TypeError(
                f"'backend' must be a string ('statevector', 'aer_statevector', 'aer'), "
                f"a Qiskit Backend "
                f"instance, a qiskit-ibm-runtime Session/Batch, or a Qiskit primitive "
                f"(BaseSamplerV1/V2 / BaseEstimatorV1/V2). Got {type(backend)!r}."
            )

        if seed is not None:
            self._random = np.random.default_rng(seed)
        else:
            self._random = np.random.default_rng()

        if not is_primitive:
            # A directly injected primitive (branch 1) is caller-configured and
            # left untouched; every other construction path built its primitives
            # from a bare backend/string/session, which never carried any shots
            # information on its own.
            self._apply_shots_option(self._estimator)
            self._apply_shots_option(self._sampler)

        # Whichever branch above built them, self._estimator/self._sampler are
        # still the raw Qiskit primitives at this point - snapshot them, then
        # apply the host's wrapper (if any) for actual execution use.
        self._raw_estimator = self._estimator
        self._raw_sampler = self._sampler
        self._estimator = self._decorate_primitive(self._raw_estimator, "estimator")
        self._sampler = self._decorate_primitive(self._raw_sampler, "sampler")

    def _decorate_primitive(self, primitive, kind: str):
        """Apply the host's ``primitive_wrapper`` (if any) to a freshly (re)built
        primitive - see the ``primitive_wrapper`` constructor argument. Called
        for every construction path: initial ``__init__``, a runtime session
        renewal (:meth:`_refresh_primitives`), and lazily deferred session
        primitives (:meth:`_ensure_runtime_primitives`).
        """
        if primitive is None or self._primitive_wrapper is None:
            return primitive
        return self._primitive_wrapper(primitive, kind)

    def _apply_shots_option(self, primitive) -> None:
        """Configure *primitive* so it honors ``self._shots``.

        Estimators express this as precision (``1 / sqrt(shots)``); samplers
        and V1-style primitives take the shot count directly. Left untouched
        when ``self._shots`` is ``None`` (exact simulation, already each
        primitive's own default).
        """
        if primitive is None or self._shots is None:
            return
        if isinstance(primitive, (RuntimeEstimatorV1, RuntimeSamplerV1)):
            execution = primitive.options.get("execution") or {}
            execution["shots"] = self._shots
            primitive.set_options(execution=execution)
        elif isinstance(primitive, (BaseEstimatorV1, BaseSamplerV1)):
            primitive.set_options(shots=self._shots)
        elif isinstance(primitive, StatevectorEstimator):
            # No .options - default_precision is a plain, directly settable
            # attribute (backed by _default_precision).
            primitive._default_precision = 1.0 / self._shots**0.5
        elif isinstance(primitive, StatevectorSampler):
            primitive._default_shots = self._shots
        elif hasattr(primitive, "options"):
            if hasattr(primitive.options, "default_precision"):
                primitive.options.default_precision = 1.0 / self._shots**0.5
            elif hasattr(primitive.options, "default_shots"):
                primitive.options.default_shots = self._shots

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shots(self) -> int | None:
        """Return the number of shots."""
        return self._shots

    @shots.setter
    def shots(self, value: int | None) -> None:
        """Set the number of shots and apply it to the current primitives in
        place, so the change takes effect on the very next ``run()`` without
        needing a rebuild - including for a primitive a host's
        ``primitive_wrapper`` still holds a reference to, since that
        reference is to the same raw object being mutated here.
        """
        self._shots = value
        self._apply_shots_option(self.raw_estimator)
        self._apply_shots_option(self.raw_sampler)

    @property
    def remote(self) -> bool:
        """Return ``True`` if the executor targets a remote backend."""
        return self._remote_backend

    @property
    def ibm_quantum(self) -> bool:
        """Return ``True`` if the executor targets a real IBM Quantum device."""
        return self._ibm_quantum_backend

    @property
    def session(self):
        """Return the active runtime Session or Batch, or ``None``."""
        return self._session

    @property
    def estimator(self):
        """Return the Qiskit estimator primitive used for execution.

        For real IBM Quantum hardware, this refreshes the underlying runtime
        session and primitives first if the session has expired since the
        last access - safe to call repeatedly across a long-running loop
        without needing a ``with`` block.
        """
        if self._ibm_quantum_backend:
            self._ensure_primitives_current()
        return self._estimator

    @property
    def sampler(self):
        """Return the Qiskit sampler primitive used for execution.

        For real IBM Quantum hardware, this refreshes the underlying runtime
        session and primitives first if the session has expired since the
        last access - safe to call repeatedly across a long-running loop
        without needing a ``with`` block.
        """
        if self._ibm_quantum_backend:
            self._ensure_primitives_current()
        return self._sampler

    @property
    def raw_estimator(self):
        """Return the raw (undecorated) Qiskit estimator primitive.

        Unlike :attr:`estimator`, which returns whatever a registered
        ``primitive_wrapper`` turned the primitive into (see the
        constructor), this is always the genuine Qiskit primitive
        qc_executor itself constructed - useful for host-side introspection
        (e.g. reading its concrete type or default options).
        """
        if self._ibm_quantum_backend:
            self._ensure_primitives_current()
        return self._raw_estimator

    @property
    def raw_sampler(self):
        """Return the raw (undecorated) Qiskit sampler primitive.

        Unlike :attr:`sampler`, which returns whatever a registered
        ``primitive_wrapper`` turned the primitive into (see the
        constructor), this is always the genuine Qiskit primitive
        qc_executor itself constructed - useful for host-side introspection
        (e.g. reading its concrete type or default options).
        """
        if self._ibm_quantum_backend:
            self._ensure_primitives_current()
        return self._raw_sampler

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _uses_managed_session(self) -> bool:
        """Return *True* when this executor should own a runtime Session/Batch."""
        return self._ibm_quantum_backend and self._execution_mode in ("session", "batch")

    def create_session(self) -> None:
        """Explicitly (re)create the runtime session now, instead of waiting
        for the first execution to trigger it lazily.

        A no-op when this executor doesn't manage its own session (local
        simulators, fake backends, or real IBM Quantum hardware in ``"job"``
        execution mode all build their primitives without one).
        """
        if not self._uses_managed_session():
            return
        self._create_session()
        self._refresh_primitives()

    def _create_session(self) -> None:
        """Create (or re-create) a :class:`~qiskit_ibm_runtime.Session` or
        :class:`~qiskit_ibm_runtime.Batch` depending on ``execution_mode``.

        ``"batch"`` mode uses ``Batch`` for independent, parallelisable jobs.
        ``"session"`` mode uses ``Session`` for iterative algorithms (VQE/QAOA)
        that require tight coupling between successive jobs.
        """
        if self._backend is None:
            raise RuntimeError("Cannot create a runtime session without a backend.")

        if self._execution_mode == "batch":
            batch_cls = _load_runtime_batch()
            self._session = batch_cls(backend=self._backend)
            logger.debug("Created new runtime Batch for %s.", self._backend)
        else:
            session_cls = _load_runtime_session()
            self._session = session_cls(backend=self._backend)
            logger.debug("Created new runtime Session for %s.", self._backend)

        if not self._inside_context_manager:
            logger.warning(
                "IBM Runtime %s opened outside of a context manager. "
                "Use 'with QiskitExecutor(...) as exe:' to ensure the session "
                "is closed when done.",
                type(self._session).__name__,
            )

    def close_session(self) -> None:
        """Close the current runtime session/batch if one is active.

        Session ownership is always managed by the executor, including
        injected ``Session`` / ``Batch`` objects.
        """
        if self._session is None:
            return

        try:
            self._session.close()
            logger.info("Closed IBM Runtime %s.", type(self._session).__name__)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.debug(
                "%s.close() raised; ignoring.",
                type(self._session).__name__,
                exc_info=True,
            )
        finally:
            self._session = None

    def __enter__(self):
        """Support ``with QiskitExecutor(...) as exe:`` usage.

        For real IBM backends this ensures sessions are properly closed on
        exit.  Context-managed use is strongly recommended whenever
        ``execution_mode`` is ``"session"`` or ``"batch"``.
        """
        self._inside_context_manager = True
        if self._uses_managed_session() and self._session is None:
            self._create_session()
            self._refresh_primitives()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._inside_context_manager = False
        self.close_session()
        return False

    def __del__(self):
        """Attempt to close any open session when the executor is garbage-collected."""
        session = getattr(self, "_session", None)
        if session is not None:
            try:
                session.close()
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    # ------------------------------------------------------------------
    # Runtime primitive factories
    # ------------------------------------------------------------------

    def _ensure_session_active(self) -> None:
        """Re-create the session when it has expired.

        Uses the public ``Session.status()`` API rather than private
        ``_active`` attributes, which are not guaranteed across runtime
        versions.
        """
        if not self._ibm_quantum_backend:
            return

        if self._uses_managed_session() and self._session is None:
            self._create_session()
            return

        if self._session is None:
            return
        try:
            status = self._session.status()
            if status not in ("open", "pending_new"):
                logger.info(
                    "%s expired (status=%r), recreating.",
                    type(self._session).__name__,
                    status,
                )
                self._create_session()
        except Exception:  # pylint: disable=broad-exception-caught
            logger.debug(
                "Could not check %s status; assuming still active.",
                type(self._session).__name__,
                exc_info=True,
            )

    def _ensure_primitives_current(self) -> None:
        """Rebuild the estimator/sampler if the runtime session was just renewed.

        ``_ensure_session_active()`` transparently re-creates an expired
        session but leaves any already-constructed estimator/sampler bound to
        the old (now-closed) one - callers that only build on first use (like
        :meth:`_ensure_runtime_primitives`) never notice a session renewed
        mid-lifetime. Comparing session identity before/after is what lets a
        long-running loop that never re-enters ``__enter__`` keep working
        across a session expiry.
        """
        session_before = self._session
        self._ensure_session_active()
        if (
            self._raw_estimator is None
            or self._raw_sampler is None
            or self._session is not session_before
        ):
            self._refresh_primitives()

    def _create_runtime_estimator(self):
        """Instantiate the runtime Estimator for the current backend / session."""
        self._ensure_session_active()
        if self._runtime_primitives_version == "v1":
            cls, _ = _load_runtime_primitives_v1()
            estimator = self._instantiate_runtime_primitive_v1(cls, self._options)
        else:
            cls, _ = _load_runtime_primitives_v2()
            estimator = self._instantiate_runtime_primitive_v2(cls, self._options)
        self._apply_shots_option(estimator)
        return estimator

    def _create_runtime_sampler(self):
        """Instantiate the runtime Sampler for the current backend / session."""
        self._ensure_session_active()
        if self._runtime_primitives_version == "v1":
            _, cls = _load_runtime_primitives_v1()
            sampler = self._instantiate_runtime_primitive_v1(cls, self._options)
        else:
            _, cls = _load_runtime_primitives_v2()
            sampler = self._instantiate_runtime_primitive_v2(cls, self._options)
        self._apply_shots_option(sampler)
        return sampler

    # -- V1 instantiation (qiskit-ibm-runtime < 0.21) ---------------------

    def _instantiate_runtime_primitive_v1(self, primitive_cls, options):
        """Create a V1 runtime primitive (``qiskit-ibm-runtime < 0.21``)."""
        if options:
            runtime_options_v1_cls = _load_runtime_options_v1()
            opts = runtime_options_v1_cls()  # pylint: disable=not-callable
            for key, val in options.items():
                try:
                    setattr(opts, key, val)
                except (AttributeError, TypeError):
                    pass
        else:
            opts = None

        if self._ibm_quantum_backend and self._session is not None:
            self._ensure_session_active()
            return (
                primitive_cls(session=self._session, options=opts)
                if opts is not None
                else primitive_cls(session=self._session)
            )
        return (
            primitive_cls(backend=self._backend, options=opts)
            if opts is not None
            else primitive_cls(backend=self._backend)
        )

    # -- V2 instantiation (qiskit-ibm-runtime >= 0.21) --------------------

    def _instantiate_runtime_primitive_v2(self, primitive_cls, options):
        """Create a V2 runtime primitive.

        * ``0.21 – 0.22``: uses ``session=`` / ``backend=`` kwargs.
        * ``>= 0.23``: uses ``mode=`` (accepts both Session and backend).
        """
        opts = options or {}

        if self._ibm_quantum_backend and self._session is not None:
            self._ensure_session_active()
            if QISKIT_RUNTIME_SMALLER_0_23:
                return (
                    primitive_cls(session=self._session, options=opts)
                    if opts
                    else primitive_cls(session=self._session)
                )
            return (
                primitive_cls(mode=self._session, options=opts)
                if opts
                else primitive_cls(mode=self._session)
            )
        # Fake backend or real backend in job mode (no session)
        if QISKIT_RUNTIME_SMALLER_0_23:
            return (
                primitive_cls(backend=self._backend, options=opts)
                if opts
                else primitive_cls(backend=self._backend)
            )
        return (
            primitive_cls(mode=self._backend, options=opts)
            if opts
            else primitive_cls(mode=self._backend)
        )

    def _refresh_primitives(self) -> None:
        """Re-create primitives after a session renewal."""
        self._raw_estimator = self._create_runtime_estimator()
        self._raw_sampler = self._create_runtime_sampler()
        self._estimator = self._decorate_primitive(self._raw_estimator, "estimator")
        self._sampler = self._decorate_primitive(self._raw_sampler, "sampler")

    def _ensure_runtime_primitives(self) -> None:
        """Lazily create runtime primitives when session management is deferred."""
        if self._raw_estimator is None:
            self._raw_estimator = self._create_runtime_estimator()
            self._estimator = self._decorate_primitive(self._raw_estimator, "estimator")
        if self._raw_sampler is None:
            self._raw_sampler = self._create_runtime_sampler()
            self._sampler = self._decorate_primitive(self._raw_sampler, "sampler")
        self._sampler_uses_v1_api = isinstance(self._raw_sampler, BaseSamplerV1)

    # ------------------------------------------------------------------
    # ISA transpilation (for IBM / fake backends)
    # ------------------------------------------------------------------

    def _isa_transpile_qiskit_circuit(self, circuit):
        """Transpile a raw Qiskit ``QuantumCircuit`` to ISA form.

        Returns the circuit unchanged when no IBM backend is configured.
        The resulting circuit retains its :class:`Parameter` objects so
        that it can still be parameterised afterwards.
        """
        if not self._isa_transpile or self._backend is None:
            return circuit

        try:
            from qiskit.transpiler.preset_passmanagers import (  # pylint: disable=import-outside-toplevel
                generate_preset_pass_manager,
            )

            pm = generate_preset_pass_manager(
                optimization_level=1,
                backend=self._backend,
            )
            return pm.run(circuit)
        except ImportError:
            from qiskit import transpile  # pylint: disable=import-outside-toplevel

            return transpile(circuit, backend=self._backend)

    def _isa_apply_layout_to_observable(self, observable, circuit):
        """Apply the transpiled circuit's layout to an observable.

        After ISA transpilation the virtual-to-physical qubit mapping may
        have changed. ``SparsePauliOp.apply_layout`` re-orders the observable
        to match the new mapping.
        """
        from qiskit.quantum_info import SparsePauliOp  # pylint: disable=import-outside-toplevel

        if not isinstance(observable, SparsePauliOp):
            return observable
        layout = getattr(circuit, "layout", None)
        if layout is None:
            return observable
        try:
            return observable.apply_layout(layout)
        except (ValueError, TypeError):
            logger.warning(
                "Failed to apply layout to observable; using original observable. "
                "This may lead to incorrect expectation values if the observable "
                "does not match the transpiled circuit's qubit mapping.",
                exc_info=True,
            )
            return observable

    def _convert_to_optree(
        self,
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
        operator: QuantumOperatorBase | List[QuantumOperatorBase] | None = None,
    ) -> Tuple[OpTreeCircuit | OpTreeNodeBase, OpTreeOperator | OpTreeNodeBase | None]:
        """
        Convert circuits and operators to OpTree format.

        When an IBM backend is configured the circuits are ISA-transpiled
        and operators are re-mapped to match the transpiled qubit layout.

        Args:
            circuit: Circuit(s) to convert
            operator: Operator(s) to convert (optional)

        Returns:
            Tuple of (circuit_tree, operator_tree)
        """
        uses_ibm_backend = self._isa_transpile

        def _to_qiskit(c):
            return getattr(c, "qiskit_circuit", c)

        if isinstance(circuit, List):
            raw_circuits = [_to_qiskit(c) for c in circuit]
        else:
            raw_circuits = [_to_qiskit(circuit)]

        if uses_ibm_backend:
            transpiled_circuits = [self._isa_transpile_qiskit_circuit(c) for c in raw_circuits]
        else:
            transpiled_circuits = raw_circuits

        if len(transpiled_circuits) == 1:
            circuit_tree = OpTreeCircuit(transpiled_circuits[0])
        else:
            circuit_tree = OpTreeList([OpTreeCircuit(c) for c in transpiled_circuits])

        if operator is None:
            return circuit_tree, None

        def _to_operator(o):
            return getattr(o, "qiskit_operator", o)

        if isinstance(operator, List):
            ops = [_to_operator(o) for o in operator]
            if uses_ibm_backend:
                if len(ops) == len(transpiled_circuits):
                    ops = [
                        self._isa_apply_layout_to_observable(o, c)
                        for o, c in zip(ops, transpiled_circuits)
                    ]
                else:
                    ops = [
                        self._isa_apply_layout_to_observable(o, transpiled_circuits[0])
                        for o in ops
                    ]
            operator_tree = OpTreeList([OpTreeOperator(o) for o in ops])
        else:
            operator = _to_operator(operator)
            if uses_ibm_backend:
                operator = self._isa_apply_layout_to_observable(operator, transpiled_circuits[0])
            operator_tree = OpTreeOperator(operator)

        return circuit_tree, operator_tree

    def _prepare_parameter_dicts(
        self,
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
        observable: QuantumOperatorBase | List[QuantumOperatorBase] | None = None,
        **parameters,
    ) -> Tuple[dict, dict]:
        """
        Prepare separate parameter dictionaries for circuits and operators.

        Args:
            circuit: The quantum circuit(s)
            observable: The quantum observable(s)
            **parameters: Keyword arguments with parameter values

        Returns:
            Tuple of (circuit_param_dict, observable_param_dict)
        """

        # helper to get the underlying qiskit objects
        def _unwrap(obj):
            if hasattr(obj, "qiskit_circuit"):
                return obj.qiskit_circuit
            if hasattr(obj, "qiskit_operator"):
                return obj.qiskit_operator
            return obj

        def _collect_objects(obj_or_list):
            if isinstance(obj_or_list, list):
                return [_unwrap(o) for o in obj_or_list]
            return [_unwrap(obj_or_list)]

        # Collect all circuits and observables
        circuits = _collect_objects(circuit)
        observables = _collect_objects(observable) if observable is not None else []

        def _build_param_dict(qiskit_objects):
            param_dict = {}
            for qobj in qiskit_objects:
                for p in qobj.parameters:
                    # Support both ParameterVector elements and standalone Parameters
                    name = p.vector.name if hasattr(p, "vector") else p.name
                    if name not in parameters:
                        continue
                    supplied = parameters[name]
                    # Normalize to numpy
                    if isinstance(supplied, (list, tuple, np.ndarray)):
                        arr = np.asarray(supplied)
                        if hasattr(p, "index"):
                            # ParameterVector element – bind by index
                            try:
                                val = arr[p.index]
                            except (IndexError, TypeError) as exc:
                                if arr.size == 1:
                                    val = arr.flat[0]
                                else:
                                    raise ValueError(
                                        f"Provided values for parameter '{name}' have length "
                                        f"{arr.size} but parameter index {p.index} is requested."
                                    ) from exc
                        else:
                            # Standalone Parameter – scalar expected; take first element
                            val = arr.flat[0] if arr.size == 1 else arr
                    else:
                        val = supplied
                    param_dict[p] = val
            return param_dict

        circuit_dict = _build_param_dict(circuits)
        observable_dict = _build_param_dict(observables) if observables else {}
        return circuit_dict, observable_dict

    def _extract_counts(self, pub_result, n_qubits=None):
        """Extract measurement counts from a primitive result object.

        Handles both the Qiskit 2.x / V2 PUB result format and the
        Qiskit 1.x / V1 ``quasi_dists`` format.
        """
        # --- Qiskit 2.x / V2 primitives ---
        # PrimitiveResult is iterable but not necessarily subscriptable;
        # materialise to a list before indexing.
        if hasattr(pub_result, "__iter__") and not isinstance(pub_result, (str, dict)):
            pubs = list(pub_result)
            if pubs and hasattr(pubs[0], "data"):
                counts_list = []
                for i, pub in enumerate(pubs):
                    data = getattr(pub, "data", None)
                    meas = getattr(data, "meas", None) if data is not None else None
                    if meas is None or not hasattr(meas, "get_counts"):
                        raise ValueError(
                            f"Unsupported sampler result format at pub index {i}: "
                            f"'data.meas.get_counts()' is not available "
                            f"(got type {type(pub)!r})."
                        )
                    counts_list.append(_convert_counts_endianness(meas.get_counts()))
                return counts_list

        # --- Qiskit 1.x / V1 primitives ---
        if hasattr(pub_result, "quasi_dists"):
            quasi_dists = pub_result.quasi_dists
            metadata = getattr(pub_result, "metadata", None)
            if metadata is None:
                raise ValueError(
                    "Unsupported sampler result format: 'metadata' attribute is "
                    "missing for quasi_dists."
                )
            counts_list = []
            for idx, qd in enumerate(quasi_dists):
                if idx >= len(metadata):
                    raise ValueError(
                        f"Unsupported sampler result format: 'metadata' has "
                        f"{len(metadata)} entries but quasi_dists has "
                        f"{len(quasi_dists)}."
                    )
                if "shots" not in metadata[idx]:
                    raise ValueError(
                        f"Unsupported sampler result format: "
                        f"'metadata[{idx}][\"shots\"]' is missing."
                    )
                shots = metadata[idx]["shots"]
                counts = {format(k, f"0{n_qubits}b"): int(round(v * shots)) for k, v in qd.items()}
                counts_list.append(_convert_counts_endianness(counts))
            return counts_list

        raise ValueError("Unsupported primitive result format: cannot extract counts.")

    def _expectation_value(
        self,
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
        observable: QuantumOperatorBase | List[QuantumOperatorBase],
        **parameter_values,
    ) -> float | np.ndarray:
        """
        Calculate the expectation value using OpTree and Qiskit Estimator.

        Args:
            circuit: The quantum circuit or a list of circuits.
            observable: The quantum observable or a list of observables.
            parameter_values: Parameter values as keyword arguments.

        Returns:
            The expectation value(s).
        """
        if self._ibm_quantum_backend:
            self._ensure_runtime_primitives()

        if self._estimator is None:
            raise RuntimeError(
                "No estimator is configured. Pass `backend` as an Estimator primitive "
                "or use a backend/session that supports estimation."
            )
        # Convert to OpTree format
        circuit_tree, observable_tree = self._convert_to_optree(circuit, observable)

        # Prepare separate parameter dictionaries
        circuit_dict, observable_dict = self._prepare_parameter_dicts(
            circuit, observable, **parameter_values
        )

        # Use OpTree evaluation with Estimator
        return OpTreeEvaluate.evaluate_with_estimator(
            circuit=circuit_tree,
            operator=observable_tree,
            dictionary_circuit=circuit_dict,
            dictionary_operator=observable_dict,
            estimator=self._estimator,
            dictionaries_combined=False,
            detect_duplicates=True,
        )

    def _expectation_value_derivatives(
        self,
        circuit: QuantumCircuitBase | List[QuantumCircuitBase],
        observable: QuantumOperatorBase | List[QuantumOperatorBase],
        *derivative_params,
        **parameter_values,
    ) -> np.ndarray | dict:
        """
        Calculate the derivatives using OpTree parameter shift.

        Args:
            circuit: The quantum circuit.
            observable: The quantum observable.
            derivative_params: Parameters to differentiate with respect to.
            parameter_values: Parameter values as keyword arguments.

        Returns:
            Derivative values.
        """
        if self._ibm_quantum_backend:
            self._ensure_runtime_primitives()

        if self._estimator is None:
            raise RuntimeError(
                "No estimator is configured. Pass `backend` as an Estimator primitive "
                "or use a backend/session that supports estimation."
            )

        # If no derivative parameters specified, return expectation value
        if len(derivative_params) == 0:
            return self._expectation_value(circuit, observable, **parameter_values)

        # Convert to OpTree format
        circuit_tree, observable_tree = self._convert_to_optree(circuit, observable)

        # Prepare separate parameter dictionaries
        circuit_dict, observable_dict = self._prepare_parameter_dicts(
            circuit, observable, **parameter_values
        )

        # Build separate parameter sets for circuit and observable so we can
        # apply the product rule correctly.
        if isinstance(circuit, list):
            circuit_param_set = set(getattr(circuit[0], "qiskit_circuit", circuit[0]).parameters)
        else:
            circ = getattr(circuit, "qiskit_circuit", circuit)
            circuit_param_set = set(circ.parameters)

        if observable is not None:
            if isinstance(observable, list):
                observable_param_set: set = set()
                for obs in observable:
                    obs_obj = getattr(obs, "qiskit_operator", obs)
                    observable_param_set |= set(obs_obj.parameters)
            else:
                obs_obj = getattr(observable, "qiskit_operator", observable)
                observable_param_set = set(obs_obj.parameters)
        else:
            observable_param_set = set()

        all_params = circuit_param_set | observable_param_set

        def _param_name(p) -> str:
            return p.vector.name if hasattr(p, "vector") else p.name

        def _derivative_for_single_param(p) -> float:
            """∂E/∂p = circuit contribution + observable contribution (product rule)."""
            total = 0.0

            # Circuit contribution: ⟨∂ψ/∂p|H|ψ⟩  (parameter shift on circuit)
            if p in circuit_param_set:
                circ_deriv = OpTreeDerivative.differentiate(circuit_tree, [p])
                total += OpTreeEvaluate.evaluate_with_estimator(
                    circuit=circ_deriv,
                    operator=observable_tree,
                    dictionary_circuit=circuit_dict,
                    dictionary_operator=observable_dict,
                    estimator=self._estimator,
                    detect_duplicates=True,
                )

            # Observable contribution: ⟨ψ|∂H/∂p|ψ⟩
            if p in observable_param_set:
                op_deriv = OpTreeDerivative.differentiate(observable_tree, [p])
                total += OpTreeEvaluate.evaluate_with_estimator(
                    circuit=circuit_tree,
                    operator=op_deriv,
                    dictionary_circuit=circuit_dict,
                    dictionary_operator=observable_dict,
                    estimator=self._estimator,
                    detect_duplicates=True,
                )

            return total

        results: dict = {}
        for dp in derivative_params:
            if isinstance(dp, str):
                matching = [p for p in all_params if _param_name(p) == dp]
                if not matching:
                    results[dp] = 0.0
                    continue
                # Sort ParameterVector elements by index; standalone Parameters have no index.
                matching.sort(key=lambda p: p.index if hasattr(p, "index") else 0)
                if len(matching) == 1:
                    results[dp] = _derivative_for_single_param(matching[0])
                else:
                    # ParameterVector: return one derivative value per element.
                    results[dp] = np.array([_derivative_for_single_param(p) for p in matching])
            elif isinstance(dp, ParameterVectorElement):
                results[dp] = _derivative_for_single_param(dp)
            else:
                raise ValueError(f"Unknown derivative parameter type: {type(dp)}")

        if len(derivative_params) == 1:
            return results[derivative_params[0]]
        return results

    def _sample(
        self, circuit: QuantumCircuitBase | List[QuantumCircuitBase], **parameter_values
    ) -> List[dict]:
        """Sample from the circuit using OpTree and Qiskit Sampler."""
        if self._ibm_quantum_backend:
            self._ensure_runtime_primitives()

        if self._sampler is None:
            raise RuntimeError(
                "No sampler is configured. Pass `backend` as a Sampler primitive "
                "or use a backend/session that supports sampling."
            )
        if self._shots is None:
            raise ValueError("Shots must be set for sampling.")

        # Convert to OpTree format (just for consistent handling)
        circuit_tree, _ = self._convert_to_optree(circuit, operator=None)

        # Prepare parameter dictionary (only for circuits)
        circuit_dict, _ = self._prepare_parameter_dicts(
            circuit, observable=None, **parameter_values
        )

        # Extract circuits from OpTree
        if isinstance(circuit_tree, OpTreeCircuit):
            circuits = [circuit_tree.circuit]
        else:
            circuits = [child.circuit for child in circuit_tree.children]

        # Bind parameters to circuits
        bound_circuits = []
        for circ in circuits:
            # Bind only parameters that exist in this circuit
            params_to_bind = {p: circuit_dict[p] for p in circ.parameters if p in circuit_dict}
            bound_circ = circ.assign_parameters(params_to_bind) if params_to_bind else circ
            if bound_circ.num_clbits == 0:
                bound_circ.measure_all()
            bound_circuits.append(bound_circ)

        if self._sampler_uses_v1_api:
            job = self._sampler.run(bound_circuits, shots=self._shots)
        else:
            pubs = [(circ,) for circ in bound_circuits]
            job = self._sampler.run(pubs, shots=self._shots)

        result = job.result()

        is_list_input = isinstance(circuit, list)
        raw_circuits_for_nq = circuit if is_list_input else [circuit]
        n_qubits_list = [getattr(c, "qiskit_circuit", c).num_qubits for c in raw_circuits_for_nq]
        counts_list = self._extract_counts(result, n_qubits_list[0])

        if not is_list_input:
            return counts_list[0] if isinstance(counts_list, list) else counts_list
        return counts_list

    def _statevector(
        self, circuit: QuantumCircuitBase | List[QuantumCircuitBase], **parameter_values
    ) -> np.ndarray:
        """Compute the statevector of the circuit using local Qiskit simulation.

        Statevector computation is always performed locally regardless of the
        configured backend, since it is a purely classical computation.
        """
        # Convert to OpTree but without ISA transpilation — use the raw circuit
        if isinstance(circuit, list):
            raw_circuits = [getattr(c, "qiskit_circuit", c) for c in circuit]
        else:
            raw_circuits = [getattr(circuit, "qiskit_circuit", circuit)]

        # Prepare parameter dictionary
        circuit_dict, _ = self._prepare_parameter_dicts(
            circuit, observable=None, **parameter_values
        )

        statevectors = []
        for circ in raw_circuits:
            params_to_bind = {p: circuit_dict[p] for p in circ.parameters if p in circuit_dict}
            bound_circ = circ.assign_parameters(params_to_bind) if params_to_bind else circ
            statevectors.append(_reverse_qubit_ordering(Statevector(bound_circ).data))

        statevectors = np.array(statevectors)
        return statevectors[0] if len(raw_circuits) == 1 else statevectors

    def _transpile_circuit(self, circuit: QuantumCircuitBase) -> QiskitCircuit:
        """Transpile a generic QuantumCircuit to a Qiskit QuantumCircuit.

        For remote IBM backends the circuit is additionally transpiled through
        ``generate_preset_pass_manager`` to produce an ISA-compliant circuit
        that conforms to the target backend's instruction set.

        Args:
            circuit (QuantumCircuitBase): The generic QuantumCircuit to transpile.

        Returns:
            QiskitCircuit: The corresponding QiskitCircuit.
        """
        qc = QiskitCircuit(circuit)
        isa_circuit = self._isa_transpile_qiskit_circuit(qc.qiskit_circuit)
        if isa_circuit is not qc.qiskit_circuit:
            return QiskitCircuit.from_qiskit(isa_circuit)
        return qc

    def _transpile_operator(self, operator: QuantumOperatorBase) -> QiskitOperator:
        """Transpile a generic QuantumOperator to a Qiskit QuantumOperator.

        Args:
            operator (QuantumOperatorBase): The generic QuantumOperator to transpile.
        Returns:
            QiskitOperator: The corresponding QiskitOperator.
        """
        if isinstance(operator, self._native_operator_class):
            return operator
        return self._native_operator_class.from_quantum_operator(operator)

    @classmethod
    def get_accepted_backend_types(cls) -> list[type]:
        """Return all types accepted as the ``backend`` argument.

        Covers:
        * Qiskit local backends (``Backend`` / ``BackendV2``)
        * Qiskit statevector primitives (``StatevectorEstimator``, ``StatevectorSampler``)
        * Qiskit local primitive base classes (V1 and V2)
        * IBM Runtime ``Session`` / ``Batch`` (when *qiskit-ibm-runtime* is installed)
        * IBM Runtime primitive classes (V1 and V2, version-gated)

        Dummy sentinel classes defined when optional dependencies are absent are
        intentionally excluded so that ``isinstance`` checks never yield false
        positives.
        """

        types: list[type] = [
            Backend,
            StatevectorEstimator,
            StatevectorSampler,
        ]

        for cls_ in (BaseEstimatorV1, BaseEstimatorV2, BaseSamplerV1, BaseSamplerV2):
            if getattr(cls_, "__module__", "").startswith("qiskit."):
                types.append(cls_)

        # ── IBM Runtime types (only when the real classes are importable) ───
        if QISKIT_RUNTIME_AVAILABLE:
            # Session and Batch are real at this point (imported at module top)
            types.extend([Session, Batch])

            # Runtime primitives: include only the non-dummy versions.
            # The module-level conditionals guarantee that the V1/V2 splits are
            # correctly resolved; we just need to skip the local dummy sentinels.
            for rt_cls in (
                RuntimeEstimatorV1,
                RuntimeEstimatorV2,
                RuntimeSamplerV1,
                RuntimeSamplerV2,
            ):
                # Dummy classes are defined in this module and have no __module__
                # pointing to qiskit_ibm_runtime — use that as the guard.
                if "qiskit_ibm_runtime" in getattr(rt_cls, "__module__", ""):
                    types.append(rt_cls)

        return types

    @classmethod
    def get_accepted_backend_aliases(cls) -> list[str]:
        """Return string aliases accepted by this executor in ``Executor.create``."""
        return ["statevector", "aer_statevector", "aer"]
