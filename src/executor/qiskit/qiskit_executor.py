import logging
import numpy as np
from typing import List, Literal, Optional, Tuple, Union

from executor.base.circuit_base import QuantumCircuitBase
from executor.base.executor_base import ExecutorBase
from executor.base.operator_base import QuantumOperatorBase
from qiskit.primitives import (
    StatevectorEstimator,
    StatevectorSampler,
)
from qiskit.quantum_info import Statevector

from executor.utils.qiskit_compat import (
    QISKIT_SMALLER_1_2,
    QISKIT_SMALLER_2_0,
    QISKIT_RUNTIME_AVAILABLE,
    QISKIT_RUNTIME_SMALLER_0_21,
    QISKIT_RUNTIME_SMALLER_0_23,
    QISKIT_RUNTIME_SMALLER_0_28,
)

from executor.qiskit.qiskit_circuit import QiskitCircuit
from executor.qiskit.optree import OpTreeDerivative
from executor.qiskit.optree import OpTreeEvaluate
from executor.qiskit.optree.optree import (
    OpTreeCircuit,
    OpTreeList,
    OpTreeNodeBase,
    OpTreeOperator,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy-loaded helpers for optional dependencies
# ---------------------------------------------------------------------------


def _load_aer_simulator():
    try:
        from qiskit_aer import AerSimulator
    except ImportError as e:
        raise ImportError(
            "qiskit-aer is required for 'backend=\"aer\"' and for shot-based "
            "sampling with 'backend=\"statevector\"'. Install with: "
            "pip install executor[qiskit-full]"
        ) from e
    return AerSimulator


def _check_runtime_available():
    """Guard: raise :class:`ImportError` when *qiskit-ibm-runtime* is missing."""
    if not QISKIT_RUNTIME_AVAILABLE:
        raise ImportError(
            "qiskit-ibm-runtime is required for IBM backend support. "
            "Install with: pip install executor[qiskit-full]"
        )


def _load_runtime_primitives_v1():
    """Return ``(RuntimeEstimatorV1, RuntimeSamplerV1)`` for *qiskit-ibm-runtime < 0.21*.

    In these old versions ``Estimator`` / ``Sampler`` *are* the V1 primitives
    (V2 does not exist yet).  The returned classes inherit from
    ``BaseEstimatorV1`` / ``BaseSamplerV1``.
    """
    _check_runtime_available()
    from qiskit_ibm_runtime import (
        Estimator as RuntimeEstimatorV1,
        Sampler as RuntimeSamplerV1,
    )

    return RuntimeEstimatorV1, RuntimeSamplerV1


def _load_runtime_primitives_v2():
    """Return ``(RuntimeEstimatorV2, RuntimeSamplerV2)`` from *qiskit-ibm-runtime*.

    The concrete import path depends on the installed runtime version:

    * ``0.21 – 0.27``: explicit ``EstimatorV2`` / ``SamplerV2`` exports.
    * ``>= 0.28``: ``Estimator`` / ``Sampler`` **are** the V2 primitives
      (V1 has been removed).
    """
    _check_runtime_available()
    if QISKIT_RUNTIME_SMALLER_0_28:
        # 0.21 – 0.27: V2 is available as an explicit named export
        from qiskit_ibm_runtime import (
            EstimatorV2 as RuntimeEstimatorV2,
            SamplerV2 as RuntimeSamplerV2,
        )
    else:
        # >= 0.28: V1 removed, the default Estimator/Sampler IS V2
        from qiskit_ibm_runtime import (
            Estimator as RuntimeEstimatorV2,
            Sampler as RuntimeSamplerV2,
        )
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


def _is_backend_instance(obj) -> bool:
    """Return *True* if *obj* is a Qiskit ``Backend`` (V1 or V2) instance.

    The check avoids importing ``BackendV2`` at module level so that
    ``qiskit-ibm-runtime`` remains an optional dependency.
    """
    try:
        from qiskit.providers import Backend  # available in all supported Qiskit versions

        return isinstance(obj, Backend)
    except ImportError:
        return False


def _detect_backend_flags(backend) -> Tuple[bool, bool]:
    """Detect whether *backend* is a remote IBM Quantum device or a fake backend.

    Returns ``(remote, ibm_quantum)`` flags following the sQUlearn heuristic:
    string-match ``"ibm"`` and ``"fake"`` on the backend representation.
    """
    backend_str = str(backend).lower()
    if "ibm" in backend_str:
        is_fake = "fake" in backend_str
        return (not is_fake, not is_fake)
    return (False, False)


# ---------------------------------------------------------------------------
# Version-gated Qiskit primitive imports (local simulators)
# ---------------------------------------------------------------------------

if QISKIT_SMALLER_1_2:
    from qiskit.primitives import (
        BackendEstimator as BackendEstimator,
        BackendSampler as BackendSampler,
    )
    from qiskit.circuit import ParameterExpression as ParameterVectorElement
elif QISKIT_SMALLER_2_0:
    from qiskit.primitives import (
        BackendEstimatorV2 as BackendEstimator,
        BackendSamplerV2 as BackendSampler,
    )
    from qiskit.circuit import ParameterExpression as ParameterVectorElement
else:
    from qiskit.primitives import (
        BackendEstimatorV2 as BackendEstimator,
        BackendSamplerV2 as BackendSampler,
    )
    from qiskit.circuit import ParameterVectorElement


class QiskitExecutor(ExecutorBase):
    """Class for executing qiskit circuits.

    Supports local simulation (``"statevector"``, ``"aer"``) **and** execution
    on real IBM Quantum hardware or noise-aware fake backends via
    `qiskit-ibm-runtime <https://github.com/Qiskit/qiskit-ibm-runtime>`_.

    When a :class:`~qiskit.providers.BackendV2` / ``IBMBackend`` instance is
    passed as *backend*, the executor transparently initialises
    ``EstimatorV2`` / ``SamplerV2`` from *qiskit-ibm-runtime*, manages an
    optional ``Session``, and applies ISA transpilation so that circuits
    conform to the target backend's instruction set.

    Args:
        shots (int, optional): Number of shots for sampling.  Defaults to
            ``None`` (exact statevector mode for local backends).
        seed (int, optional): Random seed for reproducibility.
        log_file (str, optional): Path to the log file.
        log_level (str, optional): Logging level. One of ``"DEBUG"``,
            ``"INFO"``, ``"WARNING"``, ``"ERROR"``.  Defaults to
            ``"WARNING"``.
        caching (bool, optional): Whether to use caching.
        cache_dir (str, optional): Directory for caching.  Defaults to
            ``"cache"``.
        max_cache_size (int, optional): Maximum number of entries kept in
            each in-memory cache.  ``None`` means unlimited.
        backend (str or Backend): Backend specification.  Can be
            ``"statevector"`` (default), ``"aer"``, or any Qiskit
            :class:`~qiskit.providers.Backend` / ``BackendV2`` instance
            (e.g. obtained from ``QiskitRuntimeService``).
        execution_mode (str, optional): Only relevant for IBM backends.
            One of ``"job"`` (default), ``"session"`` or ``"batch"``.
        options (dict, optional): Options forwarded to the runtime
            primitives (e.g. error-mitigation settings).  Ignored for local
            backends.
    """

    def __init__(
        self,
        shots: Union[int, None] = None,
        seed: Union[int, None] = None,
        log_file: Union[str, None] = None,
        log_level: str = "WARNING",
        caching: Union[bool, None] = None,
        cache_dir: str = "cache",
        max_cache_size: Union[int, None] = None,
        backend: Union[str, object] = "statevector",
        execution_mode: Literal["job", "session", "batch"] = "job",
        options: Optional[dict] = None,
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
        self._remote_backend: bool = False
        self._ibm_quantum_backend: bool = False
        self._execution_mode = execution_mode
        self._options = options
        self._sampler_uses_v1_api: bool = QISKIT_SMALLER_1_2
        # True only when an external Backend object (IBM/fake) is passed;
        # local AerSimulator instances created by the string shortcuts are
        # never ISA-transpiled.
        self._isa_transpile: bool = False

        # ── Local simulator backends ──────────────────────────────────────
        if isinstance(backend, str):
            if backend == "statevector":
                if shots is None:
                    self._estimator = StatevectorEstimator()
                    self._sampler = StatevectorSampler()
                    self._backend = None
                else:
                    AerSimulator = _load_aer_simulator()
                    self._backend = AerSimulator(method="statevector")
                    self._estimator = BackendEstimator(backend=self._backend)
                    self._sampler = BackendSampler(backend=self._backend)
            elif backend == "aer":
                AerSimulator = _load_aer_simulator()
                self._backend = AerSimulator()
                self._estimator = BackendEstimator(backend=self._backend)
                self._sampler = BackendSampler(backend=self._backend)
            else:
                raise ValueError(
                    f"Unknown backend string: {backend!r}. "
                    "Use 'statevector', 'aer', or pass a Backend instance."
                )

        # ── Backend object (IBMBackend / FakeBackend / any BackendV2) ─────
        elif _is_backend_instance(backend):
            if not QISKIT_RUNTIME_AVAILABLE:
                raise ImportError(
                    "qiskit-ibm-runtime is required to use a Backend instance. "
                    "Install with: pip install executor[qiskit-full]"
                )

            self._backend = backend
            self._remote_backend, self._ibm_quantum_backend = _detect_backend_flags(backend)
            self._isa_transpile = True

            # Determine which primitive generation to use
            self._runtime_primitives_version: str = "v1" if QISKIT_RUNTIME_SMALLER_0_21 else "v2"

            # Session management (only for real IBM Quantum devices)
            if self._ibm_quantum_backend and execution_mode in ("session", "batch"):
                self._create_session()

            # Create primitives.
            # For runtime < 0.21 the V1 runtime Estimator/Sampler require a
            # QiskitRuntimeService account even for fake backends, so we
            # fall back to local BackendEstimator / BackendSampler when no
            # real IBM session is available.
            if self._runtime_primitives_version == "v1" and not self._ibm_quantum_backend:
                # Local / fake backend with old runtime -> use Qiskit-local primitives
                # (BackendEstimator / BackendSampler – V2 API on Qiskit >= 1.2)
                self._estimator = BackendEstimator(backend=self._backend)
                self._sampler = BackendSampler(backend=self._backend)
                # The local fallback primitives use V2 API on Qiskit >= 1.2 even
                # though _runtime_primitives_version is "v1".
                self._sampler_uses_v1_api: bool = QISKIT_SMALLER_1_2
            else:
                self._estimator = self._create_runtime_estimator()
                self._sampler = self._create_runtime_sampler()
                self._sampler_uses_v1_api = self._runtime_primitives_version == "v1"

            logger.info(
                "Initialised QiskitExecutor with %s (remote=%s, mode=%s)",
                backend,
                self._remote_backend,
                execution_mode,
            )
        else:
            raise TypeError(
                f"'backend' must be a string ('statevector', 'aer') or a "
                f"Qiskit Backend instance, got {type(backend)!r}."
            )

        if seed is not None:
            self._random = np.random.default_rng(seed)
        else:
            self._random = np.random.default_rng()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shots(self) -> Union[int, None]:
        """Return the number of shots."""
        return self._shots

    @shots.setter
    def shots(self, value: Union[int, None]) -> None:
        """Set the number of shots."""
        self._shots = value

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
        """Return the active runtime session, or ``None``."""
        return self._session

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def _create_session(self) -> None:
        """Create (or re-create) a :class:`~qiskit_ibm_runtime.Session`."""
        Session = _load_runtime_session()
        self._session = Session(backend=self._backend)
        logger.debug("Created new runtime session for %s", self._backend)

    def close_session(self) -> None:
        """Close the current runtime session if one is active."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                logger.debug("Session.close() raised; ignoring.", exc_info=True)
            self._session = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_session()
        return False

    def __del__(self):
        try:
            self.close_session()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Runtime primitive factories
    # ------------------------------------------------------------------

    def _ensure_session_active(self) -> None:
        """Re-create the session when it has expired."""
        if (
            self._ibm_quantum_backend
            and self._session is not None
            and not getattr(self._session, "_active", True)
        ):
            self._create_session()

    def _create_runtime_estimator(self):
        """Instantiate the runtime Estimator for the current backend / session."""
        if self._runtime_primitives_version == "v1":
            cls, _ = _load_runtime_primitives_v1()
            return self._instantiate_runtime_primitive_v1(cls, self._options)
        else:
            cls, _ = _load_runtime_primitives_v2()
            return self._instantiate_runtime_primitive_v2(cls, self._options)

    def _create_runtime_sampler(self):
        """Instantiate the runtime Sampler for the current backend / session."""
        if self._runtime_primitives_version == "v1":
            _, cls = _load_runtime_primitives_v1()
            return self._instantiate_runtime_primitive_v1(cls, self._options)
        else:
            _, cls = _load_runtime_primitives_v2()
            return self._instantiate_runtime_primitive_v2(cls, self._options)

    # -- V1 instantiation (qiskit-ibm-runtime < 0.21) ---------------------

    def _instantiate_runtime_primitive_v1(self, primitive_cls, options):
        """Create a V1 runtime primitive.

        V1 primitives (``qiskit-ibm-runtime < 0.21``) always use
        ``session=`` / ``backend=`` and accept an ``Options`` object from
        ``qiskit_ibm_runtime.options``.
        """
        # Build V1 Options object from the user-supplied dict (if any)
        if options:
            RuntimeOptionsV1 = _load_runtime_options_v1()
            opts = RuntimeOptionsV1()
            for key, val in options.items():
                try:
                    setattr(opts, key, val)
                except (AttributeError, TypeError):
                    pass  # skip keys the V1 Options class doesn't know
        else:
            opts = None

        if self._ibm_quantum_backend and self._session is not None:
            self._ensure_session_active()
            return (
                primitive_cls(session=self._session, options=opts)
                if opts is not None
                else primitive_cls(session=self._session)
            )
        else:
            return (
                primitive_cls(backend=self._backend, options=opts)
                if opts is not None
                else primitive_cls(backend=self._backend)
            )

    # -- V2 instantiation (qiskit-ibm-runtime >= 0.21) --------------------

    def _instantiate_runtime_primitive_v2(self, primitive_cls, options):
        """Create a V2 runtime primitive.

        * ``qiskit-ibm-runtime < 0.23`` uses ``session=`` / ``backend=``.
        * ``qiskit-ibm-runtime >= 0.23`` uses ``mode=``.
        """
        opts = options or {}

        if self._ibm_quantum_backend and self._session is not None:
            # Real IBM device with an active session
            self._ensure_session_active()
            if QISKIT_RUNTIME_SMALLER_0_23:
                return (
                    primitive_cls(session=self._session, options=opts)
                    if opts
                    else primitive_cls(session=self._session)
                )
            else:
                return (
                    primitive_cls(mode=self._session, options=opts)
                    if opts
                    else primitive_cls(mode=self._session)
                )
        else:
            # Fake backend or real backend without session (job mode)
            if QISKIT_RUNTIME_SMALLER_0_23:
                return (
                    primitive_cls(backend=self._backend, options=opts)
                    if opts
                    else primitive_cls(backend=self._backend)
                )
            else:
                return (
                    primitive_cls(mode=self._backend, options=opts)
                    if opts
                    else primitive_cls(mode=self._backend)
                )

    def _refresh_primitives(self) -> None:
        """Re-create primitives after a session renewal."""
        self._estimator = self._create_runtime_estimator()
        self._sampler = self._create_runtime_sampler()

    # ------------------------------------------------------------------
    # ISA transpilation (for IBM / fake backends)
    # ------------------------------------------------------------------

    def _isa_transpile_qiskit_circuit(self, circuit):
        """Transpile a raw Qiskit ``QuantumCircuit`` to ISA form.

        If no IBM backend is configured the circuit is returned unchanged.
        The resulting circuit retains its :class:`Parameter` objects so that
        it can still be parameterised afterwards.
        """
        if not self._isa_transpile or self._backend is None:
            return circuit

        try:
            from qiskit.transpiler.preset_passmanagers import (
                generate_preset_pass_manager,
            )

            pm = generate_preset_pass_manager(
                optimization_level=1,
                backend=self._backend,
            )
            return pm.run(circuit)
        except ImportError:
            from qiskit import transpile

            return transpile(circuit, backend=self._backend)

    def _isa_apply_layout_to_operator(self, operator, circuit):
        """Apply the transpiled circuit's layout to an operator.

        After ISA transpilation the virtual-to-physical qubit mapping
        may have changed.  ``SparsePauliOp.apply_layout`` re-orders the
        operator to match.
        """
        from qiskit.quantum_info import SparsePauliOp

        if not isinstance(operator, SparsePauliOp):
            return operator
        layout = getattr(circuit, "layout", None)
        if layout is None:
            return operator
        try:
            return operator.apply_layout(layout)
        except Exception:
            logger.warning(
                "Failed to apply layout to operator; using original operator. "
                "This may lead to incorrect expectation values if the operator "
                "does not match the transpiled circuit's qubit mapping.",
                exc_info=True,
            )
            return operator

    def _convert_to_optree(
        self,
        circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]],
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase], None] = None,
    ) -> Tuple[Union[OpTreeCircuit, OpTreeNodeBase], Union[OpTreeOperator, OpTreeNodeBase, None]]:
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

        # --- Convert & optionally transpile circuits ---
        def _to_qiskit(c):
            return c._qiskit_circuit if hasattr(c, "_qiskit_circuit") else c

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

        # --- Convert operators (apply layout if transpiled) ---
        if operator is None:
            return circuit_tree, None

        def _to_operator(o):
            return o._qiskit_operator if hasattr(o, "_qiskit_operator") else o

        if isinstance(operator, List):
            ops = [_to_operator(o) for o in operator]
            if uses_ibm_backend:
                # Apply per-circuit layout when counts match; fall back to
                # circuit[0] layout when a single operator is broadcast across
                # multiple circuits (N circuits × 1 operator is handled by the
                # scalar branch below, but N×M with M<N is an edge case we
                # cover conservatively with circuit[0]).
                if len(ops) == len(transpiled_circuits):
                    ops = [
                        self._isa_apply_layout_to_operator(o, c)
                        for o, c in zip(ops, transpiled_circuits)
                    ]
                else:
                    # Mismatched counts – use circuit[0] layout as best effort
                    ops = [
                        self._isa_apply_layout_to_operator(o, transpiled_circuits[0]) for o in ops
                    ]
            operator_tree = OpTreeList([OpTreeOperator(o) for o in ops])
        else:
            op = _to_operator(operator)
            if uses_ibm_backend:
                # Single operator broadcast across all circuits – use circuit[0]
                # layout (ISA transpilation with optimization_level=1 produces
                # a consistent virtual→physical mapping for identical circuits,
                # and a single operator is paired with each circuit identically).
                op = self._isa_apply_layout_to_operator(op, transpiled_circuits[0])
            operator_tree = OpTreeOperator(op)

        return circuit_tree, operator_tree

    def _prepare_parameter_dicts(
        self,
        circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]],
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase], None] = None,
        **parameters,
    ) -> Tuple[dict, dict]:
        """
        Prepare separate parameter dictionaries for circuits and operators.

        Args:
            circuit: The quantum circuit(s)
            operator: The quantum operator(s)
            **parameters: Keyword arguments with parameter values

        Returns:
            Tuple of (circuit_param_dict, operator_param_dict)
        """

        # helper to get the underlying qiskit objects
        def _unwrap(obj):
            """Extract underlying qiskit object"""
            if hasattr(obj, "_qiskit_circuit"):
                return obj._qiskit_circuit
            elif hasattr(obj, "_qiskit_operator"):
                return obj._qiskit_operator
            else:
                return obj

        def _collect_objects(obj_or_list):
            """Convert to list of objects"""
            if isinstance(obj_or_list, list):
                return [_unwrap(o) for o in obj_or_list]
            else:
                return [_unwrap(obj_or_list)]

        # Collect all circuits and operators
        circuits = _collect_objects(circuit)
        operators = _collect_objects(operator) if operator is not None else []

        def _build_param_dict(qiskit_objects):
            """Build parameter dict for list of qiskit objects"""
            param_dict = {}

            for qobj in qiskit_objects:
                for p in qobj.parameters:
                    name = p.vector.name
                    if name not in parameters:
                        continue

                    supplied = parameters[name]

                    # Normalize to numpy
                    if isinstance(supplied, (list, tuple, np.ndarray)):
                        arr = np.asarray(supplied)
                        try:
                            val = arr[p.index]
                        except (IndexError, TypeError):
                            if arr.size == 1:
                                val = arr.flat[0]
                            else:
                                raise ValueError(
                                    f"Provided values for parameter '{name}' have length {arr.size} "
                                    f"but parameter index {p.index} is requested."
                                )
                    else:
                        val = supplied

                    param_dict[p] = val

            return param_dict

        circuit_dict = _build_param_dict(circuits)
        operator_dict = _build_param_dict(operators) if operators else {}

        return circuit_dict, operator_dict

    def _extract_counts(self, pub_result, n_qubits=None):
        """
        Extract counts from the primitive result object.
        """
        # --- Qiskit 2.x ---
        # Expect an iterable of SamplerPubResult-like objects, each with data.meas.get_counts().
        if (
            hasattr(pub_result, "__iter__")
            and not isinstance(pub_result, (str, dict))
            and len(pub_result) > 0
            and hasattr(pub_result[0], "data")
        ):
            counts_list = []
            for i, pub in enumerate(pub_result):
                data = getattr(pub, "data", None)
                meas = getattr(data, "meas", None) if data is not None else None
                if meas is None or not hasattr(meas, "get_counts"):
                    raise ValueError(
                        f"Unsupported sampler result format at pub index {i}: "
                        f"'data.meas.get_counts()' is not available "
                        f"(got type {type(pub)!r})."
                    )
                counts_list.append(meas.get_counts())
            return counts_list

        # --- Qiskit 1.x ---
        # Expect an object with quasi_dists and metadata per circuit.
        if hasattr(pub_result, "quasi_dists"):
            quasi_dists = pub_result.quasi_dists
            metadata = getattr(pub_result, "metadata", None)
            if metadata is None:
                raise ValueError(
                    "Unsupported sampler result format: 'metadata' attribute is missing for quasi_dists."
                )
            counts_list = []
            for idx, qd in enumerate(quasi_dists):
                if idx >= len(metadata):
                    raise ValueError(
                        f"Unsupported sampler result format: 'metadata' has {len(metadata)} "
                        f"entries but quasi_dists has {len(quasi_dists)}."
                    )
                if "shots" not in metadata[idx]:
                    raise ValueError(
                        f"Unsupported sampler result format: 'metadata[{idx}][\"shots\"]' is missing."
                    )
                shots = metadata[idx]["shots"]
                counts = {format(k, f"0{n_qubits}b"): int(round(v * shots)) for k, v in qd.items()}
                counts_list.append(counts)
            return counts_list

        raise ValueError("Unsupported primitive result format: cannot extract counts.")

    def _expectation_value(
        self,
        circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]],
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase]],
        **parameter_values,
    ) -> Union[float, np.array]:
        """
        Calculate the expectation value using OpTree and Qiskit Estimator.

        Args:
            circuit: The quantum circuit or a list of circuits.
            operator: The quantum operator or a list of operators.
            parameter_values: Parameter values as keyword arguments.

        Returns:
            The expectation value(s).
        """
        # Convert to OpTree format
        circuit_tree, operator_tree = self._convert_to_optree(circuit, operator)

        # Prepare separate parameter dictionaries
        circuit_dict, operator_dict = self._prepare_parameter_dicts(
            circuit, operator, **parameter_values
        )

        # Use OpTree evaluation with Estimator
        result = OpTreeEvaluate.evaluate_with_estimator(
            circuit=circuit_tree,
            operator=operator_tree,
            dictionary_circuit=circuit_dict,
            dictionary_operator=operator_dict,
            estimator=self._estimator,
            dictionaries_combined=False,
            detect_duplicates=True,
        )

        return result

    def _expectation_value_derivatives(
        self,
        circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]],
        operator: Union[QuantumOperatorBase, List[QuantumOperatorBase]],
        *derivative_params,
        **parameter_values,
    ) -> Union[np.array, dict]:
        """
        Calculate the derivatives using OpTree parameter shift.

        Args:
            circuit: The quantum circuit.
            operator: The quantum operator.
            derivative_params: Parameters to differentiate with respect to.
            parameter_values: Parameter values as keyword arguments.

        Returns:
            Derivative values.
        """

        # If no derivative parameters specified, return expectation value
        if len(derivative_params) == 0:
            return self._expectation_value(circuit, operator, **parameter_values)

        # Convert to OpTree format
        circuit_tree, operator_tree = self._convert_to_optree(circuit, operator)

        # Prepare separate parameter dictionaries
        circuit_dict, operator_dict = self._prepare_parameter_dicts(
            circuit, operator, **parameter_values
        )

        # Build list of parameters to differentiate
        if isinstance(circuit, list):
            all_params = circuit[0]._qiskit_circuit.parameters
        else:
            circ = circuit._qiskit_circuit if hasattr(circuit, "_qiskit_circuit") else circuit
            all_params = circ.parameters

        params_to_diff = []
        for dp in derivative_params:
            if isinstance(dp, str):
                # Find matching parameters by name
                matching = [p for p in all_params if p.vector.name == dp]
                params_to_diff.extend(matching)
            elif isinstance(dp, ParameterVectorElement):
                params_to_diff.append(dp)
            else:
                raise ValueError(f"Unknown derivative parameter type: {type(dp)}")

        # Differentiate circuit and operator separately
        circuit_derivative = OpTreeDerivative.differentiate(circuit_tree, params_to_diff)
        operator_derivative = OpTreeDerivative.differentiate(operator_tree, params_to_diff)

        results_list = []

        num_params = len(params_to_diff)

        for i in range(num_params):
            # Extract i-th derivative
            if isinstance(circuit_derivative, OpTreeList) and len(circuit_derivative.children) > 0:
                circ_deriv_i = (
                    circuit_derivative.children[i]
                    if i < len(circuit_derivative.children)
                    else circuit_tree
                )
            else:
                circ_deriv_i = circuit_derivative if i == 0 else circuit_tree

            if (
                isinstance(operator_derivative, OpTreeList)
                and len(operator_derivative.children) > 0
            ):
                op_deriv_i = (
                    operator_derivative.children[i]
                    if i < len(operator_derivative.children)
                    else operator_tree
                )
            else:
                op_deriv_i = operator_derivative if i == 0 else operator_tree

            result1 = OpTreeEvaluate.evaluate_with_estimator(
                circuit=circ_deriv_i,
                operator=operator_tree,
                dictionary_circuit=circuit_dict,
                dictionary_operator=operator_dict,
                estimator=self._estimator,
                detect_duplicates=True,
            )

            result2 = 0.0
            if operator_tree != op_deriv_i:
                result2 = OpTreeEvaluate.evaluate_with_estimator(
                    circuit=circuit_tree,
                    operator=op_deriv_i,
                    dictionary_circuit=circuit_dict,
                    dictionary_operator=operator_dict,
                    estimator=self._estimator,
                    detect_duplicates=True,
                )

            results_list.append(result1 + result2)

        if len(derivative_params) == 1:
            return results_list[0] if len(results_list) > 0 else 0.0
        else:
            # Multiple parameters - return dict
            result_dict = {}
            for i, dp in enumerate(derivative_params):
                if i < len(results_list):
                    result_dict[dp] = results_list[i]
            return result_dict

    def _sample(
        self, circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]], **parameter_values
    ) -> List[dict]:
        """
        Sample from the circuit using OpTree and Qiskit Sampler.

        Args:
            circuit: The quantum circuit(s).
            parameter_values: Parameter values as keyword arguments.

        Returns:
            Dictionary or list of dictionaries with measurement counts.
        """
        if self._shots is None:
            raise ValueError("Shots must be set for sampling")

        # Convert to OpTree format (just for consistent handling)
        circuit_tree, _ = self._convert_to_optree(circuit, operator=None)

        # Prepare parameter dictionary (only for circuits)
        circuit_dict, _ = self._prepare_parameter_dicts(circuit, operator=None, **parameter_values)

        # Extract circuits from OpTree
        circuits = []
        if isinstance(circuit_tree, OpTreeCircuit):
            circuits = [circuit_tree.circuit]
        else:
            circuits = [child.circuit for child in circuit_tree.children]

        # Bind parameters to circuits
        bound_circuits = []
        for circ in circuits:
            # Bind only parameters that exist in this circuit
            params_to_bind = {p: circuit_dict[p] for p in circ.parameters if p in circuit_dict}

            if params_to_bind:
                bound_circ = circ.assign_parameters(params_to_bind)
            else:
                bound_circ = circ

            # Add measurements if not present
            if bound_circ.num_clbits == 0:
                bound_circ.measure_all()

            bound_circuits.append(bound_circ)

        # Run sampler – the call signature depends on the primitive version.
        #
        # * Qiskit < 1.2  → BackendSampler (V1 API): run(circuits, ...)
        # * Runtime V1 (< 0.21) → also V1 API: run(circuits, ...)
        # * Qiskit >= 1.2 with V2 primitives → run(pubs, shots=...)
        #   This applies to StatevectorSampler, BackendSamplerV2 and
        #   RuntimeSamplerV2 alike.
        use_v1_api = self._sampler_uses_v1_api
        if use_v1_api:
            # V1: pass a plain list of circuits
            job = self._sampler.run(bound_circuits, shots=self._shots)
        else:
            # V2: PUBs format – each element is a tuple (circuit,)
            pubs = [(circ,) for circ in bound_circuits]
            job = self._sampler.run(pubs, shots=self._shots)

        result = job.result()

        # Determine num_qubits per circuit for binary-string formatting
        # (needed by the V1 quasi_dists path in _extract_counts).
        is_list_input = isinstance(circuit, list)
        raw_circuits_for_nq = circuit if is_list_input else [circuit]
        n_qubits_list = [
            (c._qiskit_circuit if hasattr(c, "_qiskit_circuit") else c).num_qubits
            for c in raw_circuits_for_nq
        ]
        # Use the first value as the single n_qubits hint; _extract_counts
        # iterates over all distributions independently.
        counts_list = self._extract_counts(result, n_qubits_list[0])

        # Return a single dict for scalar input, list for list input
        if not is_list_input:
            return counts_list[0] if isinstance(counts_list, list) else counts_list
        return counts_list

    def _statevector(
        self, circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]], **parameter_values
    ) -> np.ndarray:
        """Compute the statevector of the circuit.

        Raises:
            RuntimeError: If the executor targets a remote backend where
                statevector simulation is not available.
        """
        if self._remote_backend:
            raise RuntimeError(
                "Statevector simulation is not available on remote IBM Quantum "
                "backends. Use expectation_value() or sample() instead."
            )
        return self._statevector_local(circuit, **parameter_values)

    def _statevector_local(
        self, circuit: Union[QuantumCircuitBase, List[QuantumCircuitBase]], **parameter_values
    ) -> np.ndarray:
        """
        Compute the statevector of the circuit (local only).

        Args:
            circuit: The quantum circuit(s).
            parameter_values: Parameter values as keyword arguments.

        Returns:
            Statevector(s) as numpy array(s).
        """
        # Convert to OpTree format
        circuit_tree, _ = self._convert_to_optree(circuit, operator=None)

        # Prepare parameter dictionary (only for circuits)
        circuit_dict, _ = self._prepare_parameter_dicts(circuit, operator=None, **parameter_values)

        # Extract circuits
        if isinstance(circuit_tree, OpTreeCircuit):
            circuits = [circuit_tree.circuit]
        else:
            circuits = [child.circuit for child in circuit_tree.children]

        # Compute statevectors
        statevectors = []
        for circ in circuits:
            # Bind parameters
            params_to_bind = {p: circuit_dict[p] for p in circ.parameters if p in circuit_dict}

            if params_to_bind:
                bound_circ = circ.assign_parameters(params_to_bind)
            else:
                bound_circ = circ

            # Get statevector
            sv = Statevector(bound_circ)
            statevectors.append(sv.data)

        statevectors = np.array(statevectors)

        if len(circuits) == 1:
            return statevectors[0]

        return statevectors

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
        isa_circuit = self._isa_transpile_qiskit_circuit(qc._qiskit_circuit)
        if isa_circuit is not qc._qiskit_circuit:
            return QiskitCircuit._from_qiskit(isa_circuit)
        return qc
