class CircuitIdentityMixin:
    """
    Mixin class to provide content-based hashing for quantum circuits.
    By default, it falls back to object identity (i.e., different instances are not equal).
    Subclasses can override the `_circuit_hash_key` method to provide a content-based hash key.
    """

    def _circuit_hash_key(self) -> tuple:
        """
        Returns a tuple that uniquely identifies the circuit content.

        Override this in subclasses to provide content-based hashing.

        NOTE: Current implementations are Qiskit-based since Qiskit is the
        underlying workhorse of the framework. Once Qiskit is replaced by an
        own abstract circuit representation, this method should be updated
        accordingly in all subclasses.
        """
        raise NotImplementedError

    def __hash__(self) -> int:
        return hash(self._circuit_hash_key())
