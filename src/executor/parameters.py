"""Parameter and ParameterVector types for parameterised quantum circuits."""

from __future__ import annotations

from typing import TypeAlias
from uuid import UUID, uuid4

from qiskit.circuit.parametervector import ParameterVectorElement as QiskitParameterVectorElement

#: Library-native parameter type. Alias for Qiskit's ``ParameterVectorElement``,
#: decoupling user-facing code from the Qiskit class hierarchy.
Parameter: TypeAlias = QiskitParameterVectorElement


class Parameters:
    """An ordered, resizable vector of :class:`Parameter` instances.

    Behaves like a sequence: supports indexing, iteration, and ``len()``.
    """

    def __init__(self, name, length=0):
        self._name = name
        self._root_uuid = uuid4()
        root_uuid_int = self._root_uuid.int
        self._params = [Parameter(self, i, UUID(int=root_uuid_int + i)) for i in range(length)]

    @property
    def name(self):
        """The name of the :class:`ParameterVector`."""
        return self._name

    @property
    def params(self):
        """A list of the contained :class:`ParameterVectorElement` instances.

        It is not safe to mutate this list."""
        return self._params

    def index(self, value):
        """Find the index of a :class:`ParameterVectorElement` within the list.

        It is typically much faster to use the :attr:`ParameterVectorElement.index` property."""
        return self._params.index(value)

    def __getitem__(self, key):
        return self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self._params)

    def __str__(self):
        return f"{self.name}, {[str(item) for item in self.params]}"

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)}, length={len(self)})"

    def resize(self, length):
        """Resize the parameter vector.  If necessary, new elements are generated.

        Note that the UUID of each :class:`.Parameter` element will be generated
        deterministically given the root UUID of the ``ParameterVector`` and the index
        of the element.  In particular, if a ``ParameterVector`` is resized to
        be smaller and then later resized to be larger, the UUID of the later
        generated element at a given index will be the same as the UUID of the
        previous element at that index.
        This is to ensure that the parameter instances do not change.

        >>> from qiskit.circuit import ParameterVector
        >>> pv = ParameterVector("theta", 20)
        >>> elt_19 = pv[19]
        >>> rv.resize(10)
        >>> rv.resize(20)
        >>> pv[19] == elt_19
        True
        """
        if length > len(self._params):
            root_uuid_int = self._root_uuid.int
            self._params.extend(
                [
                    Parameter(self, i, UUID(int=root_uuid_int + i))
                    for i in range(len(self._params), length)
                ]
            )
        else:
            del self._params[length:]
