"""Framework-independent helpers for shaping input arrays.

``ensure_complex_coeffs`` used to live here; it works on a Qiskit
``SparsePauliOp`` and moved to ``qiskit/_compat.py`` with the rest of the
version-specific Qiskit handling, leaving this module free of Qiskit.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def resolve_parameter_batch_size(lengths: Iterable[int]) -> int:
    """Resolve and validate the batch size implied by a set of parameters.

    Args:
        lengths: The leading-axis length of each named parameter's value
            array, as produced by :func:`adjust_features`/
            :func:`adjust_parameters` (i.e. ``array.shape[0]``). A length of
            1 means that parameter was not batched - it broadcasts across
            whatever batch size the other parameters imply.

    Returns:
        int: The common batch size across every entry greater than 1 (or 1
            if none are batched).

    Raises:
        ValueError: If two parameters imply different batch sizes greater
            than 1.
    """
    sizes = {n for n in lengths if n > 1}
    if len(sizes) > 1:
        raise ValueError(
            "All batched parameters must share the same batch size, got " f"{sorted(sizes)}."
        )
    return sizes.pop() if sizes else 1


def adjust_features(x: np.ndarray | float, x_length: int) -> Tuple[np.ndarray, bool]:
    """Adjust the feature vector to the form [[]] if necessary.

    Args:
        x (np.ndarray): Input array.
        x_length (int): Dimension of the input array, e.g. feature dimension.

    Return:
        Adjusted feature array and a boolean flag for multiple inputs.
    """

    return _adjust_input(x, x_length, allow_single_array=False)


def adjust_parameters(x: np.ndarray, x_length: int) -> Tuple[np.ndarray, bool]:
    """Adjust the parameter vector to the form [[]] if necessary.

    In contrast to feature vectors, one dimensional parameters are not considered
    as multiple inputs.

    Args:
        x (np.ndarray): Input array.
        x_length (int): Dimension of the input array, e.g. feature dimension.

    Return:
        Adjusted parameter array and a boolean flag for multiple inputs.
    """

    return _adjust_input(x, x_length, allow_single_array=True)


def _adjust_input(
    x: float | np.ndarray, x_length: int, allow_single_array: bool
) -> Tuple[np.ndarray, bool]:
    """Adjust the input to the form [[]] if necessary.

    If allow_single_array is True, a one dimensional array is not considered as multiple outputs.

    Args:
        x (np.ndarray): Input array.
        x_length (int): Dimension of the input array, e.g. feature dimension.
        allow_single_array (bool): If True, a one dimensional array is not considered as
                                   multiple outputs.

    Return:
        Adjusted input array and a boolean flag for multiple inputs.
    """
    multiple_inputs = False
    shape = np.shape(x)

    if shape == () and x_length == 1:
        # Single floating point number
        xx = np.array([[x]])
    elif sum(shape) == 0 and x_length > 0:
        raise ValueError("Wrong format of an input variable.")
    elif len(shape) == 1:
        arr = np.asarray(x)
        if x_length == 1:
            xx = np.array([np.array([xi]) for xi in arr])
            multiple_inputs = shape[0] != 1 if allow_single_array else True
        elif len(arr) == x_length:
            # We have a single multi dimensional x (e.g. parameter vector)
            xx = np.array([arr])
        else:
            raise ValueError("Wrong format of an input variable.")
    elif len(shape) == 2:
        if shape[1] == x_length:
            xx = x
            multiple_inputs = True
        else:
            raise ValueError("Wrong format of an input variable.")
    else:
        raise ValueError("Wrong format of an input variable.")

    return convert_to_float64(xx), multiple_inputs


def convert_to_float64(x: float | np.ndarray | list) -> np.ndarray:
    """Convert to float64 format, raise Error for complex values

    Args:
        x (float | np.ndarray): Data that is converted

    Returns:
        Converted numpy float64 array
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if x.dtype != np.float64:
        x = np.real_if_close(x)
        if np.iscomplexobj(x):
            raise ValueError(
                "Only real values for parameters and features are supported in sQUlearn!"
            )
        x = np.array(x, dtype=np.float64)

    return x


def to_tuple(x: int | str | float | np.ndarray | list | tuple, flatten: bool = True) -> Tuple:
    """Function for converting data into hashable tuples

    Args:
        x (float | np.ndarray | list | tuple): Input data.

    Return:
        Flattened tuple of the input data
    """

    if flatten:

        def recursive_flatten(container):
            for i in container:
                if isinstance(i, (list, tuple, np.ndarray)):
                    yield from recursive_flatten(i)
                else:
                    yield i

        if isinstance(x, (float, int, str)) or not isinstance(x, (list, tuple, np.ndarray)):
            # Scalars, numpy scalars and symbols such as ``Parameter`` are
            # not containers: wrap them rather than trying to iterate them.
            return tuple([x])
        if len(np.shape(x)) == 1:
            return tuple(list(x))
        return tuple(recursive_flatten(x))

    def array_to_nested_tuple(arr):
        if isinstance(arr, (list, tuple, np.ndarray)):
            return tuple(array_to_nested_tuple(subarr) for subarr in arr)
        return arr

    if isinstance(x, (list, tuple, np.ndarray)):
        return array_to_nested_tuple(x)
    return tuple([x])
