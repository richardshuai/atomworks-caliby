from collections.abc import Callable

import biotite.structure as struc
import numpy as np
from biotite.structure import Atom, AtomArray


def array(atoms: list[Atom]) -> AtomArray:
    """Patch of Biotite's `array` function to not truncate the datatype of annotations.

    Args:
        atoms: The atoms to be combined in an array. All atoms must share the same
            annotation categories.

    Returns:
        The listed atoms as array.

    Raises:
        ValueError: If atoms do not share the same annotation categories.

    Examples:
        Creating an atom array from atoms:

        >>> atom1 = Atom([1, 2, 3], chain_id="A")
        >>> atom2 = Atom([2, 3, 4], chain_id="A")
        >>> atom3 = Atom([3, 4, 5], chain_id="B")
        >>> atom_array = array([atom1, atom2, atom3])
        >>> print(atom_array)
            A       0                       1.000    2.000    3.000
            A       0                       2.000    3.000    4.000
            B       0                       3.000    4.000    5.000
    """
    # Check if all atoms have the same annotation names
    # Equality check requires sorting
    names = sorted(atoms[0]._annot.keys())
    for i, atom in enumerate(atoms):
        if sorted(atom._annot.keys()) != names:
            raise ValueError(
                f"The atom at index {i} does not share the same " f"annotation categories as the atom at index 0"
            )
    array = AtomArray(len(atoms))

    for name in names:
        if hasattr(atoms[0]._annot[name], "dtype"):
            # (Preserve dtype if possible)
            dtype = atoms[0]._annot[name].dtype
        else:
            dtype = type(atoms[0]._annot[name])
        annotation_values = [atom._annot[name] for atom in atoms]
        annotation_values = np.array(annotation_values, dtype=dtype)  # maintain dtype
        array.set_annotation(name, annotation_values)
    array._coord = np.stack([atom.coord for atom in atoms])
    return array


def apply_and_spread(
    segment_start_stop_idxs: np.ndarray, data: np.ndarray, function: Callable, axis: int | None = None
) -> np.ndarray:
    """
    Apply a function segment-wise and then spread the result to the original data size.

    This function applies a given function to segments of the input data and then
    spreads the result back to the original data size, effectively assigning the
    segment-wise result to all elements within each segment.

    Args:
        segment_start_stop_idxs: A 1D array indicating the start and stop indices
            of each segment.  This is expected to be in the format returned by
            `biotite.structure.segments.get_segment_starts`.
        data: The input data array.
        function: The function to apply to each segment.  This function should
            take a segment of the data array as input and return a single value
            or an array of reduced values.
        axis: The axis along which to apply the function. If `None`, the function
            is applied to the entire segment.

    Returns:
        A new array with the same shape as `data`, where the result of the
        function applied to each segment has been spread across the elements
        of that segment.

    Example:
        >>> import numpy as np
        >>> segment_start_stop_idxs = np.array([0, 3, 6])
        >>> data = np.array([1, 2, 3, 4, 5, 6])
        >>> result = apply_and_spread(segment_start_stop_idxs, data, np.sum)
        >>> print(result)
        [ 6  6  6 15 15 15]
    """
    data_after_apply = struc.segments.apply_segment_wise(segment_start_stop_idxs, data, function, axis)
    data_after_spread = struc.segments.spread_segment_wise(segment_start_stop_idxs, data_after_apply)
    return data_after_spread
