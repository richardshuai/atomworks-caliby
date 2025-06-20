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

    # Add all (also optional) annotation categories, preserving dtype if possible
    for name in names:
        if hasattr(atoms[0]._annot[name], "dtype"):
            # (Preserve dtype if possible)
            dtype = atoms[0]._annot[name].dtype
        else:
            dtype = type(atoms[0]._annot[name])
        array.add_annotation(name, dtype=dtype)

    # Add all atoms to AtomArray
    for i in range(len(atoms)):
        for name in names:
            array._annot[name][i] = atoms[i]._annot[name]
        array._coord[i] = atoms[i].coord
    return array
