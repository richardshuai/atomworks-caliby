"""
Utility functions for computing bioassemblies based on rototranslations of the asymmetric unit.
"""

import biotite.structure.io.pdbx as pdbx
from biotite.structure.io.pdbx import CIFCategory
from biotite.structure import AtomArrayStack
from typing import Literal
from cifutils.transforms.atom_array import (
    maybe_patch_non_polymer_at_symmetry_center,
    add_iid_annotations_to_assemblies,
)
import numpy as np
from biotite.structure.atoms import repeat


def _matrix_rotate(v: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Perform a rotation using a rotation matrix.

    Args
        v (ndarray): The coordinates to rotate.
        matrix (ndarray): The rotation matrix.

    Returns:
        ndarray: The rotated coordinates.
    """
    # For proper rotation reshape into a maximum of 2 dimensions
    orig_ndim = v.ndim
    if orig_ndim > 2:
        orig_shape = v.shape
        v = v.reshape(-1, 3)
    # Apply rotation
    v = np.dot(matrix, v.T).T
    # Reshape back into original shape
    if orig_ndim > 2:
        v = v.reshape(*orig_shape)
    return v


def _parse_transformations(struct_oper: CIFCategory) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Get transformation operation in terms of rotation matrix and
    translation for each operation ID in `pdbx_struct_oper_list`.

    Copied from: https://github.com/biotite-dev/biotite/blob/v0.40.0/src/biotite/structure/io/pdbx/convert.py#L1398
    """
    transformation_dict = {}
    for index, id in enumerate(struct_oper["id"].as_array(str)):
        rotation_matrix = np.array(
            [[struct_oper[f"matrix[{i}][{j}]"].as_array(float)[index] for j in (1, 2, 3)] for i in (1, 2, 3)]
        )
        translation_vector = np.array([struct_oper[f"vector[{i}]"].as_array(float)[index] for i in (1, 2, 3)])
        transformation_dict[id] = (rotation_matrix, translation_vector)
    return transformation_dict


def _apply_assembly_transformation(
    structure: AtomArrayStack, transformation_dict: dict, operation: tuple[str]
) -> AtomArrayStack:
    """
    Get subassembly by applying the given operation to the input
    structure containing affected asym IDs.

    Modified from: https://github.com/biotite-dev/biotite/blob/v0.40.0/src/biotite/structure/io/pdbx/convert.py#L1398
    """
    coord = structure.coord
    # Execute for each transformation step
    # in the operation expression
    for op_step in operation:
        rotation_matrix, translation_vector = transformation_dict[op_step]
        # Rotate
        coord = _matrix_rotate(coord, rotation_matrix)
        # Translate
        coord += translation_vector

    # Add a dimension to coord to match expected shape or `repeat` (first dimension is # repeats)
    coord = coord[np.newaxis, ...]

    return repeat(structure, coord)


def _build_bioassembly_from_asym_unit(
    assembly_gen_category: CIFCategory,
    struct_oper_category: CIFCategory,
    atom_array_stack: AtomArrayStack,
    assembly_ids: Literal["all", "first"] | list[str] = "first",
) -> AtomArrayStack:
    """
    Build the first biological assembly found within the mmCIF file and update the `transformation_id` annotation.

    Code modified from: https://github.com/biotite-dev/biotite/blob/v0.40.0/src/biotite/structure/io/pdbx/convert.py#L1318

    Args:
        cif_block (CIFBlock): The CIF block containing the structure data.
        atom_array_stack (AtomArrayStack): The atom array stack to which the transformations will be applied.
        assembly_id (int, optional): The ID of the assembly to build. Defaults to None, which means the first assembly will be built.

    Returns:
        AtomArray: The atom array with the biological assembly built and transformation_id annotations updated.
    """

    # Parse CIF blocks and select assembly (either by passed assembly_id or the first assembly)
    available_assembly_ids = assembly_gen_category["assembly_id"].as_array(str)

    # parse `assembly_ids` option
    if assembly_ids == "first":
        to_build = [available_assembly_ids[0]]
    elif assembly_ids == "all":
        to_build = available_assembly_ids
    else:
        to_build = assembly_ids

    # ensure instructions for each requested assembly id exist
    if not all(_id in available_assembly_ids for _id in to_build):
        raise ValueError(
            f"Invalid assembly ID(s) provided: {to_build}. Available assembly IDs: {available_assembly_ids}"
        )

    # get the transformations and apply to affected asym IDs
    transformations = _parse_transformations(struct_oper_category)  # {id: rotation, translation}
    assemblies = {}
    for _id, op_expr, asym_id_expr in zip(
        assembly_gen_category["assembly_id"].as_array(str),
        assembly_gen_category["oper_expression"].as_array(str),
        assembly_gen_category["asym_id_list"].as_array(str),
    ):
        # Find the operation expressions for given assembly ID
        if _id in to_build:
            operations = pdbx.convert._parse_operation_expression(op_expr)
            asym_ids = asym_id_expr.split(",")
            # Filter affected asym IDs
            sub_structure = atom_array_stack[..., np.isin(atom_array_stack.chain_id, asym_ids)]
            for operation in operations:
                sub_assembly = _apply_assembly_transformation(sub_structure, transformations, operation)
                # Add transformation ID annotation (e.g., 1 for identity operation)
                if len(operation) > 1:
                    # Rarely, operation expressions will have multiple elements defining their name
                    # (e.g. ('1', 'X0') for `2fs3`), in this case we combine them into a single string
                    # for referencing the operation later on
                    operation = "".join(operation)
                sub_assembly.set_annotation("transformation_id", np.full(sub_assembly.array_length(), operation))
                # Merge the chains with asym IDs for this operation with chains from other operations
                assemblies[_id] = assemblies[_id] + sub_assembly if _id in assemblies else sub_assembly

    return assemblies


def process_assemblies(
    assembly_gen_category: CIFCategory,
    struct_oper_category: CIFCategory,
    atom_array_stack: AtomArrayStack,
    patch_symmetry_centers: bool,
    build_assembly: Literal["all", "first"] | list[str] = "first",
) -> None:
    """
    Processes the assemblies from the CIF file and updates the data_dict accordingly.

    Args:
        assembly_gen_category (CIFCategory): The `pdbx_struct_assembly_gen` category from the CIF file.
        struct_oper_category (CIFCategory): The `pdbx_struct_oper_list` category from the CIF file.
        atom_array_stack (AtomArrayStack): The atom array stack to which the transformations will be applied.
        patch_symmetry_centers (bool): Flag to indicate if symmetry centers should be patched.
        build_assembly (str, optional): The assembly ID to build. Defaults to "first".

    Returns:
        assemblies (dict[str, AtomArrayStack]): The dictionary containing the built assemblies.
    """
    assemblies = _build_bioassembly_from_asym_unit(
        assembly_gen_category=assembly_gen_category,
        struct_oper_category=struct_oper_category,
        atom_array_stack=atom_array_stack,
        assembly_ids=build_assembly,
    )

    # Add instance-level (iid) annotations for chain, PN unit, and molecule
    assemblies = add_iid_annotations_to_assemblies(assemblies)

    # Optionally, patch symmetry centers for non-polymer residues that clash with themselves
    if patch_symmetry_centers and len(assemblies) > 0:
        for idx, assembly in assemblies.items():
            assemblies[idx] = maybe_patch_non_polymer_at_symmetry_center(assembly)

    return assemblies
