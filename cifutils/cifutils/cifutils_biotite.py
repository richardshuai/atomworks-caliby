"""
Implementation of original `citufils` functions using `biotite` library.
Retains all functionality and of orginal library, with increased performance. 
"""

import gzip
import os
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.pdbx as pdbx
import biotite.database.rcsb as rcsb
import pandas as pd
import numpy as np

class CIFParser:
    def __init__(self):
        pass

    def read_cif_file(self, filename):
        """
        Reads a CIF, BCIF, or gzipped CIF/BCIF file and returns its contents.

        Parameters:
        filename (str): The path to the file to be read.

        Returns:
        cif_file: The contents of the CIF or BCIF file as read by the appropriate parser.
        """
        file_ext = os.path.splitext(filename)[-1]
        
        if file_ext == '.gz':
            with gzip.open(filename, 'rt') as f:
                # Handle gzipped CIF files
                if filename.endswith('.cif.gz'):
                    cif_file = pdbx.CIFFile.read(f)
                elif filename.endswith('.bcif.gz'):
                    with gzip.open(filename, 'rb') as bf:
                        cif_file = pdbx.BinaryCIFFile.read(bf)
                else:
                    raise ValueError("Unsupported file format for gzip compressed file")
        elif file_ext == '.bcif':
            # Handle BinaryCIF files
            cif_file = pdbx.BinaryCIFFile.read(filename)
        elif file_ext == '.cif':
            # Handle plain CIF files
            cif_file = pdbx.CIFFile.read(filename)
        else:
            raise ValueError("Unsupported file format")
        
        return cif_file


    def parse(self, filename):
        """
        Parse the CIF file and return chain information, atom array stack, and metadata.

        Parameters:
        filename (str): Path to the CIF file. Must be a binary CIF file.

        Returns:
        dict: chain_information containing sequence details of each chain.
        AtomArrayStack: All atoms and coordinates.
        dict: metadata including method, date, and resolution.
        """
        cif_file = self.read_cif_file(filename)
        cif_block = cif_file.block

        # Load modified residues
        modified_residues_df = self._category_to_df(cif_block, 'pdbx_struct_mod_residue')

        # Load structure
        # We first load using the rcsb labels for sequence ids, and later update for non-polymers
        atom_array_stack = pdbx.get_structure(
            cif_block, 
            extra_fields=[
                'label_entity_id', 
                'auth_seq_id', 
                "atom_id", 
                "b_factor", 
                "occupancy",
                "label_alt_id",
                "charge"
            ], 
            use_author_fields=False, 
            altloc='occupancy',
            model=1
        )

        # Load chain information (uses atom_array_stack to build chain list)
        chain_information_df = self._get_chain_information(cif_block, atom_array_stack)

        # Replace non-polymeric chain sequence ids with author sequence ids
        self._update_nonpoly_seq_ids(atom_array_stack, chain_information_df)

        # Handle sequence heterogeneity by selecting the residue that appears last
        atom_array_stack = self._keep_last_residue(atom_array_stack)

        # Load bond information
        atom_array_stack.bonds = struc.connect_via_residue_names(atom_array_stack)

        # Load metadata
        metadata = self._get_metadata(cif_block)
        
        return chain_information_df, atom_array_stack, metadata, modified_residues_df

    def _category_to_df(self, cif_block, category):
        """
        Convert a CIF block to a pandas DataFrame.

        Parameters:
        - cif_block (CIFBlock): The parsed CIF block.
        - category (str): The category of the CIF block to convert.

        Returns:
        pd.DataFrame: A DataFrame containing the information from the CIF block.
        If the specified category does not exist in the CIF block, None is returned.
        """
        if category in cif_block.keys():
            return pd.DataFrame({key: column.as_array() for key, column in cif_block[category].items()})
        else:
            return None

    def _keep_last_residue(self, atom_array_stack):
        """
        Removes duplicate residues in the atom array stack, keeping only the last occurrence.

        This function creates a DataFrame from the atom array stack, identifies duplicate residues based on the combination of chain_id, res_id, and res_name, and removes them from the atom array stack.

        Parameters:
        atom_array_stack (AtomArrayStack): The atom array stack containing the chain information.

        Returns:
        AtomArrayStack: The atom array stack with duplicate residues removed.
        """
        atom_df = pd.DataFrame({
            'chain_id': atom_array_stack.chain_id,
            'res_id': atom_array_stack.res_id,
            'res_name': atom_array_stack.res_name,
        })

        # Get the mask of duplicates based on the combination of chain_id, res_id, and res_name
        collapsed_df = atom_df.drop_duplicates(subset=['chain_id', 'res_id', 'res_name'])

        # Get duplicates based on res_id, keeping the last
        duplicate_mask = collapsed_df.duplicated(subset=['chain_id', 'res_id'], keep='last')
        duplicates_df = collapsed_df[duplicate_mask]

        # Perform a left merge to find rows in atom_df that are also in duplicates_df
        merged_df = atom_df.merge(duplicates_df, on=['chain_id', 'res_id', 'res_name'], how='left', indicator=True)

        # Create a mask where True indicates the row is not in duplicates_df
        mask = merged_df['_merge'] == 'left_only'

        # Remove rows from atom_array_stack with the deletion mask
        return atom_array_stack[mask]

    def _update_nonpoly_seq_ids(self, atom_array_stack, chain_information_df):
        """
        Updates the sequence IDs of non-polymeric chains in the atom array stack.

        This method replaces the sequence IDs of non-polymeric chains in the atom array stack with the author sequence IDs.

        Parameters:
        atom_array_stack (AtomArrayStack): The atom array stack containing the chain information.
        chain_information_df (DataFrame): DataFrame containing the sequence details of each chain.

        Returns:
        None: This method updates the atom array stack in-place and does not return anything.
        """
        
        # For non-polymeric chains, we use the author sequence ids
        author_seq_ids = atom_array_stack.get_annotation('auth_seq_id')
        chain_ids = atom_array_stack.get_annotation('chain_id')

        # Step 1: Convert chain_ids array to a DataFrame
        chain_ids_df = pd.DataFrame(chain_ids, columns=['chain_id'])

        # Step 2: Merge with chain_information_df on chain_id
        merged_df = chain_ids_df.merge(chain_information_df[['chain_id', 'is_polymer']], on='chain_id', how='left')

        # Step 3: Create the mask based on the is_polymer column
        non_polymer_mask = ~merged_df['is_polymer'].fillna(True)  # Fill NaN with True and invert the boolean mask

        # Step 4: Update the atom_array_stack_label with the author sequence ids
        atom_array_stack.res_id[non_polymer_mask] = author_seq_ids[non_polymer_mask]

    def _get_chain_information(self, cif_block, atom_array_stack):
        """
        Extracts chain information from the CIF block.

        Parameters:
        cif_block (CIFBlock): Parsed CIF block.
        atom_array_stack (AtomArrayStack): Atom array stack containing the chain information.

        Returns:
        DataFrame: DataFrame containing the sequence details of each chain.
        """

        # Step 1: Build a dataframe mapping chain id to entity id from the `atom_site` 
        chain_information_df = pd.DataFrame({
            'chain_id': atom_array_stack.get_annotation('chain_id'),
            'entity_id': atom_array_stack.get_annotation('label_entity_id')
        })

        # Convert entity_id to int
        chain_information_df['entity_id'] = chain_information_df['entity_id'].astype(str)
        chain_information_df = chain_information_df.drop_duplicates(subset='chain_id').reset_index(drop=True)

        # Step 2: Load additional chain information
        entity_df = self._category_to_df(cif_block, 'entity')
        entity_df['id'] = entity_df['id'].astype(str)
        entity_df.rename(columns={'type': 'entity_type'}, inplace=True)

        polymer_df = self._category_to_df(cif_block, 'entity_poly')
        # polymer_seq_df = self._category_to_df(cif_block, 'entity_poly_seq')

        # Step 3: Merge additional information with chain_df on entity_id
        chain_information_df = chain_information_df.merge(entity_df, left_on='entity_id', right_on='id', how='left').drop(columns=['id'])
        polymer_df = polymer_df[['entity_id', 'type', 'pdbx_seq_one_letter_code', 'pdbx_seq_one_letter_code_can', 'pdbx_strand_id']]
        polymer_df.rename(columns={'type': 'polymer_type', 'pdbx_seq_one_letter_code': 'non_canonical_sequence', 'pdbx_seq_one_letter_code_can': 'canonical_sequence'}, inplace=True)
        polymer_df['entity_id'] = polymer_df['entity_id'].astype(str)
        chain_information_df = chain_information_df.merge(polymer_df, left_on='entity_id', right_on='entity_id', how='left')
        chain_information_df['chain_type'] = np.where(chain_information_df['entity_type'] == 'polymer', chain_information_df['polymer_type'], chain_information_df['entity_type'])
        chain_information_df['is_polymer'] = chain_information_df['entity_type'] == 'polymer'
        chain_information_df = chain_information_df.drop(columns=['polymer_type', 'entity_type'])

        return chain_information_df

    def _get_metadata(self, cif_block):
        """
        Extract metadata from the CIF block.

        Parameters:
        cif_block (CIFBlock): Parsed CIF block.

        Returns:
        dict: Dictionary containing metadata information.
        """
        metadata = {}
        exptl = cif_block["exptl"] if "exptl" in cif_block.keys() else None
        status = cif_block["pdbx_database_status"] if "pdbx_database_status" in cif_block.keys() else None
        refine = cif_block["refine"] if "refine" in cif_block.keys() else None
        em_reconstruction = cif_block["em_3d_reconstruction"] if "em_3d_reconstruction" in cif_block.keys() else None

        if exptl:
            metadata["method"] = exptl["method"].as_item().replace(' ', '_')
        if status:
            metadata["date"] = status["recvd_initial_deposition_date"].as_item()
        if refine:
            try:
                metadata["resolution"] = float(refine["ls_d_res_high"].as_item())
            except (KeyError, ValueError):
                metadata["resolution"] = None
        if em_reconstruction and "resolution" not in metadata:
            try:
                metadata["resolution"] = float(em_reconstruction["resolution"].as_item())
            except (KeyError, ValueError):
                metadata["resolution"] = None

        return metadata

if __name__ == "__main__":
    parser = CIFParser()
    pdb_ids = ["1cbn", "6dmh", "1en2", "1ivo", "4js1", "1zy8", "1lys"]
    # pdb_ids = ["6wjc", "1out", "1cbn", "5xnl"]
    for ipdb_id in pdb_ids:
        pdb_id = ipdb_id.lower()
        filename = os.path.join("data", f"{pdb_id}.bcif")
        # Check if filename exists
        if not os.path.exists(filename):
            # Download the file if it doesn't exist
            print(f"File {filename} not found. Downloading...")
            file = rcsb.fetch(pdb_id, "bcif")
            with open(filename, 'wb') as f:
                f.write(file.getvalue())
            print(f"File {filename} downloaded.")

        chain_information, atom_array_stack, metadata, modified_residues_df = parser.parse(filename)
        # print(chain_information)
        # print(atom_array_stack)
        # print(metadata)
        # print("Done.")