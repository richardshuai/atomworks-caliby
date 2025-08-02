import os
import sys

import pandas as pd
import tqdm
from cifutils.enums import ChainType

from datahub.utils import hash_sequence

# extract sequence and hash from all csv file


def extract_sequence_from_csv(Dataroot_path):
    protein_sequence_dict = {}
    for file in tqdm.tqdm(list(os.listdir(Dataroot_path))):
        if file.endswith(".csv"):
            try:
                df = pd.read_csv(os.path.join(Dataroot_path, file))
            except pd.errors.EmptyDataError:
                print(f"found no sequence in {file}")
                continue
            for row in df.itertuples():
                # only consider protein sequence
                if row.q_pn_unit_type == ChainType.POLYPEPTIDE_L.value:
                    protein_sequence_dict[str(hash_sequence(row.q_pn_unit_processed_entity_canonical_sequence))] = (
                        row.q_pn_unit_processed_entity_canonical_sequence
                    )

    # according to the first 2 digits of the hash,
    # devide into subgroups
    hash_dict = {}
    for hash_ in protein_sequence_dict.keys():
        if str(hash_)[:2] not in hash_dict:
            hash_dict[str(hash_)[:2]] = []
        hash_dict[str(hash_)[:2]].append(hash_)

    return protein_sequence_dict, hash_dict


def store_fasta_files(protein_sequence_dict, hash_dict, fasta_store_path):
    # for each subgroup, generate a fasta file
    for key in tqdm.tqdm(hash_dict.keys()):
        with open(f"{fasta_store_path}/{key}.fasta", "w") as f:
            # sort hash_dict[key] according to the length of the sequence
            hash_dict[key].sort(key=lambda x: len(protein_sequence_dict[x]))
            for hash_ in hash_dict[key]:
                f.write(f"> {hash_}\n{protein_sequence_dict[hash_]}\n")


def main():
    Dataroot_path = sys.argv[1]
    fasta_store_path = sys.argv[2]
    protein_sequence_dict, hash_dict = extract_sequence_from_csv(Dataroot_path)
    store_fasta_files(protein_sequence_dict, hash_dict, fasta_store_path)


if __name__ == "__main__":
    main()
