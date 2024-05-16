import gzip
import io
from Bio.PDB.MMCIFParser import MMCIFParser, MMCIF2Dict
import timeit
from cifutils import cifutils_extended, cifutils_legacy, parser_utils, cifutils_biotite
import os

# Initialize parsers
cifutils_legacy_parser = cifutils_legacy.CIFParser()
biopython_parser = MMCIFParser()
cifutils_extended_parser = cifutils_extended.CIFParser()
cifutils_biotite_parser = cifutils_biotite.CIFParser()

def load_with_cifutils_legacy(pdbids):
    for pdbid in pdbids:
        filename = f'/databases/rcsb/cif/{pdbid[1:3]}/{pdbid}.cif.gz'
        with gzip.open(filename, 'rt') as file:
            cif_data = file.read()
        cifutils_legacy_parser.parse(filename)

def load_with_cifutils_extended(pdbids):
    for pdbid in pdbids:
        filename = f'/databases/rcsb/cif/{pdbid[1:3]}/{pdbid}.cif.gz'
        with gzip.open(filename, 'rt') as file:
            cif_data = file.read()
        cifutils_extended_parser.parse(filename)
        
def load_with_biopython(pdbids):
    for pdbid in pdbids:
        filename = f'/databases/rcsb/cif/{pdbid[1:3]}/{pdbid}.cif.gz'
        with gzip.open(filename, 'rt') as file:
            cif_data = file.read()
            # biopython_parser.get_structure(pdbid, io.StringIO(cif_data))
            mmcif_dict = MMCIF2Dict(io.StringIO(cif_data))

def load_with_cifutils_biotite(pdbids):
    for pdbid in pdbids:
        # filename = os.path.join("data", f"{pdbid}.bcif")
        filename = f'/databases/rcsb/cif/{pdbid[1:3]}/{pdbid}.cif.gz'
        cifutils_biotite_parser.parse(filename)

if __name__ == "__main__":
    pdbids = ["6wjc", "1out", "1cbn", "5xnl"]  # List of PDB IDs to load (Difficult example: "5xnl")

    # Benchmark using timeit
    # cifutils_extended_time = timeit.timeit(lambda: load_with_cifutils_extended(pdbids), number=5)
    # cifutils_legacy_time = timeit.timeit(lambda: load_with_cifutils_legacy(pdbids), number=5)
    biopython_time = timeit.timeit(lambda: load_with_biopython(pdbids), number=2)
    cifutils_biotite_time = timeit.timeit(lambda: load_with_cifutils_biotite(pdbids), number=2)

    # print(f"Total loading time with cifutils_legacy for all PDB IDs: {cifutils_legacy_time:.4f} seconds.")
    # print(f"Total loading time with cifutils_extended for all PDB IDs: {cifutils_extended_time:.4f} seconds.")
    print(f"Total loading time with BioPython for all PDB IDs: {biopython_time:.4f} seconds.")
    print(f"Total loading time with cifutils_biotite for all PDB IDs: {cifutils_biotite_time:.4f} seconds.")
