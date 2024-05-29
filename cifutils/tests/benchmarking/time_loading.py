from __future__ import annotations
import timeit
from cifutils.cifutils_biotite import cifutils_biotite
from cifutils.cifutils_legacy import cifutils_legacy

# Initialize parsers
cifutils_legacy_parser = cifutils_legacy.CIFParser()
cifutils_biotite_parser = cifutils_biotite.CIFParser(add_bonds=True, add_missing_atoms=True, build_assembly=False)


def load_with_cifutils_legacy(pdbids):
    for pdbid in pdbids:
        filename = f"/databases/rcsb/cif/{pdbid[1:3]}/{pdbid}.cif.gz"
        cifutils_legacy_parser.parse(filename)


def load_with_cifutils_biotite(pdbids):
    for pdbid in pdbids:
        # filename = os.path.join("data", f"{pdbid}.bcif")
        filename = f"/databases/rcsb/cif/{pdbid[1:3]}/{pdbid}.cif.gz"
        cifutils_biotite_parser.parse(filename)


if __name__ == "__main__":
    pdbids = ["6wjc", "1out", "1cbn", "5xnl"]  # List of PDB IDs to load (Difficult example: "5xnl")

    # Benchmark using timeit
    print("Benchmarking for cifutils_legacy...")
    cifutils_legacy_time = timeit.timeit(lambda: load_with_cifutils_legacy(pdbids), number=2)
    print("Benchmarking for cifutils_biotite...")
    cifutils_biotite_time = timeit.timeit(lambda: load_with_cifutils_biotite(pdbids), number=2)

    print(f"Total loading time with cifutils_legacy for all PDB IDs: {cifutils_legacy_time:.4f} seconds.")
    print(f"Total loading time with cifutils_biotite for all PDB IDs: {cifutils_biotite_time:.4f} seconds.")
