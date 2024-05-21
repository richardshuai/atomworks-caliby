import cProfile
import pstats
import io
from cifutils import cifutils_biotite

cifutils_biotite_parser = cifutils_biotite.CIFParser()

def load_with_cifutils_biotite(pdbids):
    for pdbid in pdbids:
        filename = f'/databases/rcsb/cif/{pdbid[1:3]}/{pdbid}.cif.gz'
        cifutils_biotite_parser.parse(filename)
        
if __name__ == "__main__":
    pdbids = ["6wjc", "1out", "1cbn", "5xnl"]  # List of PDB IDs to load (Difficult example: "5xnl")

    print("Profiling cifutils_biotite...")
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()
    load_with_cifutils_biotite(pdbids)
    # Stop profiling
    profiler.disable()

    # Print profiling results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())
