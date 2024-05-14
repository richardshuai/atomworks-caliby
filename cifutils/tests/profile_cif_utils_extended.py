import cProfile
import pstats
import io
import gzip
from Bio.PDB.MMCIFParser import MMCIFParser, MMCIF2Dict
from cifutils import cifutils_extended, cifutils_legacy

# Initialize parsers
cifutils_legacy_parser = cifutils_legacy.CIFParser()
biopython_parser = MMCIFParser()
cifutils_extended_parser = cifutils_extended.CIFParser()

def load_with_cifutils_extended(pdbids):
    for pdbid in pdbids:
        filename = f'/databases/rcsb/cif/{pdbid[1:3]}/{pdbid}.cif.gz'
        with gzip.open(filename, 'rt') as file:
            cif_data = file.read()
        cifutils_extended_parser.parse(filename)
        

if __name__ == "__main__":
    pdbids = ["5xnl"]  # Example PDB IDs

    # Create a profiler instance
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()

    # Run only the function you want to profile
    load_with_cifutils_extended(pdbids)

    # Stop profiling
    profiler.disable()

    # Create a StringIO object to print the stats to a string
    s = io.StringIO()
    
    # Sort the statistics by cumulative time and print them out
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats()

    # Print the profiling results
    print(s.getvalue())
