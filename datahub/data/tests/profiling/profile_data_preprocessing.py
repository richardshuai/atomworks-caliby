from __future__ import annotations

import cProfile
import io
import pstats

import pandas as pd

import data.data_preprocessor as data_preprocessor

data_preprocessor = data_preprocessor.DataPreprocessor()


def load_examples(pdb_ids: list[str]):
    for pdb_ids in pdbids:
        rows = data_preprocessor.get_rows(pdb_ids)
        _ = pd.DataFrame(rows)


if __name__ == "__main__":
    pdbids = ["6wjc", "1out", "1cbn"]  # List of PDB IDs to load (Difficult example: "5xnl")

    print("Profiling data preprocessing...")
    profiler = cProfile.Profile()

    # Start profiling
    profiler.enable()
    load_examples(pdbids)
    # Stop profiling
    profiler.disable()

    # Print profiling results
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())
