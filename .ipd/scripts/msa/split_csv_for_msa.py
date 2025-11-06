#!/usr/bin/env python3
"""Split a CSV file with sequences into chunks for batch MSA generation.

This script takes a CSV file with protein sequences (one per row) and splits it
into smaller chunks that can be processed in parallel via SLURM array jobs.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def split_csv_for_msa(
    input_csv: Path,
    output_dir: Path,
    chunk_size: int = 200,
    sequence_column: str | None = None,
) -> dict:
    """Split CSV into chunks and create manifest file.

    Args:
        input_csv: Path to input CSV file with sequences
        output_dir: Directory to write chunk files
        chunk_size: Number of sequences per chunk
        sequence_column: Name of column containing sequences (auto-detect if None)

    Returns:
        Dictionary with manifest information
    """
    # Create output directory
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV
    print(f"Reading {input_csv}...")
    df = pd.read_csv(input_csv)

    # Determine sequence column
    if sequence_column is None:
        if len(df.columns) != 1:
            raise ValueError(
                f"CSV has {len(df.columns)} columns. " f"Either provide exactly 1 column or specify --sequence-column"
            )
        sequence_column = df.columns[0]

    if sequence_column not in df.columns:
        raise ValueError(f"Column '{sequence_column}' not found in CSV. Available: {list(df.columns)}")

    # Get unique sequences
    sequences = df[sequence_column].dropna().unique()
    total_sequences = len(sequences)

    print(f"Found {total_sequences:,} unique sequences in column '{sequence_column}'")

    # Calculate number of chunks
    num_chunks = (total_sequences + chunk_size - 1) // chunk_size  # Ceiling division
    print(f"Splitting into {num_chunks:,} chunks of ~{chunk_size} sequences each")

    # Split and write chunks
    chunk_files = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_sequences)
        chunk_sequences = sequences[start_idx:end_idx]

        # Create chunk DataFrame
        chunk_df = pd.DataFrame({sequence_column: chunk_sequences})

        # Write chunk file (1-indexed for SLURM array jobs)
        chunk_filename = f"chunk_{i+1:04d}.csv"
        chunk_path = chunks_dir / chunk_filename
        chunk_df.to_csv(chunk_path, index=False)
        chunk_files.append(chunk_filename)

        print(f"  Created {chunk_filename}: {len(chunk_sequences)} sequences")

    # Create manifest file
    manifest = {
        "input_csv": str(input_csv.absolute()),
        "output_dir": str(output_dir.absolute()),
        "chunks_dir": str(chunks_dir.absolute()),
        "sequence_column": sequence_column,
        "total_sequences": int(total_sequences),
        "chunk_size": chunk_size,
        "num_chunks": num_chunks,
        "chunk_files": chunk_files,
    }

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest written to {manifest_path}")
    print(f"\nReady to submit SLURM array job with --array=1-{num_chunks}")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Split CSV into chunks for batch MSA generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split into default chunks of 200 sequences
  python split_csv_for_msa.py sequences.csv output/

  # Custom chunk size
  python split_csv_for_msa.py sequences.csv output/ --chunk-size 500

  # Specify sequence column for multi-column CSV
  python split_csv_for_msa.py data.csv output/ --sequence-column seq
        """,
    )

    parser.add_argument("input_csv", type=Path, help="Input CSV file with sequences")
    parser.add_argument("output_dir", type=Path, help="Output directory for chunks")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of sequences per chunk (default: 200)",
    )
    parser.add_argument(
        "--sequence-column",
        "-c",
        type=str,
        default=None,
        help="Name of column containing sequences (auto-detect if not specified)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    if args.chunk_size < 1:
        raise ValueError(f"Chunk size must be >= 1, got {args.chunk_size}")

    # Split CSV
    split_csv_for_msa(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        sequence_column=args.sequence_column,
    )


if __name__ == "__main__":
    main()
