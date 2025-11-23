#!/usr/bin/env python
"""Merge ASE validation results from multiple chunks into a single file.

Usage:
    python merge_results.py --input-dir ase_validation_results --output validation_merged.parquet

    # Merge with an existing merged file (incremental merge):
    python merge_results.py --input-dir ase_validation_results --output new_merged.parquet \
        --existing-merged validation_merged_11_22_2025_14_00.parquet
"""

import argparse
from pathlib import Path

import pandas as pd

# PB checks to EXCLUDE from pass/fail calculation
EXCLUDED_PB_CHECKS = {"pb_all_atoms_connected"}


def merge_results(
    input_dir: str,
    output_file: str,
    existing_merged: str | None = None,
) -> None:
    """Merge all chunk parquet files into single result file.

    Args:
        input_dir: Directory containing chunk_*.parquet files.
        output_file: Path for merged output parquet file.
        existing_merged: Optional path to a pre-existing merged parquet file.
            Data from this file will be included in the final merge.
    """
    input_path = Path(input_dir)

    # Find all chunk files
    chunk_files = sorted(input_path.glob("chunk_*.parquet"))

    if not chunk_files and not existing_merged:
        print(f"ERROR: No chunk_*.parquet files found in {input_dir}")
        return

    print(f"Found {len(chunk_files)} chunk files")

    # Read and concatenate
    dfs = []

    # Load existing merged file first if provided
    if existing_merged:
        existing_path = Path(existing_merged)
        if existing_path.exists():
            try:
                df = pd.read_parquet(existing_path)
                if len(df) > 0:
                    dfs.append(df)
                    print(f"Loaded {len(df):,} entries from existing merged file: {existing_merged}")
            except Exception as e:
                print(f"WARNING: Failed to load existing merged file: {e}")
        else:
            print(f"WARNING: Existing merged file not found: {existing_merged}")

    print("Reading chunk files...")
    for chunk_file in chunk_files:
        try:
            df = pd.read_parquet(chunk_file)
            if len(df) > 0:
                dfs.append(df)
                print(f"  {chunk_file.name}: {len(df):,} entries")
        except Exception as e:
            print(f"  {chunk_file.name}: ERROR - {e}")

    if not dfs:
        print("ERROR: No valid data found in chunk files")
        return

    # Concatenate all chunks
    merged = pd.concat(dfs, ignore_index=True)

    # Remove duplicates (keep first occurrence)
    initial_count = len(merged)
    merged = merged.drop_duplicates(subset=["lmdb_idx"], keep="first")
    duplicates_removed = initial_count - len(merged)

    # Sort by index
    merged = merged.sort_values("lmdb_idx").reset_index(drop=True)

    # Calculate passes_stereochemical_validation if not present
    # (for backwards compatibility with older chunk files)
    if "passes_stereochemical_validation" not in merged.columns:
        print("\nCalculating passes_stereochemical_validation column...")
        pb_cols = [c for c in merged.columns if c.startswith("pb_") and c not in ["pb_bust_failed"]]
        relevant_cols = [c for c in pb_cols if c not in EXCLUDED_PB_CHECKS]

        def calc_stereo_validation(row):
            # If pb_bust_failed, consider it passing (metals, etc.)
            if row.get("pb_bust_failed", False):
                return True
            # If rdkit conversion failed, it fails
            if row.get("rdkit_conversion_failed", False):
                return False
            # Check all relevant PB columns
            for col in relevant_cols:
                if col in row and not row[col]:
                    return False
            return True

        merged["passes_stereochemical_validation"] = merged.apply(calc_stereo_validation, axis=1)

    # Save merged results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print("Merged Results Summary")
    print(f"{'=' * 60}")
    print(f"Total entries: {len(merged):,}")
    print(f"Duplicates removed: {duplicates_removed:,}")
    print(f"Index range: {merged['lmdb_idx'].min()} - {merged['lmdb_idx'].max()}")
    print()

    # RDKit conversion stats
    rdkit_failed = merged["rdkit_conversion_failed"].sum()
    rdkit_success = len(merged) - rdkit_failed
    rdkit_rate = rdkit_success / len(merged) * 100
    print("RDKit conversion:")
    print(f"  Success: {rdkit_success:,} / {len(merged):,} ({rdkit_rate:.1f}%)")
    print(f"  Failed:  {rdkit_failed:,}")
    print()

    # PoseBusters stats
    pb_failed = merged["pb_bust_failed"].sum()
    pb_success = len(merged) - pb_failed
    pb_rate = pb_success / len(merged) * 100
    print("PoseBusters validation:")
    print(f"  Ran successfully: {pb_success:,} / {len(merged):,} ({pb_rate:.1f}%)")
    print(f"  Failed (metals, etc.): {pb_failed:,}")
    print()

    # Stereochemical validation (the main metric)
    stereo_passed = merged["passes_stereochemical_validation"].sum()
    stereo_rate = stereo_passed / len(merged) * 100
    print("Stereochemical validation (main metric):")
    print(f"  PASSED: {stereo_passed:,} / {len(merged):,} ({stereo_rate:.1f}%)")
    print(f"  (Excludes pb_all_atoms_connected; pb_bust_failed counts as pass)")
    print()

    # Detailed PB check stats (excluding pb_all_atoms_connected)
    pb_cols = [c for c in merged.columns if c.startswith("pb_") and c not in ["pb_bust_failed"]]
    relevant_cols = [c for c in pb_cols if c not in EXCLUDED_PB_CHECKS]

    if relevant_cols:
        # Filter to molecules where PB ran successfully
        pb_ran = merged[~merged["rdkit_conversion_failed"] & ~merged["pb_bust_failed"]]
        if len(pb_ran) > 0:
            print(f"Per-check pass rates (of {len(pb_ran):,} molecules where PB ran):")
            for col in sorted(relevant_cols):
                if col in pb_ran.columns:
                    passed = pb_ran[col].sum()
                    rate = passed / len(pb_ran) * 100
                    print(f"  {col}: {passed:,}/{len(pb_ran):,} ({rate:.1f}%)")

            # Show excluded check separately
            if "pb_all_atoms_connected" in pb_ran.columns:
                print()
                print("Excluded from stereochemical validation:")
                passed = pb_ran["pb_all_atoms_connected"].sum()
                rate = passed / len(pb_ran) * 100
                print(f"  pb_all_atoms_connected: {passed:,}/{len(pb_ran):,} ({rate:.1f}%)")

    print()
    print(f"Saved to: {output_file}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ASE validation chunk files")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./ase_validation_results",
        help="Directory containing chunk_*.parquet files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./ase_validation_results/validation_merged.parquet",
        help="Output path for merged parquet file",
    )
    parser.add_argument(
        "--existing-merged",
        type=str,
        default=None,
        help="Path to existing merged parquet file to include in merge",
    )
    args = parser.parse_args()

    merge_results(args.input_dir, args.output, args.existing_merged)


if __name__ == "__main__":
    main()
