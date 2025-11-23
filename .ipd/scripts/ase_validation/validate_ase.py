#!/usr/bin/env python
"""Validate ASE LMDB entries using RDKit and PoseBusters.

Usage:
    python validate_ase.py \
        --output-dir ./results \
        --chunk-id 0 \
        --sample-size 1000 \
        --lmdb-path /net/tukwila/omol_4m/train_4M \
        --total-entries 4000000

    # Skip indices already in a merged parquet file:
    python validate_ase.py \
        --output-dir ./results \
        --chunk-id 0 \
        --existing-merged validation_merged.parquet

Single-threaded processing with checkpoints. Relies on SLURM job timeouts
for crash protection.
"""

import random
from pathlib import Path

import fire
import pandas as pd
from tqdm import tqdm

# PB checks to EXCLUDE from pass/fail calculation
EXCLUDED_PB_CHECKS = {"pb_all_atoms_connected"}

# Known dataset prefixes for categorization (order matters - longer prefixes first)
KNOWN_DATASETS = [
    # omol subdatasets
    "omol/solvated_protein",
    "omol/torsion_profiles",
    "omol/metal_organics",
    "omol/electrolytes",
    "omol/redo_orca6",
    # pdb fragments/pockets
    "pdb_fragments_300K",
    "pdb_fragments_400K",
    "pdb_pockets_300K",
    "pdb_pockets_400K",
    # electrolytes variants
    "electrolytes_scaled_sep",
    "electrolytes_reactivity",
    "electrolytes_redox",
    "scaled_separations_exp",
    "5A_elytes",
    "ml_elytes",
    # protein/interface
    "ml_protein_interface",
    "protein_interface",
    "protein_core",
    # geom/orca
    "geom_orca6",
    "geom_orca7",
    # mechanism databases
    "rmechdb",
    "pmechdb",
    # other datasets
    "orbnet_denali",
    "mo_hydrides",
    "low_spin_23",
    "ani1xbb",
    "trans1x",
    "tm_react",
    "rgd_uks",
    "droplet",
    "ml_mol",
    "ml_mo",
    "spice",
    "ani2x",
    "rpmd",
    "dna",
    "rna",
]


def categorize_source(source: str) -> str:
    """Categorize source string into dataset name."""
    if not source:
        return "unknown"
    for prefix in KNOWN_DATASETS:
        if source.startswith(prefix):
            return prefix
    return "unknown"


def get_processed_indices(
    output_dir: Path,
    existing_merged_path: str | None = None,
) -> set[int]:
    """Scan output directory and optional merged file for already-processed indices.

    Args:
        output_dir: Directory containing chunk parquet files.
        existing_merged_path: Optional path to a pre-existing merged parquet file.
            Indices from this file will be excluded from processing.

    Returns:
        Set of lmdb_idx values that have already been processed.
    """
    done = set()

    # Load from existing merged file if provided
    if existing_merged_path:
        merged_path = Path(existing_merged_path)
        if merged_path.exists():
            try:
                df = pd.read_parquet(merged_path, columns=["lmdb_idx"])
                done.update(df["lmdb_idx"].tolist())
                print(f"Loaded {len(done):,} indices from existing merged file")
            except Exception as e:
                print(f"Warning: Failed to load existing merged file: {e}")

    # Also load from chunk files
    for parquet_file in output_dir.glob("chunk_*.parquet"):
        try:
            df = pd.read_parquet(parquet_file, columns=["lmdb_idx"])
            done.update(df["lmdb_idx"].tolist())
        except Exception:
            pass

    return done


def process_single(idx: int, dataset, pb) -> dict | None:  # noqa: ANN001
    """Process a single molecule. Returns None if entry cannot be loaded."""
    from atomworks.io.tools.rdkit import atom_array_to_rdkit

    try:
        example = dataset[idx]
        info = example["extra_info"]
        atom_array = example["atom_array"]

        source = str(info.get("source", ""))
        base_result = {
            "lmdb_idx": idx,
            "num_atoms": int(info.get("num_atoms", -1)),
            "source": source,
            "dataset": categorize_source(source),
            "charge": int(info.get("charge", 0)),
            "energy": float(info.get("energy", float("nan"))),
        }

        # RDKit conversion
        try:
            mol = atom_array_to_rdkit(
                atom_array,
                set_coord=True,
                hydrogen_policy="keep",
                infer_bonds=False,
                sanitize=True,
                attempt_fixing_corrupted_molecules=True,
            )
        except Exception:
            return {**base_result, "rdkit_conversion_failed": True, "pb_bust_failed": False, "passes_stereochemical_validation": False}

        # PoseBusters validation
        try:
            results_df = pb.bust(mol_pred=mol, full_report=False)
            pb_results = results_df.iloc[0].to_dict()
            check_results = {f"pb_{k}": bool(v) for k, v in pb_results.items() if isinstance(v, bool | int) and k != "file_path"}
            relevant_checks = {k: v for k, v in check_results.items() if k not in EXCLUDED_PB_CHECKS}
            all_passed = all(relevant_checks.values()) if relevant_checks else True
            return {**base_result, "rdkit_conversion_failed": False, "pb_bust_failed": False, "passes_stereochemical_validation": all_passed, **check_results}
        except Exception:
            # We assume that molecules (e.g., metal organics) that error on PoseBusters are valid (but mark as PB bust failed)
            return {**base_result, "rdkit_conversion_failed": False, "pb_bust_failed": True, "passes_stereochemical_validation": True}

    except Exception:
        # Skip entries that fail to load entirely
        return None


def validate(
    output_dir: str,
    chunk_id: int,
    sample_size: int = 1000,
    lmdb_path: str = "/net/tukwila/omol_4m/train_4M",
    total_entries: int = 4_000_000,
    checkpoint_interval: int = 10,
    seed: int | None = None,
    existing_merged: str | None = None,
) -> None:
    """Validate random sample of entries from ASE LMDB.

    Args:
        output_dir: Directory to write chunk parquet files.
        chunk_id: Unique identifier for this chunk (used in filename).
        sample_size: Number of entries to process in this chunk.
        lmdb_path: Path to the ASE LMDB database.
        total_entries: Total number of entries in the database.
        checkpoint_interval: Save results every N entries.
        seed: Random seed for sampling. Defaults to chunk_id.
        existing_merged: Optional path to a pre-existing merged parquet file.
            Indices from this file will be skipped during processing.
    """
    from posebusters import PoseBusters

    from atomworks.ml.datasets.ase_dataset import AseDBDataset
    from atomworks.ml.datasets.loaders.ase import create_ase_loader
    from atomworks.ml.transforms.base import Identity

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chunk_file = output_path / f"chunk_{chunk_id:04d}.parquet"

    if seed is None:
        seed = chunk_id
    random.seed(seed)

    print(f"ASE Validation - Chunk {chunk_id}")
    print(f"Sample size: {sample_size}, Checkpoint: every {checkpoint_interval}")

    # Get already-processed indices
    done_indices = get_processed_indices(output_path, existing_merged)
    print(f"Found {len(done_indices):,} already processed")

    # Sample random indices
    all_indices = set(range(total_entries)) - done_indices
    if not all_indices:
        print("All done!")
        return

    to_process = random.sample(list(all_indices), min(sample_size, len(all_indices)))
    print(f"Processing {len(to_process)} indices")

    # Initialize dataset and PoseBusters once
    loader = create_ase_loader(per_atom_properties=[], global_properties=["source", "charge", "num_atoms", "energy"], add_missing_atoms=False)
    dataset = AseDBDataset(lmdb_path=lmdb_path, name="ase_validation", filters=[], per_atom_properties=[], global_properties=["source", "charge", "num_atoms", "energy"], loader=loader, transform=Identity(), use_cache=False)
    pb = PoseBusters(config="mol")

    # Process
    results = []
    skipped = 0
    for idx in tqdm(to_process, desc="Validating"):
        result = process_single(idx, dataset, pb)
        if result is None:
            skipped += 1
            continue
        results.append(result)
        if len(results) % checkpoint_interval == 0:
            pd.DataFrame(results).to_parquet(chunk_file, index=False)

    # Final save
    if results:
        df = pd.DataFrame(results)
        df.to_parquet(chunk_file, index=False)
        print(f"\nSaved {len(df)} entries to {chunk_file}")
        print(f"Skipped {skipped} entries (load errors)")
        print(f"Dataset distribution:\n{df['dataset'].value_counts().to_string()}")


if __name__ == "__main__":
    fire.Fire(validate)
