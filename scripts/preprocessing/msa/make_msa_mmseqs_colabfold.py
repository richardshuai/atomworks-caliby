"""
Python script to generate multiple sequence alignments (MSAs) using MMseqs2 (with GPU support)
as implemented in the ColabFold repository. Much of this code is based on the
ColabFold mmseqs search script: https://github.com/sokrypton/ColabFold/blob/main/colabfold/mmseqs/search.py

This script has two main modes:
    - run_pipeline: Generate MSAs from a DataHub parquet file (e.g., training or validation dataset)
    - generate_msas: Generate MSAs from a list of sequences (e.g., for inference)

Some modifications to the original script were made to fit datahub pipeline conventions.

Example usage:
    # Full pipeline with DataHub parquet file (e.g., training or validation dataset)
    python make_msa_mmseqs_colabfold.py \
        --input_file /net/scratch/mkazman/datahub/example_datahub_files/example_datahub_file.parquet \
        --output_dir /net/scratch/mkazman/msa/example_msas/ \
        --gpu \
        --gpu_server

    # Simple interface with direct sequences
    python make_msa_mmseqs_colabfold.py generate_msas \
        --sequences "MSYIWRQLGSPTVAITLSVSTVIYVTVICPIVFIHLFGDHL..." \
        --output_dir /path/to/output/

    # Multiple sequences
    python make_msa_mmseqs_colabfold.py generate_msas \
        --sequences '["MSYIWRQLGSPTVAITLSVSTVIYVTVICPIVFIHLFGDHL...", "MKKKEVEKDDLIENASRVASCISIFLIIASTTMYIFIGLKI..."]' \
        --output_dir /path/to/output/
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
from os import PathLike
from pathlib import Path

import fire
import pandas as pd

from atomworks.io.enums import ChainType
from atomworks.ml.utils.misc import hash_sequence

logger = logging.getLogger(__name__)

# Convention from atomworks.ml-output pn_units dataframes
SEQUENCE_COLUMN_NAME = "q_pn_unit_processed_entity_non_canonical_sequence"
SEQUENCE_TYPE_COLUMN_NAME = "q_pn_unit_type"

MMSEQS_CMD = "/software/mmseqs-gpu/bin/mmseqs"

# NOTE: MMseqs2 databases are stored on the local drive of DB nodes. You can access them from other nodes with /net/databases/colabfold, but this can run into IO-related issues
LOCAL_CPU_DB_PATH = "/local/colabfold/"
LOCAL_GPU_DB_PATH = LOCAL_CPU_DB_PATH + "gpu/"
NET_DB_PATH = "/net/databases/colabfold/"

COLABFOLD_DB_NAME = "colabfold_envdb_202108_db"
UNIREF30_DB_NAME = "uniref30_2302_db"
PDB100_DB_NAME = "pdb100_230517_db"  # TODO this is not fully tested yet
PDB70_DB_NAME = "pdb70_220313"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

MODULE_OUTPUT_POS = {
    "align": 4,
    "convertalis": 4,
    "expandaln": 5,
    "filterresult": 4,
    "lndb": 2,
    "mergedbs": 2,
    "mvdb": 2,
    "pairaln": 4,
    "result2msa": 4,
    "search": 3,
}


def _get_database_path(gpu: bool = False) -> Path:
    """
    Determine which database path to use, falling back from local to network paths.

    Args:
        gpu: Whether to use GPU databases

    Returns:
        Path to the database directory
    """
    if gpu:
        # For GPU, try local GPU path first, then network path
        if Path(LOCAL_GPU_DB_PATH).exists():
            logger.info(f"Using local GPU database path: {LOCAL_GPU_DB_PATH}")
            return Path(LOCAL_GPU_DB_PATH)
        else:
            logger.info(f"Local GPU database path not found, using network path: {NET_DB_PATH}")
            return Path(NET_DB_PATH)
    else:
        # For CPU, try local CPU path first, then network path
        if Path(LOCAL_CPU_DB_PATH).exists():
            logger.info(f"Using local CPU database path: {LOCAL_CPU_DB_PATH}")
            return Path(LOCAL_CPU_DB_PATH)
        else:
            logger.info(f"Local CPU database path not found, using network path: {NET_DB_PATH}")
            return Path(NET_DB_PATH)


def _make_fasta_file_from_sequence_strings(sequence_strings: list[str], output_fasta_file: PathLike) -> None:
    """
    Create a FASTA file from a list of sequence strings. Header is the hash of the sequence string.

    Args:
        sequence_strings: List of sequence strings
        output_fasta_file: Path to the output FASTA file
    Returns:
        None
    """
    with open(output_fasta_file, "w") as f:
        for sequence_string in sequence_strings:
            f.write(f">{hash_sequence(sequence_string)}\n{sequence_string}\n")


def _make_mmseqs_db_from_fasta(fasta_file: PathLike, output_dir: PathLike):
    """
    Create a MMseqs2 database from a FASTA file.

    Args:
        fasta_file: Path to the FASTA file
        output_dir: Path to the output directory
    Returns:
        Path to the output database
    """
    output_db = Path(output_dir) / "qdb"
    subprocess.check_call([MMSEQS_CMD, "createdb", fasta_file, output_db])
    return output_db


def _parse_input_sequence_database(
    input_file: PathLike,
    sequence_col: str = SEQUENCE_COLUMN_NAME,
    chain_type_col: str = SEQUENCE_TYPE_COLUMN_NAME,
) -> list[str]:
    """
    Parse the input DataHub parquet file to get the sequence strings. Also removes duplicate sequences.

    Args:
        input_file: Path to the input DataHub parquet or csv file
        sequence_col: Name of the column containing the sequence strings
    Returns:
        List of sequence strings
    """
    # handle parquet or csv
    if str(input_file).endswith(".parquet"):
        df = pd.read_parquet(input_file)
    elif str(input_file).endswith(".csv"):
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file type: {input_file}")

    # filter for sequence type
    if chain_type_col in df.columns:
        df = df[df[chain_type_col] == ChainType.POLYPEPTIDE_L]
    else:
        logger.warning(f"No sequence type column found in {input_file}. Will assume everything is L-polypeptide.")

    sequences = df[sequence_col].tolist()
    start_len = len(sequences)
    sequences = list(set(sequences))
    logger.info(f"Removed {start_len - len(sequences)} duplicate sequences from {input_file}")

    return sequences


def _start_gpu_server(
    mmseqs: Path, db_name: Path, max_seqs: int, db_load_mode: int, prefilter_mode: int, wait_time: int = 20
):
    """
    Starts the GPU server. Waits for it to start and then returns the process object. We can't do something like
    subprocess.check_call because the server will run indefinitely and we can't wait for it to finish. Instead, we
    have to check the stderr for the server starting up.
    """

    cmd = [
        mmseqs,
        "gpuserver",
        db_name,
        "--max-seqs",
        max_seqs,
        "--db-load-mode",
        db_load_mode,
        "--prefilter-mode",
        prefilter_mode,
    ]
    gpu_server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)

    time.sleep(
        wait_time
    )  # TODO They recently updated MMseqs2 to automatically wait for the server to start. Once they release a new version and we update our local installation, this can be removed

    return gpu_server_process


def _run_mmseqs(mmseqs: Path, params: list[str | Path]) -> None:
    """
    Run an MMseqs2 command.

    Args:
        mmseqs: Path to the MMseqs2 executable
        params: List of parameters to pass to the MMseqs2 command
    Returns:
        None
    """
    module = params[0]
    if module in MODULE_OUTPUT_POS:
        output_pos = MODULE_OUTPUT_POS[module]
        output_path = Path(params[output_pos]).with_suffix(".dbtype")
        if output_path.exists():
            logger.info(f"Skipping {module} because {output_path} already exists")
            return

    params_log = " ".join(str(i) for i in params)
    logger.info(f"Running {mmseqs} {params_log}")
    # hide MMseqs2 verbose paramters list that clogs up the log
    os.environ["MMSEQS_CALL_DEPTH"] = "1"
    subprocess.check_call([mmseqs, *params])


def _run_mmseqs_search_and_filter(
    mmseqs: str,
    base: str,
    dbbase: str,
    db_name: str,
    db_suffix1: str,
    db_suffix2: str,
    output_name: str,
    db_load_mode: int,
    threads: int,
    search_param: list[str],
    expand_param: list[str],
    filter_param: list[str],
    align_eval: float,
    max_accept: int,
    qsc: float,
    align_alt_ali: int = 10,
    filter_qid: bool = False,
    filter_diff: int = 0,
    filter_max_seq_id: float = 1.0,
    filter_min_enable: int = 1000,
    profile_input: str = "qdb",
    tmp_dir: str = "tmp",
) -> None:
    """
    Helper function to run MMseqs2 search, expand alignment and filter results. This is the basic pipeline used in ColabFold to generate
    high quality MSAs using MMseqs2.

    First we search against the target database and create alignments of the results. Then we expand these alignments using an alignment
    of the target database. We can then realign the expanded alignments which ultimately results in a higher quality alignments. The
    resulting alignments are then filtered and converted to an MSA format.

    See the "Faster MSA generation with MMSeqs2" section in the ColabFold paper for more details: https://www.nature.com/articles/s41592-022-01488-1

    Args:
        mmseqs: Path to the MMseqs2 executable
        base: Directory for the results (and intermediate files)
        dbbase: Path to the database and indices you downloaded and created with setup_databases.sh
        db_name: Name of the database to search against
        db_suffix1: Suffix for the database to search against
        db_suffix2: Suffix for the database to search against
        output_name: Name of the output file
        db_load_mode: Database preload mode 0: auto, 1: fread, 2: mmap, 3: mmap+touch
        threads: Number of threads to use
        search_param: Extra parameters for the search
        expand_param: Extra parameters for the expandaln
        filter_param: Extra parameters for the filterresult
        align_eval: e-val threshold for align
        align_alt_ali: Number of alternative alignments to keep
        max_accept: Maximum accepted alignments before alignment calculation for a query is stopped
        qsc: filterresult - reduce diversity of output MSAs using min score thresh
        filter_qid: filterresult - Reduce diversity of output MSAs using min.seq. idendity with query sequences
        filter_diff: filterresult - Keep at least this many seqs in each MSA block
        filter_max_seq_id: filterresult - Maximum sequence identity for filtering
        filter_min_enable: filterresult - Minimum number of sequences to keep in each MSA block
        profile_input: Profile input (usually qdb)
        tmp_dir: Temporary directory

    Returns:
        None
    """

    if "--gpu-server" in search_param:
        logger.info("Setting up GPU server...")
        gpu_server_process = _start_gpu_server(mmseqs, dbbase.joinpath(db_name), search_param[8], search_param[3], "1")
        logger.info("GPU server setup complete")

    _run_mmseqs(
        mmseqs,
        [
            "search",
            base.joinpath(profile_input),
            dbbase.joinpath(db_name),
            base.joinpath("res"),
            base.joinpath(tmp_dir),
            "--threads",
            str(threads),
            *search_param,
        ],
    )

    if "--gpu-server" in search_param:
        logger.info("Stopping GPU server...")
        gpu_server_process.terminate()  # Send SIGTERM
        gpu_server_process.wait()
        logger.info("GPU server stopped")

    if profile_input == "qdb":
        # Move and symlink databases (only needed for first uniref search)
        _run_mmseqs(mmseqs, ["mvdb", base.joinpath(f"{tmp_dir}/latest/profile_1"), base.joinpath("prof_res")])
        _run_mmseqs(mmseqs, ["lndb", base.joinpath("qdb_h"), base.joinpath("prof_res_h")])
        align_profile = "prof_res"
    else:
        align_profile = f"{tmp_dir}/latest/profile_1"

    # Expand the alignment from search against an alignment of the target database to improve alignment quality
    _run_mmseqs(
        mmseqs,
        [
            "expandaln",
            base.joinpath(profile_input),
            dbbase.joinpath(f"{db_name}{db_suffix1}"),
            base.joinpath("res"),
            dbbase.joinpath(f"{db_name}{db_suffix2}"),
            base.joinpath("res_exp"),
            "--db-load-mode",
            str(db_load_mode),
            "--threads",
            str(threads),
            *expand_param,
        ],
    )

    # Realign using the expanded alignment to improve alignment quality
    _run_mmseqs(
        mmseqs,
        [
            "align",
            base.joinpath(align_profile),
            dbbase.joinpath(f"{db_name}{db_suffix1}"),
            base.joinpath("res_exp"),
            base.joinpath("res_exp_realign"),
            "--db-load-mode",
            str(db_load_mode),
            "-e",
            str(align_eval),
            "--max-accept",
            str(max_accept),
            "--threads",
            str(threads),
            "--alt-ali",
            str(align_alt_ali),
            "-a",
        ],
    )

    # Filter the alignment to remove low quality alignments
    _run_mmseqs(
        mmseqs,
        [
            "filterresult",
            base.joinpath("qdb"),
            dbbase.joinpath(f"{db_name}{db_suffix1}"),
            base.joinpath("res_exp_realign"),
            base.joinpath("res_exp_realign_filter"),
            "--db-load-mode",
            str(db_load_mode),
            "--qid",
            str(int(filter_qid)),
            "--qsc",
            str(qsc),
            "--diff",
            str(filter_diff),
            "--threads",
            str(threads),
            "--max-seq-id",
            str(filter_max_seq_id),
            "--filter-min-enable",
            str(filter_min_enable),
        ],
    )

    # Convert the filtered alignment to a multiple sequence alignment
    _run_mmseqs(
        mmseqs,
        [
            "result2msa",
            base.joinpath("qdb"),
            dbbase.joinpath(f"{db_name}{db_suffix1}"),
            base.joinpath("res_exp_realign_filter"),
            base.joinpath(output_name),
            "--msa-format-mode",
            "6",
            "--db-load-mode",
            str(db_load_mode),
            "--threads",
            str(threads),
            *filter_param,
        ],
    )

    # Cleanup intermediate files
    _run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp_realign_filter")])
    _run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp_realign")])
    _run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_exp")])
    _run_mmseqs(mmseqs, ["rmdb", base.joinpath("res")])


def _mmseqs_search_monomer(
    dbbase: Path,
    base: Path,
    uniref_db: Path,
    template_db: Path,
    metagenomic_db: Path,
    mmseqs: Path = Path(MMSEQS_CMD),
    use_env: bool = True,
    use_templates: bool = False,
    filter: bool = True,
    search_eval: float = 0.1,
    expand_eval: float = 1e-3,
    expand_max_seq_id: float = 0.95,
    align_eval: int = 10,
    diff: int = 3000,
    qsc: float = -20.0,
    filter_qsc: float = 0.0,
    filter_max_seq_id: float = 0.95,
    filter_min_enable: int = 1000,
    filter_qid: str = "0.0,0.2,0.4,0.6,0.8,1.0",
    max_accept: int = 10000,  # this was 1000000 in the original ColabFold script
    num_iterations: int = 3,
    max_seqs: int = 10000,
    prefilter_mode: int = 0,
    s: float = 8,  # Set to None to use k-score instead
    db_load_mode: int = 2,
    threads: int = 32,
    gpu: int = 0,
    gpu_server: int = 0,
    unpack: bool = True,
    template_sensitivity: float = 7.5,
    template_eval: float = 0.1,
) -> None:
    """
    Run mmseqs with a local colabfold database set. Adapted from ColabFold mmseqs search script:
    https://github.com/sokrypton/ColabFold/blob/main/colabfold/mmseqs/search.py

    Runs search (see _run_mmseqs_search_and_filter) on each database (uniref, metagenomic, templates) in turn.
    Alignments from these searches are then merged and converted to an MSA format. If unpack is True, the results
    will be formatted in individual .a3m files with names corresponding to the input sequence index (0.a3m, 1.a3m, etc.)
    If unpack is False, the results will be formatted in a single .a3m file with the name "final.a3m". If use_templates is True,
    there will be additional .m8 files with the same name as the .a3m files but with the suffix ".m8" showing the
    search results from the templates database.

    NOTE: Unless specified otherwise, all parameters' defualt values are from the ColabFold script.

    Args:
        dbbase: Path to the database and indices you downloaded and created with setup_databases.sh
        base: Directory for the results (and intermediate files)
        uniref_db: UniRef database
        template_db: Templates database (usually PDB70)
        metagenomic_db: Environmental database (usually ColabFold metagenomics database)
        use_env: Whether to use the environmental database
        use_templates: Whether to use the templates database
        filter: Whether to filter the MSA
        search_eval: Search e-value threshold
        expand_eval: e-val threshold for 'expandaln'
        expand_max_seq_id: Maximum sequence identity for 'expandaln'
        align_eval: e-val threshold for 'align'
        diff: filterresult - Keep at least this many seqs in each MSA block
        qsc: filterresult - reduce diversity of output MSAs using min score thresh
        filter_qsc: filterresult - reduce diversity of output MSAs using min score thresh
        filter_max_seq_id: filterresult - Maximum sequence identity for filtering
        filter_min_enable: filterresult - Minimum number of sequences to keep in each MSA block
        filter_qid: filterresult - Reduce diversity of output MSAs using min.seq. idendity with query sequences
        max_accept: align - Maximum accepted alignments before alignment calculation for a query is stopped
        num_iterations: Number of iterations for the search
        max_seqs: Maximum number of sequences to search
        prefilter_mode: Prefiltering algorithm to use: 0: k-mer (high-mem), 1: ungapped (high-cpu), 2: exhaustive (no prefilter, very slow)
        s: MMseqs2 sensitivity. Lowering this will result in a much faster search but possibly sparser MSAs. By default, the k-mer threshold is directly set to the same one of the server, which corresponds to a sensitivity of ~8
        db_load_mode: Database preload mode 0: auto, 1: fread, 2: mmap, 3: mmap+touch
        threads: Number of threads to use
        gpu: Whether to use GPU (1) or not (0)
        gpu_server: Whether to use GPU server (1) or not (0)
        unpack: Whether to unpack the results to loose files or keep MMseqs2 databases
        template_sensitivity: Templates search sensitivity
        template_eval: Templates search e-value threshold
    Returns:
        None
    """
    if filter:
        align_eval = 1e-3
        qsc = 0.8
        max_accept = 10000

    # check db types and make sure they exist
    used_dbs = [uniref_db]
    if use_templates:
        used_dbs.append(template_db)
    if use_env:
        used_dbs.append(metagenomic_db)
    for db in used_dbs:
        if not dbbase.joinpath(f"{db}.dbtype").is_file():
            raise FileNotFoundError(f"Database {dbbase.joinpath(db)} does not exist")
        if (
            not dbbase.joinpath(f"{db}.idx").is_file() and not dbbase.joinpath(f"{db}.idx.index").is_file()
        ) or os.environ.get("MMSEQS_IGNORE_INDEX", False):
            logger.info("Search does not use index")
            db_load_mode = 0
            dbSuffix1 = "_seq"
            dbSuffix2 = "_aln"
            dbSuffix3 = ""
        else:
            dbSuffix1 = ".idx"
            dbSuffix2 = ".idx"
            dbSuffix3 = ".idx"

    # prep additional params for search, filter, and expand
    search_param = [
        "--num-iterations",
        str(num_iterations),
        "--db-load-mode",
        str(db_load_mode),
        "-a",
        "-e",
        str(search_eval),
        "--max-seqs",
        str(max_seqs),
    ]
    if gpu:
        search_param += [
            "--gpu",
            str(gpu),
            "--prefilter-mode",
            "1",
        ]  # gpu version only supports ungapped prefilter currently
    else:
        search_param += ["--prefilter-mode", str(prefilter_mode)]
        if s is not None:  # sensitivy can only be set for non-gpu version, gpu version runs at max sensitivity
            search_param += ["-s", f"{s:.1f}"]
        else:
            search_param += ["--k-score", "'seq:96,prof:80'"]
    if gpu_server:
        search_param += ["--gpu-server", str(gpu_server)]

    filter_param = [
        "--filter-msa",
        str(int(filter)),
        "--filter-min-enable",
        str(filter_min_enable),
        "--diff",
        str(diff),
        "--qid",
        str(filter_qid),
        "--qsc",
        str(filter_qsc),
        "--max-seq-id",
        str(filter_max_seq_id),
    ]
    expand_param = [
        "--expansion-mode",
        "0",
        "-e",
        str(expand_eval),
        "--expand-filter-clusters",
        str(int(filter)),
        "--max-seq-id",
        str(expand_max_seq_id),
    ]

    # search and filter uniref
    if not base.joinpath("uniref.a3m").with_suffix(".a3m.dbtype").exists():
        _run_mmseqs_search_and_filter(
            mmseqs,
            base,
            dbbase,
            uniref_db,
            dbSuffix1,
            dbSuffix2,
            "uniref.a3m",
            db_load_mode,
            threads,
            search_param,
            expand_param,
            filter_param,
            align_eval,
            max_accept,
            qsc,
        )
    else:
        logger.info(f"Skipping {uniref_db} search because uniref.a3m already exists")

    # search and filter metagenomic
    if use_env and not base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m").with_suffix(".a3m.dbtype").exists():
        _run_mmseqs_search_and_filter(
            mmseqs,
            base,
            dbbase,
            metagenomic_db,
            dbSuffix1,
            dbSuffix2,
            "bfd.mgnify30.metaeuk30.smag30.a3m",
            db_load_mode,
            threads,
            search_param,
            expand_param,
            filter_param,
            align_eval,
            max_accept,
            qsc,
            profile_input="prof_res",
            tmp_dir="tmp3",
        )
    elif use_env:
        logger.info(f"Skipping {metagenomic_db} search because bfd.mgnify30.metaeuk30.smag30.a3m already exists")

    # search and filter templates
    if use_templates and not base.joinpath(f"{template_db}.m8").with_suffix(".m8.dbtype").exists():
        _run_mmseqs(
            mmseqs,
            [
                "search",
                base.joinpath("prof_res"),
                dbbase.joinpath(template_db),
                base.joinpath("res_pdb"),
                base.joinpath("tmp2"),
                "--db-load-mode",
                str(db_load_mode),
                "--threads",
                str(threads),
                "-s",
                str(template_sensitivity),
                "-a",
                "-e",
                str(template_eval),
                "--prefilter-mode",
                str(prefilter_mode),
            ],
        )
        _run_mmseqs(
            mmseqs,
            [
                "convertalis",
                base.joinpath("prof_res"),
                dbbase.joinpath(f"{template_db}{dbSuffix3}"),
                base.joinpath("res_pdb"),
                base.joinpath(f"{template_db}"),
                "--format-output",
                "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,cigar",
                "--db-output",
                "1",
                "--db-load-mode",
                str(db_load_mode),
                "--threads",
                str(threads),
            ],
        )
        _run_mmseqs(mmseqs, ["rmdb", base.joinpath("res_pdb")])
    elif use_templates:
        logger.info(f"Skipping {template_db} search because {template_db}.m8 already exists")

    # merge alignments
    if use_env:
        _run_mmseqs(
            mmseqs,
            [
                "mergedbs",
                base.joinpath("qdb"),
                base.joinpath("final.a3m"),
                base.joinpath("uniref.a3m"),
                base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m"),
            ],
        )
        _run_mmseqs(mmseqs, ["rmdb", base.joinpath("bfd.mgnify30.metaeuk30.smag30.a3m")])
        _run_mmseqs(mmseqs, ["rmdb", base.joinpath("uniref.a3m")])
    else:
        _run_mmseqs(mmseqs, ["mvdb", base.joinpath("uniref.a3m"), base.joinpath("final.a3m")])
        _run_mmseqs(mmseqs, ["rmdb", base.joinpath("uniref.a3m")])

    # unpack alignments into individual .a3m (or .m8) files
    if unpack:
        _run_mmseqs(
            mmseqs,
            [
                "unpackdb",
                base.joinpath("final.a3m"),
                base.joinpath("."),
                "--unpack-name-mode",
                "0",
                "--unpack-suffix",
                ".a3m",
            ],
        )
        _run_mmseqs(mmseqs, ["rmdb", base.joinpath("final.a3m")])

        if use_templates:
            _run_mmseqs(
                mmseqs,
                [
                    "unpackdb",
                    base.joinpath(f"{template_db}"),
                    base.joinpath("."),
                    "--unpack-name-mode",
                    "0",
                    "--unpack-suffix",
                    ".m8",
                ],
            )
            if base.joinpath(f"{template_db}").exists():
                _run_mmseqs(mmseqs, ["rmdb", base.joinpath(f"{template_db}")])

    # cleanup
    _run_mmseqs(mmseqs, ["rmdb", base.joinpath("prof_res")])
    _run_mmseqs(mmseqs, ["rmdb", base.joinpath("prof_res_h")])
    shutil.rmtree(base.joinpath("tmp"))
    if use_templates:
        shutil.rmtree(base.joinpath("tmp2"))
    if use_env:
        shutil.rmtree(base.joinpath("tmp3"))


def run_mmseqs_pipeline(
    input_file: PathLike,
    output_dir: PathLike,
    use_templates: bool = False,
    gpu: bool = True,
    gpu_server: bool = False,
    sequence_col: str = SEQUENCE_COLUMN_NAME,
    chain_type_col: str = SEQUENCE_TYPE_COLUMN_NAME,
    num_iterations: int = 3,
    max_seqs: int = 10000,
    use_local_temp_dir: bool = False,
) -> None:
    """
    Run the MMseqs2 pipeline as done in ColabFold. See helper functions for more details.
    Takes input from a DataHub parquet or csv file and creates MSAs in the .a3m format in the output directory.

    Note: For the regular pipeline runs we don't expose many of the advanced variables

    Args:
        input_file: Path to the input DataHub parquet or csv file
        output_dir: Path to the output directory
        use_templates: Whether to use the templates database
        gpu: Whether to use GPU acceleration
        gpu_server: Whether to use GPU server
        num_iterations: Number of iterations for the search
        max_seqs: Maximum number of sequences to search
        use_local_temp_dir: Whether to use a local temporary directory
    Returns:
        None
    """

    # use tempfile to make temporary directory that should automatically be on local drive
    if use_local_temp_dir:
        intermediate_dir = tempfile.mkdtemp()
    else:
        intermediate_dir = output_dir

    # parse all the sequences from input file
    sequences = _parse_input_sequence_database(input_file, sequence_col, chain_type_col)
    fasta_file = Path(intermediate_dir) / "input_sequences.fasta"
    _make_fasta_file_from_sequence_strings(sequences, fasta_file)
    _make_mmseqs_db_from_fasta(fasta_file, intermediate_dir)

    if gpu_server and not gpu:
        raise ValueError("gpu_server is True but gpu is False")

    start_time = time.time()
    _mmseqs_search_monomer(
        dbbase=_get_database_path(gpu=gpu),
        base=Path(intermediate_dir),
        uniref_db=Path(UNIREF30_DB_NAME),
        template_db=Path(PDB100_DB_NAME),
        metagenomic_db=Path(COLABFOLD_DB_NAME),
        gpu=int(gpu),
        gpu_server=int(gpu_server),
        use_templates=use_templates,
        num_iterations=num_iterations,
        max_seqs=max_seqs,
    )
    logger.info(f"Completed {len(sequences)} sequences in {time.time() - start_time} with MMSeqs2 search and alignment")

    # cleanup by removing any file or directory that isn't .a3m or .m8
    for file in Path(intermediate_dir).iterdir():
        if not file.name.endswith((".a3m", ".m8")):
            file.unlink()

    if use_local_temp_dir:
        # copy over everything from intermediate_dir to output_dir
        for file in Path(intermediate_dir).iterdir():
            shutil.copy(file, Path(output_dir) / file.name)
        # remove the intermediate_dir
        shutil.rmtree(intermediate_dir)


def generate_msas_from_sequences(
    sequences: str | list[str],
    output_dir: PathLike,
    use_templates: bool = False,
    gpu: bool = False,
    gpu_server: bool = False,
    num_iterations: int = 3,
    max_seqs: int = 10000,
    use_local_temp_dir: bool = True,
) -> None:
    """
    Simple entrypoint to generate MSAs directly from protein sequences.

    Args:
        sequences: A single protein sequence string or list of protein sequences
        output_dir: Path to the output directory where MSA files will be saved
        use_templates: Whether to use template databases for structure prediction
        gpu: Whether to use GPU acceleration
        gpu_server: Whether to use GPU server (requires gpu=True)
        num_iterations: Number of search iterations (default: 3)
        max_seqs: Maximum number of sequences in MSA (default: 10000)
        use_local_temp_dir: Whether to use local temporary directory for intermediate files

    Returns:
        None; MSA files are saved to the output directory

    Example:
        # Single sequence
        generate_msas_from_sequences(
            "MSYIWRQLGSPTVAITLSVSTVIYVTVICPIVFIHLFGDHL...",
            "output_msas/"
        )

        # Multiple sequences
        generate_msas_from_sequences([
            "MSYIWRQLGSPTVAITLSVSTVIYVTVICPIVFIHLFGDHL...",
            "MKKKEVEKDDLIENASRVASCISIFLIIASTTMYIFIGLKI..."
        ], "output_msas/")
    """
    # Ensure sequences is a list for unified processing
    if isinstance(sequences, str):
        sequences = [sequences]

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Use tempfile to make temporary directory that should automatically be on local drive
    if use_local_temp_dir:
        intermediate_dir = tempfile.mkdtemp()
    else:
        intermediate_dir = output_dir

    # Create FASTA file from sequences
    fasta_file = Path(intermediate_dir) / "input_sequences.fasta"
    _make_fasta_file_from_sequence_strings(sequences, fasta_file)
    _make_mmseqs_db_from_fasta(fasta_file, intermediate_dir)

    if gpu_server and not gpu:
        raise ValueError("gpu_server is True but gpu is False")

    start_time = time.time()
    _mmseqs_search_monomer(
        dbbase=_get_database_path(gpu=gpu),
        base=Path(intermediate_dir),
        uniref_db=Path(UNIREF30_DB_NAME),
        template_db=Path(PDB100_DB_NAME),
        metagenomic_db=Path(COLABFOLD_DB_NAME),
        gpu=int(gpu),
        gpu_server=int(gpu_server),
        use_templates=use_templates,
        num_iterations=num_iterations,
        max_seqs=max_seqs,
    )
    logger.info(
        f"Completed {len(sequences)} sequences in {time.time() - start_time} seconds with MMSeqs2 search and alignment"
    )

    # cleanup by removing any file or directory that isn't .a3m or .m8
    for file in Path(intermediate_dir).iterdir():
        if not file.name.endswith((".a3m", ".m8")):
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)

    if use_local_temp_dir:
        # copy over everything from intermediate_dir to output_dir
        for file in Path(intermediate_dir).iterdir():
            shutil.copy(file, Path(output_dir) / file.name)
        # remove the intermediate_dir
        shutil.rmtree(intermediate_dir)

    logger.info(f"MSA files saved to: {Path(output_dir).absolute()}")


if __name__ == "__main__":
    fire.Fire(
        {
            "run_pipeline": run_mmseqs_pipeline,
            "generate_msas": generate_msas_from_sequences,
        }
    )
