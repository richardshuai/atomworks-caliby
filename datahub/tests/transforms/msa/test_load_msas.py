import copy
import logging
import os
import time
from multiprocessing import Pool
from typing import Any

import numpy as np
import pytest
from cifutils.enums import ChainType
from tqdm import tqdm

from datahub.datasets.parsers import PNUnitsDFParser, load_example_from_metadata_row
from datahub.transforms.base import Compose
from datahub.transforms.filters import RemoveHydrogens, RemoveUnsupportedChainTypes
from datahub.transforms.msa._msa_constants import (
    AMINO_ACID_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE,
    RNA_NUCLEOTIDE_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE,
)
from datahub.transforms.msa.msa import LoadPolymerMSAs
from datahub.utils.io import get_sharded_file_path
from datahub.utils.misc import hash_sequence
from tests.conftest import PN_UNITS_DF, PROTEIN_MSA_DIRS, RNA_MSA_DIRS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MSA_TEST_CASES = [
    {
        # Protein
        "pdb_id": "5gam",
        "chain_id": "C",
        "sequence": "MEGDDLFDEFGNLIGVDPFDSDEEESVLDEQEQYQTNTFEGSGNNNEIESRQLTSLGSKKELGISLEHPYGKEVEVLMETKNTQSPQTPLVEPVTERTKLQEHTIFTQLKKNIPKTRYNRDYMLSMANIPERIINVGVIGPLHSGKTSLMDLLVIDSHKRIPDMSKNVELGWKPLRYLDNLKQEIDRGLSIKLNGSTLLCTDLESKSRMINFLDAPGHVNFMDETAVALAASDLVLIVIDVVEGVTFVVEQLIKQSIKNNVAMCFVINKLDRLILDLKLPPMDAYLKLNHIIANINSFTKGNVFSPIDNNIIFASTKLGFTFTIKEFVSYYYAHSIPSSKIDDFTTRLWGSVYYHKGNFRTKPFENVEKYPTFVEFILIPLYKIFSYALSMEKDKLKNLLRSNFRVNLSQEALQYDPQPFLKHVLQLIFRQQTGLVDAITRCYQPFELFDNKTAHLSIPGKSTPEGTLWAHVLKTVDYGGAEWSLVRIYSGLLKRGDTVRILDTSQSESRQKRQLHDISKTETSNEDEDSKTETPSCEVEEIGLLGGRYVYPVHEAHKGQIVLIKGISSAYIKSATLYSVKSKEDMKQLKFFKPLDYITEAVFKIVLQPLLPRELPKLLDALNKISKYYPGVIIKVEESGEHVILGNGELYMDCLLYDLRASYAKIEIKISDPLTVFSESCSNESFASIPVSNSISRLGEENLPGLSISVAAEPMDSKMIQDLSRNTLGKGQNCLDIDGIMDNPRKLSKILRTEYGWDSLASRNVWSFYNGNVLINDTLPDEISPELLSKYKEQIIQGFYWAVKEGPLAEEPIYGVQYKLLSISVPSDVNIDVMKSQIIPLMKKACYVGLLTAIPILLEPIYEVDITVHAPLLPIVEELMKKRRGSRIYKTIKVAGTPLLEVRGQVPVIESAGFETDLRLSTNGLGMCQLYFWHKIWRKVPGDVLDKDAFIPKLKPAPINSLSRDFVMKTRRRKGISTGGFMSNDGPTLEKYISAELYAQLRENGLVP",
        "min_sequences_in_msa": 1000,  # common protein, should have many sequences
        "spot_check": {
            "index": 1,
            "sequence": "--MDDLYDEFGQFLGFPQEFTSYEQSSEE----VQGEAAYSTLQG-DLDQE-------ATDVVLDANGNFDDDVEVLLEVED-REPDKPLVAGDL-------RPKGYDKCDKIPKAMFDREYLQSILAIPERQLNVGIFGPLHSGKTSFADMFALDTHHNLPSLTKKVKEGWLPFKYLDQERIEKERGVSLRLNGMTFGYESSRGRTYAVTMLDTPGHVNFWDDVGITLTCCQYGIVVIDVAEGVTSVVLKLFKELEQNGIEFIVVLNKIDRLALDLRLPADAAYWRLLHIVEQVNRHTKE-TFSPELGNVLFSSTKFGFVFSIESFVNSFYAKSLKDK-TEQFVARLWGLINYWDGEFNETEF--ISERNSFFVFILQPLYKVITHGLSASAEELQRVIKDNFQVNLSDETLSKDPQPLLFSIFRSIFPHHHCVIDSISRLRDRSFDISA-----------ND-GETLVHVLRHIKVNGTNWSLCRIAQGSLITGRKLYIFNESVDSIVDHAD--------------D---EYPKITIERIALMGGRYAYEVKEAQQGQLVLLKGFEDEFTKFATLS-------STVRNPLPPINYLNESVFKFAIQPQKPSDLPRLLHGLQLANGFYPSLVVRVEESGENIIVGTGELYLDCVMDELRKTFCEIEIKISQPLVQITESCNSESFASIPVKSNNGI--------VSISVMAEKLDDKIVHDLTHGEIN--------LSELNNVRKFSKRLRTEYGWDSLAARNFWGLSQCNVFVDDTLPDETDKKLLKRYKEYILQGFEWAVKEGPLADERMHACQFKLLELKVQEDKIDEFIPSQLVPLTRKACYIALMTAAPIVMEPIYEVDIV-------------------------NVQGTPFTEIKAQLPVIESIGFETDLRTATIGKGMCQMHFWNKIWRRVPGDVLDEEAFIPKLKPAPAASLSRDFVVKTRRRKGLSESGHMTQDGPSLKNYIDDELFEKLKQKGLV-",
            "num_insertions": 2,
            "tax_id": "1427455",
        },
    },
    {
        # RNA
        "pdb_id": "5gam",
        "chain_id": "A",
        "sequence": "AAGCAGCUUUACAGAUCAAUGGCGGAGGGAGGUCAACAUCAAGAACUGUGGGCCUUUUAUUGCCUAUAGAACUUAUAACGAACAUGGUUCUUGCCUUUUACCAGAACCAUCCGGGUGUUGUCUCCAUAGAAACAGGUAAAGCUGUCCGUUACUGUGGGCUUGCCAUAUUUUUUGGAAC",
        "min_sequences_in_msa": 10,  # NA MSAs are much shorter than protein MSAs
        "spot_check": {
            "index": 1,
            "sequence": "AAGCAGCUUGACAGAUCAAUGGCGGAGGGAGGUCAACAUCAAGAACUGUGGGACUUUUAUUGCCUAUAGAACUUAUAACGAACAUGGUUCUUGCCUUUUACCAGAACCAUCCGGGUGUUGUCUCCAUAGAAACAGGUAAAGCUGUCCGUUACUGCCAGCUUGCCAUAUUUUUUGGAA-",
            "num_insertions": 0,
            "tax_id": "",
        },
    },
]


@pytest.mark.parametrize("test_case", MSA_TEST_CASES)
def test_load_msas(test_case: dict[str, Any]):
    """
    Test a series of hand-picked cases to ensure that the MSA loading pipeline is functioning correctly.
    We will check that the MSA has a minimum number of sequences, that the query sequence is correct, and that a spot check is correct.
    """
    # Load a row from the pn_units dataframe with the given PDB ID and chain ID
    pdb_id = test_case["pdb_id"]
    chain_id = test_case["chain_id"]
    row = PN_UNITS_DF[(PN_UNITS_DF["pdb_id"] == pdb_id.lower()) & (PN_UNITS_DF["q_pn_unit_id"] == chain_id)].iloc[0]
    chain_type = ChainType(row["q_pn_unit_type"])

    assert row is not None

    # Process the row
    data = load_example_from_metadata_row(row, PNUnitsDFParser())

    # Apply transforms
    # fmt: off
    pipeline = Compose([
        RemoveHydrogens(),
        RemoveUnsupportedChainTypes(),
        LoadPolymerMSAs(
            protein_msa_dirs=PROTEIN_MSA_DIRS, 
            rna_msa_dirs=RNA_MSA_DIRS, 
            max_msa_sequences=2000,
            msa_cache_dir=None
        ),
    ], track_rng_state=False)
    # fmt: on

    output = pipeline(data)
    result = output["polymer_msas_by_chain_id"][chain_id]
    spot_check_index = test_case["spot_check"]["index"]

    # Check that the MSA has a minimum number of sequences
    assert result["msa"].shape[0] >= test_case["min_sequences_in_msa"]

    # Check that the sequence is correct (first row)
    lookup_table = (
        AMINO_ACID_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE
        if chain_type.is_protein()
        else RNA_NUCLEOTIDE_ONE_LETTER_ASCII_TO_INT_LOOKUP_TABLE
    )
    assert np.all(result["msa"][0] == lookup_table[np.frombuffer(test_case["sequence"].encode(), dtype=np.int8)])

    # For our spot check, we will check the sequence, number of insertions, shapes, and tax ID
    assert np.all(
        result["msa"][spot_check_index]
        == lookup_table[np.frombuffer(test_case["spot_check"]["sequence"].encode(), dtype=np.int8)]
    )
    assert np.sum(result["ins"][spot_check_index]) == test_case["spot_check"]["num_insertions"]
    assert result["ins"][spot_check_index].shape[0] == len(result["msa"][spot_check_index])
    assert result["tax_ids"][spot_check_index].item() == str(test_case["spot_check"]["tax_id"])

    # Compute the sequence similarity between the spot check sequence and the query sequence
    calculated_sequence_similarity = np.mean(result["msa"][spot_check_index] == result["msa"][0])
    assert calculated_sequence_similarity == result["sequence_similarity"][spot_check_index]
    assert result["sequence_similarity"][0] == 1.0  # query sequence should have 100% similarity with itself


@pytest.mark.slow
@pytest.mark.parametrize("test_case", MSA_TEST_CASES)
def test_cache_msas(test_case: dict[str, Any], tmp_path: str):
    """
    Tests the MSA caching functionality by loading the same MSA with and without caching and comparing the results.
    """
    # ...same as above
    pdb_id = test_case["pdb_id"]
    chain_id = test_case["chain_id"]
    row = PN_UNITS_DF[(PN_UNITS_DF["pdb_id"] == pdb_id.lower()) & (PN_UNITS_DF["q_pn_unit_id"] == chain_id)].iloc[0]
    assert row is not None
    data = load_example_from_metadata_row(row, PNUnitsDFParser())

    # ...perform setup
    common_pipeline = Compose(
        [
            RemoveHydrogens(),
            RemoveUnsupportedChainTypes(),
        ]
    )
    result_before_msa_loading = common_pipeline(data)

    # ...load with caching turned off
    # fmt: off
    no_cache_pipeline = Compose([
        LoadPolymerMSAs(
            protein_msa_dirs=PROTEIN_MSA_DIRS, 
            rna_msa_dirs=RNA_MSA_DIRS, 
            max_msa_sequences=10000,
            msa_cache_dir=None
        ),
    ], track_rng_state=False)
    # fmt: on
    start_time = time.time()
    out_without_cache = no_cache_pipeline(copy.deepcopy(result_before_msa_loading))
    first_run_time = time.time() - start_time

    # ...load with caching turned on
    # fmt: off
    cache_pipeline = Compose([
        LoadPolymerMSAs(
            protein_msa_dirs=PROTEIN_MSA_DIRS, 
            rna_msa_dirs=RNA_MSA_DIRS, 
            max_msa_sequences=10000,
            msa_cache_dir=tmp_path / "msa_cache"
        ),
    ], track_rng_state=False)
    # fmt: on
    out_with_cache_1 = cache_pipeline(copy.deepcopy(result_before_msa_loading))

    # ...and again
    start_time = time.time()
    out_with_cache_2 = cache_pipeline(copy.deepcopy(result_before_msa_loading))
    last_run_time = time.time() - start_time

    # ...check that the results are the same
    for key in out_without_cache["polymer_msas_by_chain_id"][chain_id]:
        assert np.array_equal(
            out_without_cache["polymer_msas_by_chain_id"][chain_id][key],
            out_with_cache_1["polymer_msas_by_chain_id"][chain_id][key],
        )
        assert np.array_equal(
            out_with_cache_1["polymer_msas_by_chain_id"][chain_id][key],
            out_with_cache_2["polymer_msas_by_chain_id"][chain_id][key],
        )

    # ...check that the second run was faster by at least 10%
    if first_run_time > 1:
        assert last_run_time < first_run_time * 0.9


def calculate_msa_coverage_of_pdb_id(pdb_id):
    """Take as input a PDB ID and return the number of proteins and RNA/DNA chains with MSAs. Used in the MSA coverage test."""
    num_proteins = num_proteins_with_msas = num_rna = num_rna_with_msa = num_dna = num_dna_with_msa = 0

    # Filter rows by PDB ID and chain type (we're looking for the query row PN unit only)
    filtered_rows = FILTERED_PN_UNITS_DF[FILTERED_PN_UNITS_DF["pdb_id"] == pdb_id.lower()]

    # Loop through query pn_units and check if they have MSAs
    for _, row in filtered_rows.iterrows():
        chain_type = ChainType(row["q_pn_unit_type"])
        sequence = row["q_pn_unit_processed_entity_non_canonical_sequence"]
        if chain_type.is_protein():
            if not sequence:
                logger.warning(f"Protein chain with no sequence: {pdb_id} : {row['q_pn_unit_iid']}")
                continue
            # Ignore short proteins
            elif len(sequence) > 50:
                # Hash the sequence so we can look up the MSA file
                sequence_hash = hash_sequence(sequence)
                num_proteins += 1

                # ...loop through all protein MSA directories, checking if the requested MSA file exists
                for protein_msa_dir in PROTEIN_MSA_DIRS:
                    protein_msa_file = get_sharded_file_path(
                        protein_msa_dir["dir"],
                        sequence_hash,
                        protein_msa_dir["extension"],
                        protein_msa_dir["directory_depth"],
                    )
                    if protein_msa_file.exists():
                        num_proteins_with_msas += 1
                        break

        elif chain_type == ChainType.RNA:
            if not sequence:
                logger.warning(f"RNA chain with no sequence: {pdb_id} : {row['q_pn_unit_iid']}")
                continue

            num_rna += 1
            # Handle legacy behavior where RNA MSAs use T instead of U
            sequence = sequence.replace("U", "T")
            # Hash the sequence so we can look up the MSA file
            sequence_hash = hash_sequence(sequence)

            # ...loop through all RNA MSA directories, checking if the requested MSA file exists
            for rna_msa_dir in RNA_MSA_DIRS:
                rna_msa_file = get_sharded_file_path(
                    rna_msa_dir["dir"], sequence_hash, rna_msa_dir["extension"], rna_msa_dir["directory_depth"]
                )
                if rna_msa_file.exists():
                    num_rna_with_msa += 1
                    break

        elif chain_type == ChainType.DNA:
            num_dna += 1
            # DNA - always load in single sequence mode (no MSAs)
        else:
            raise ValueError(f"Unexpected chain type: {chain_type}")

    return {
        "pdb_id": pdb_id,
        "num_proteins": num_proteins,
        "num_proteins_with_msas": num_proteins_with_msas,
        "num_rna": num_rna,
        "num_rna_with_msa": num_rna_with_msa,
        "num_dna": num_dna,
        "num_dna_with_msa": num_dna_with_msa,
    }


# Build up a filtered pn_units dataframe (used in the MSA coverage tests)
FILTERED_PN_UNITS_DF = PN_UNITS_DF.copy()
FILTERED_PN_UNITS_DF["q_pn_unit_type"] = FILTERED_PN_UNITS_DF["q_pn_unit_type"].apply(lambda x: ChainType(x))
FILTERED_PN_UNITS_DF = FILTERED_PN_UNITS_DF[
    FILTERED_PN_UNITS_DF["q_pn_unit_type"].isin(
        [ChainType.POLYPEPTIDE_D, ChainType.POLYPEPTIDE_L, ChainType.RNA, ChainType.DNA]
    )
]

# Filter to only entries with a deposition date before August 2nd, 2021
FILTERED_PN_UNITS_DF = FILTERED_PN_UNITS_DF[FILTERED_PN_UNITS_DF["deposition_date"] < "2021-08-02"]


def test_msa_coverage():
    """
    Function to validate MSA coverage for a random subset of PDB IDs.
    Asserts that the coverage for proteins, RNA, and DNA is above a specified threshold.
    """
    PROTEIN_COVERAGE_THRESHOLD = 0.90  # NOTE: We will increase this threshold in the future
    RNA_COVERAGE_THRESHOLD = 0.10  # NOTE: We will increase this threshold in the future
    DNA_COVERAGE_THRESHOLD = 0.0  # NOTE: Currently, we do not compute DNA MSAs

    pdb_ids = FILTERED_PN_UNITS_DF["pdb_id"].unique()

    aggregate_results = {
        "num_proteins": 0,
        "num_proteins_with_msas": 0,
        "num_rna": 0,
        "num_rna_with_msa": 0,
        "num_dna": 0,
        "num_dna_with_msa": 0,
    }

    num_processes = min(8, os.cpu_count() or 8)
    with Pool(processes=num_processes) as pool:
        results_generator = pool.imap(calculate_msa_coverage_of_pdb_id, pdb_ids, chunksize=20)
        for results in tqdm(results_generator, total=len(pdb_ids)):
            if results is None:
                continue

            aggregate_results["num_proteins"] += results["num_proteins"]
            aggregate_results["num_proteins_with_msas"] += results["num_proteins_with_msas"]
            aggregate_results["num_rna"] += results["num_rna"]
            aggregate_results["num_rna_with_msa"] += results["num_rna_with_msa"]
            aggregate_results["num_dna"] += results["num_dna"]
            aggregate_results["num_dna_with_msa"] += results["num_dna_with_msa"]

    pool.join()  # see https://pytest-cov.readthedocs.io/en/v2.6.1/mp.html

    # Assert protein ratio is > 95% and RNA and DNA ratios are > 80%
    assert (
        aggregate_results["num_proteins"] == 0
        or aggregate_results["num_proteins_with_msas"] / aggregate_results["num_proteins"] >= PROTEIN_COVERAGE_THRESHOLD
    )
    assert (
        aggregate_results["num_rna"] == 0
        or aggregate_results["num_rna_with_msa"] / aggregate_results["num_rna"] >= RNA_COVERAGE_THRESHOLD
    )
    assert (
        aggregate_results["num_dna"] == 0
        or aggregate_results["num_dna_with_msa"] / aggregate_results["num_dna"] >= DNA_COVERAGE_THRESHOLD
    )

    # Log the results
    logger.info(
        f"Proteins: {aggregate_results['num_proteins_with_msas']}/{aggregate_results['num_proteins']} ({(aggregate_results['num_proteins_with_msas'] / aggregate_results['num_proteins'] * 100 if aggregate_results['num_proteins'] else 0):.2f}%)"
    )
    logger.info(
        f"RNA: {aggregate_results['num_rna_with_msa']}/{aggregate_results['num_rna']} ({(aggregate_results['num_rna_with_msa'] / aggregate_results['num_rna'] * 100 if aggregate_results['num_rna'] else 0):.2f}%)"
    )
    logger.info(
        f"DNA: {aggregate_results['num_dna_with_msa']}/{aggregate_results['num_dna']} ({(aggregate_results['num_dna_with_msa'] / aggregate_results['num_dna'] * 100 if aggregate_results['num_dna'] else 0):.2f}%)"
    )


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
