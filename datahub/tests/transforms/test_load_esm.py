import copy
import logging
import os
import time
from multiprocessing import Pool
from typing import Any

import numpy as np
import pytest
import torch
from cifutils.enums import ChainType
from tqdm import tqdm

from datahub.datasets.parsers import PNUnitsDFParser, load_example_from_metadata_row
from datahub.transforms.base import Compose
from datahub.transforms.esm.esm import LoadPolymerESMs
from datahub.transforms.filters import RemoveHydrogens, RemoveUnsupportedChainTypes
from datahub.utils.io import get_sharded_file_path
from datahub.utils.misc import hash_sequence
from tests.conftest import CIF_PARSER, PN_UNITS_DF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ESM_TEST_CASES = [
    {
        "pdb_id": "5gam",
        "chain_id": "C",
        "sequence": "MEGDDLFDEFGNLIGVDPFDSDEEESVLDEQEQYQTNTFEGSGNNNEIESRQLTSLGSKKELGISLEHPYGKEVEVLMETKNTQSPQTPLVEPVTERTKLQEHTIFTQLKKNIPKTRYNRDYMLSMANIPERIINVGVIGPLHSGKTSLMDLLVIDSHKRIPDMSKNVELGWKPLRYLDNLKQEIDRGLSIKLNGSTLLCTDLESKSRMINFLDAPGHVNFMDETAVALAASDLVLIVIDVVEGVTFVVEQLIKQSIKNNVAMCFVINKLDRLILDLKLPPMDAYLKLNHIIANINSFTKGNVFSPIDNNIIFASTKLGFTFTIKEFVSYYYAHSIPSSKIDDFTTRLWGSVYYHKGNFRTKPFENVEKYPTFVEFILIPLYKIFSYALSMEKDKLKNLLRSNFRVNLSQEALQYDPQPFLKHVLQLIFRQQTGLVDAITRCYQPFELFDNKTAHLSIPGKSTPEGTLWAHVLKTVDYGGAEWSLVRIYSGLLKRGDTVRILDTSQSESRQKRQLHDISKTETSNEDEDSKTETPSCEVEEIGLLGGRYVYPVHEAHKGQIVLIKGISSAYIKSATLYSVKSKEDMKQLKFFKPLDYITEAVFKIVLQPLLPRELPKLLDALNKISKYYPGVIIKVEESGEHVILGNGELYMDCLLYDLRASYAKIEIKISDPLTVFSESCSNESFASIPVSNSISRLGEENLPGLSISVAAEPMDSKMIQDLSRNTLGKGQNCLDIDGIMDNPRKLSKILRTEYGWDSLASRNVWSFYNGNVLINDTLPDEISPELLSKYKEQIIQGFYWAVKEGPLAEEPIYGVQYKLLSISVPSDVNIDVMKSQIIPLMKKACYVGLLTAIPILLEPIYEVDITVHAPLLPIVEELMKKRRGSRIYKTIKVAGTPLLEVRGQVPVIESAGFETDLRLSTNGLGMCQLYFWHKIWRKVPGDVLDKDAFIPKLKPAPINSLSRDFVMKTRRRKGISTGGFMSNDGPTLEKYISAELYAQLRENGLVP",
    },
]


ESM_HOMO_TEST_CASES = [
    [
        {
            "pdb_id": "3sv4",
            "chain_id": "A",
            "sequence": "ALEEAPWPPPEGAFVGFVLSRKEPMWADLLALAAARGGRVHRAPEPYKALRDLKEARGLLAKDLSVLALREGLGLPPGDDPMLLAYLLDPSNTTPEGVARRYGGEWTEEAGERAALSERLFANLWGRLEGEERLLWLYREVERPLSAVLAHMEATGVRLDVAYLRALSLEVAEEIARLEAEVFRLAGHPFNLNSRDQLERVLFDELGLPAIGKTEKTGKRSTSAAVLEALREAHPIVEKILQYRELTKLKSTYIDPLPDLIHPRTGRLHTRFNQTATATGRLSSSDPNLQNIPVRTPLGQRIRRAFIAEEGWLLVALDYSQIELRVLAHLSGDENLIRVFQEGRDIHTETASWMFGVPREAVDPLMRRAAKTINFGVLYGMSAHRLSQELAIPYEEAQAFIERYFQSFPKVRAWIEKTLEEGRRRGYVETLFGRRRYVPDLEARVKSVREAAERMAFNMPVQGTAADLMKLAMVKLFPRLEEMGARMLLQVHDELVLEAPKERAEAVARLAKEVMEGVYPLAVPLEVEVGIGEDWLSAKE",
        },
        {
            "pdb_id": "1p6l",
            "chain_id": "A",
            "sequence": "GPKFPRVKNWELGSITYDTLCAQSQQDGPCTPRRCLGSLVLPRKLQTRPSPGPPPAEQLLSQARDFINQYYSSIKRSGSQAHEERLQEVEAEVASTGTYHLRESELVFGAKQAWRNAPRCVGRIQWGKLQVFDARDCSSAQEMFTYICNHIKYATNRGNLRSAITVFPQRAPGRGDFRIWNSQLVRYAGYRQQDGSVRGDPANVEITELCIQHGWTPGNGRFDVLPLLLQAPDEAPELFVLPPELVLEVPLEHPTLEWFAALGLRWYALPAVSNMLLEIGGLEFSAAPFSGWYMSTEIGTRNLCDPHRYNILEDVAVCMDLDTRTTSSLWKDKAAVEINLAVLHSFQLAKVTIVDHHAATVSFMKHLDNEQKARGGCPADWAWIVPPISGSLTPVFHQEMVNYILSPAFRYQPDPWK",
        },
        {
            "pdb_id": "3ti0",
            "chain_id": "A",
            "sequence": "MESPSSEEEKPLAKMAFTLADRVTEEMLADKAALVVEVVEENYHDAPIVGIAVVNEHGRFFLRPETALADPQFVAWLGDETKKKSMFDSKRAAVALKWKGIELCGVSFDLLLAAYLLDPAQGVDDVAAAAKMKQYEAVRPDEAVYGKGAKRAVPDEPVLAEHLVRKAAAIWELERPFLDELRRNEQDRLLVELEQPLSSILAEMEFAGVKVDTKRLEQMGKELAEQLGTVEQRIYELAGQEFNINSPKQLGVILFEKLQLPVLKKTKTGYSTSADVLEKLAPYHEIVENILHYRQLGKLQSTYIEGLLKVVRPATKKVHTIFNQALTQTGRLSSTEPNLQNIPIRLEEGRKIRQAFVPSESDWLIFAADYSQIELRVLAHIAEDDNLMEAFRRDLDIHTKTAMDIFQVSEDEVTPNMRRQAKAVNYGIVYGISDYGLAQNLNISRKEAAEFIERYFESFPGVKRYMENIVQEAKQKGYVTTLLHRRRYLPDITSRNFNVRSFAERMAMNTPIQGSAADIIKKAMIDLNARLKEERLQAHLLLQVHDELILEAPKEEMERLCRLVPEVMEQAVTLRVPLKVDYHYGSTWYDAK",
        },
    ],
]

EMBEDDING_DIM = 2560

ESM_EMBEDDING_DIRS = [
    {
        "dir": "/projects/ml/RF2_allatom/PDB_ESM_embedding_full",
        "extension": ".pt",
        "directory_depth": 2,
    },
]


@pytest.mark.parametrize("test_case", ESM_TEST_CASES)
def test_load_esms(test_case: dict[str, Any]):
    """
    Test the LoadPolymerESMs class to ensure that ESM embeddings are loaded correctly,
    and that the padding behavior works as expected when embeddings are missing.
    """
    # Load a row from the pn_units dataframe with the given PDB ID and chain ID
    pdb_id = test_case["pdb_id"]
    chain_id = test_case["chain_id"]

    # Ensure PDB ID is lowercase in the dataframe
    row = PN_UNITS_DF[(PN_UNITS_DF["pdb_id"] == pdb_id.lower()) & (PN_UNITS_DF["q_pn_unit_id"] == chain_id)].iloc[0]

    assert row is not None

    # Process the row
    data = load_example_from_metadata_row(row, PNUnitsDFParser(), cif_parser=CIF_PARSER)

    # Apply transforms
    pipeline = Compose(
        [
            RemoveHydrogens(),
            RemoveUnsupportedChainTypes(),
            LoadPolymerESMs(esm_embedding_dirs=ESM_EMBEDDING_DIRS, esm_cache_dir=None),
        ],
        track_rng_state=False,
    )

    output = pipeline(data)
    polymer_esms_by_chain_id = output["polymer_esms_by_chain_id"]
    result = polymer_esms_by_chain_id.get(chain_id)

    sequence_length = len(test_case["sequence"])

    esm_embedding = result["esm_embedding"]
    esm_is_padded_mask = result["esm_is_padded_mask"]

    # Check the shape of the embedding
    assert esm_embedding.shape == (sequence_length, EMBEDDING_DIM), f"ESM embedding shape mismatch for chain {chain_id}"

    # Check the padding mask and embedding content
    if not np.all(esm_is_padded_mask):
        # Embedding was found
        pass
    elif np.all(esm_is_padded_mask):
        # Embedding was not found, and zero-filled embedding was created
        # Verify that the embedding is all zeros
        assert np.all(esm_embedding == 0), f"Embedding for chain {chain_id} should be zero-filled"
    else:
        # Mixed padding mask is unexpected
        assert False, f"Unexpected esm_is_padded_mask for chain {chain_id}"


@pytest.mark.skip
@pytest.mark.parametrize("test_case", ESM_TEST_CASES)
def test_cache_esms(test_case: dict[str, Any], tmp_path: str):
    """
    Tests the ESM caching functionality by loading the same ESM embedding with and without caching and comparing the results.
    """
    pdb_id = test_case["pdb_id"]
    chain_id = test_case["chain_id"]
    row = PN_UNITS_DF[(PN_UNITS_DF["pdb_id"] == pdb_id.lower()) & (PN_UNITS_DF["q_pn_unit_id"] == chain_id)].iloc[0]

    assert row is not None

    data = load_example_from_metadata_row(row, PNUnitsDFParser(), cif_parser=CIF_PARSER)

    # Perform setup
    common_pipeline = Compose(
        [
            RemoveHydrogens(),
            RemoveUnsupportedChainTypes(),
        ]
    )
    result_before_esm_loading = common_pipeline(data)

    # Load with caching turned off
    no_cache_pipeline = Compose(
        [
            LoadPolymerESMs(esm_embedding_dirs=ESM_EMBEDDING_DIRS, esm_cache_dir=None),
        ],
        track_rng_state=False,
    )
    start_time = time.time()
    out_without_cache = no_cache_pipeline(copy.deepcopy(result_before_esm_loading))
    first_run_time = time.time() - start_time

    # Load with caching turned on
    cache_pipeline = Compose(
        [
            LoadPolymerESMs(esm_embedding_dirs=ESM_EMBEDDING_DIRS, esm_cache_dir=tmp_path / "esm_cache"),
        ],
        track_rng_state=False,
    )
    out_with_cache_1 = cache_pipeline(copy.deepcopy(result_before_esm_loading))

    # Load again to test caching speed
    start_time = time.time()
    out_with_cache_2 = cache_pipeline(copy.deepcopy(result_before_esm_loading))
    last_run_time = time.time() - start_time

    # Check that the results are the same
    for key in out_without_cache["polymer_esms_by_chain_id"][chain_id]:
        assert np.array_equal(
            out_without_cache["polymer_esms_by_chain_id"][chain_id][key],
            out_with_cache_1["polymer_esms_by_chain_id"][chain_id][key],
        )
        assert np.array_equal(
            out_with_cache_1["polymer_esms_by_chain_id"][chain_id][key],
            out_with_cache_2["polymer_esms_by_chain_id"][chain_id][key],
        )

    # Check that the second run was faster by at least 10%
    if first_run_time > 1:
        assert last_run_time < first_run_time * 0.9


def get_esm_coverage_for_pdb_id(pdb_id):
    """
    Process a PDB ID and return the number of protein chains with ESM embeddings.
    Used in the ESM coverage test.
    """
    num_proteins = num_proteins_with_esm = 0

    # Filter rows by PDB ID
    filtered_rows = FILTERED_PN_UNITS_DF[FILTERED_PN_UNITS_DF["pdb_id"] == pdb_id.lower()]

    # Loop through chains and check for ESM embeddings
    for _, row in filtered_rows.iterrows():
        chain_type = ChainType(row["q_pn_unit_type"])
        sequence = row["q_pn_unit_processed_entity_non_canonical_sequence"]
        if chain_type.is_protein():
            if not sequence:
                logger.warning(f"Protein chain with no sequence: {pdb_id} : {row['q_pn_unit_id']}")
                continue
            elif len(sequence) > 50:
                sequence_hash = hash_sequence(sequence)
                num_proteins += 1

                # Check if ESM embedding exists
                for esm_dir in ESM_EMBEDDING_DIRS:
                    esm_file = get_sharded_file_path(
                        esm_dir["dir"],
                        sequence_hash,
                        esm_dir["extension"],
                        esm_dir.get("directory_depth", 0),
                    )
                    if esm_file.exists():
                        num_proteins_with_esm += 1
                        break

    return {
        "pdb_id": pdb_id,
        "num_proteins": num_proteins,
        "num_proteins_with_esm": num_proteins_with_esm,
    }


# Build up a filtered pn_units dataframe (used in the ESM coverage tests)
FILTERED_PN_UNITS_DF = PN_UNITS_DF.copy()
FILTERED_PN_UNITS_DF["q_pn_unit_type"] = FILTERED_PN_UNITS_DF["q_pn_unit_type"].apply(lambda x: ChainType(x))
FILTERED_PN_UNITS_DF = FILTERED_PN_UNITS_DF[
    FILTERED_PN_UNITS_DF["q_pn_unit_type"].isin([ChainType.POLYPEPTIDE_D, ChainType.POLYPEPTIDE_L])
]

# Filter to only entries with a deposition date before a certain date, if necessary
FILTERED_PN_UNITS_DF = FILTERED_PN_UNITS_DF[FILTERED_PN_UNITS_DF["deposition_date"] < "2021-08-02"]


@pytest.mark.slow
def test_esm_coverage():
    """
    Function to validate ESM embedding coverage for a set of PDB IDs.
    Asserts that the coverage for proteins is above a specified threshold.
    """
    PROTEIN_COVERAGE_THRESHOLD = 0.90

    pdb_ids = FILTERED_PN_UNITS_DF["pdb_id"].unique()

    aggregate_results = {
        "num_proteins": 0,
        "num_proteins_with_esm": 0,
    }

    num_processes = min(8, os.cpu_count() or 8)
    with Pool(processes=num_processes) as pool:
        results_generator = pool.imap(get_esm_coverage_for_pdb_id, pdb_ids, chunksize=20)
        for results in tqdm(results_generator, total=len(pdb_ids)):
            if results is None:
                continue

            aggregate_results["num_proteins"] += results["num_proteins"]
            aggregate_results["num_proteins_with_esm"] += results["num_proteins_with_esm"]

    pool.join()  # See https://pytest-cov.readthedocs.io/en/v2.6.1/mp.html

    # Assert protein ratio is above the threshold
    if aggregate_results["num_proteins"] > 0:
        coverage = aggregate_results["num_proteins_with_esm"] / aggregate_results["num_proteins"]
        assert (
            coverage >= PROTEIN_COVERAGE_THRESHOLD
        ), f"ESM embedding coverage {coverage:.2%} is below the threshold {PROTEIN_COVERAGE_THRESHOLD:.2%}"

    # Log the results
    logger.info(
        f"Proteins with ESM embeddings: {aggregate_results['num_proteins_with_esm']}/"
        f"{aggregate_results['num_proteins']} "
        f"({(coverage * 100 if aggregate_results['num_proteins'] else 0):.2f}%)"
    )


@pytest.mark.parametrize("test_cases_homo", ESM_HOMO_TEST_CASES)
def test_esm_homology(test_cases_homo: list):
    # We test whether ESM embeddings capture the correct info
    # input 3 sequence : Chain A; Chain B; Chain A^hat
    # Chain A and Chain A^hat are homologous protein from human and mouse

    # Load esm embeddings
    esm_data = []
    for test_case in test_cases_homo:
        pdb_id = test_case["pdb_id"]
        chain_id = test_case["chain_id"]

        # Ensure PDB ID is lowercase in the dataframe
        row = PN_UNITS_DF[(PN_UNITS_DF["pdb_id"] == pdb_id.lower()) & (PN_UNITS_DF["q_pn_unit_id"] == chain_id)].iloc[0]

        assert row is not None

        # Process the row
        data = load_example_from_metadata_row(row, PNUnitsDFParser(), cif_parser=CIF_PARSER)

        # Apply transforms
        pipeline = Compose(
            [
                RemoveHydrogens(),
                RemoveUnsupportedChainTypes(),
                LoadPolymerESMs(esm_embedding_dirs=ESM_EMBEDDING_DIRS, esm_cache_dir=None),
            ],
            track_rng_state=False,
        )

        output = pipeline(data)
        polymer_esms_by_chain_id = output["polymer_esms_by_chain_id"]
        result = polymer_esms_by_chain_id.get(chain_id)

        esm_embedding = result["esm_embedding"]
        esm_data.append(torch.tensor(esm_embedding.mean(0)))

    # compare cosine similarity
    esm_data = torch.stack(esm_data)
    esm_data = esm_data / torch.norm(esm_data, dim=-1, keepdim=True)

    # cosine similarity between Chain A and Chain B should be smaller than Chain A and Chain A^hat
    assert torch.dot(esm_data[0], esm_data[1]) < torch.dot(esm_data[0], esm_data[2])
    print("Cosine similarity between Chain A and Chain B: ", torch.dot(esm_data[0], esm_data[1]))
    print("Cosine similarity between Chain A and Chain A^hat: ", torch.dot(esm_data[0], esm_data[2]))


if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])
