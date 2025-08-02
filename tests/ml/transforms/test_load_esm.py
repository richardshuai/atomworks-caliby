import logging
from typing import Any

import numpy as np
import pytest
import torch

from atomworks.ml.transforms.base import Compose
from atomworks.ml.transforms.esm.esm import LoadPolymerESMs
from atomworks.ml.transforms.filters import RemoveHydrogens, RemoveUnsupportedChainTypes
from atomworks.ml.utils.testing import cached_parse

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
    chain_id = test_case["chain_id"]
    data = cached_parse(test_case["pdb_id"])

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


@pytest.mark.parametrize("test_cases_homo", ESM_HOMO_TEST_CASES)
def test_esm_homology(test_cases_homo: list):
    # We test whether ESM embeddings capture the correct info
    # input 3 sequence : Chain A; Chain B; Chain A^hat
    # Chain A and Chain A^hat are homologous protein from human and mouse

    # Load esm embeddings
    esm_data = []
    for test_case in test_cases_homo:
        chain_id = test_case["chain_id"]
        data = cached_parse(test_case["pdb_id"])

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
