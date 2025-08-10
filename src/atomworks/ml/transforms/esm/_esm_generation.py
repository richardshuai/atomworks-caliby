"""Code for generating ESM embeddings
Adapted from the ESM repository: https://github.com/facebookresearch/esm/blob/main/scripts/extract.py
"""

import torch
from esm import FastaBatchedDataset

ESM_2_EMBED_DIM = {
    "esm2_t48_15B_UR50D": 5120,
    "esm2_t36_3B_UR50D": 2560,
    "esm2_t33_650M_UR50D": 1280,
    "esm2_t30_150M_UR50D": 640,
    "esm2_t12_35M_UR50D": 480,
    "esm2_t6_8M_UR50D": 320,
}


def generate_esm_embedding(
    sequences: dict,
    model: dict,
    toks_per_batch: int = 4096,
) -> dict:
    """Generate ESM embeddings for a set of sequences.

    Args:
        sequences: dict
            {sequence_hash: sequence} sequences to be embedded.
        model: dict
            The model dictionary containing the model(ProteinBertModel) \
            and alphabet(for converting input to dataloader).
        toks_per_batch: int
            The number of tokens per batch.

    Returns:
        dict: {sequence_hash: embedding} dictionary of embeddings.
    """
    model, alphabet = model["model"], model["alphabet"]

    # Create a dataset object for ESM
    dataset = FastaBatchedDataset(list(sequences.keys()), [sequences[k] for k in sequences])
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)

    # Pass None to truncation_seq_length to avoid truncation
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(truncation_seq_length=None), batch_sampler=batches
    )
    representations = {}
    with torch.no_grad():
        for _, (labels, _, toks) in enumerate(data_loader):
            out = model(toks, repr_layers=[model.num_layers], return_contacts=False)
            for sequence_hash, representation in zip(labels, out["representations"][model.num_layers], strict=False):
                representations[sequence_hash] = representation[1:-1]  # remove the BOS and EOS tokens

    return representations
