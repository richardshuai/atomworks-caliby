# This script is modified from the `esm/scripts/extract.py` from the ESM repository.

# given a dict of {sequence_hash: sequence}, generate ESM embeddings for each sequence.

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pathlib

import torch
from esm import FastaBatchedDataset, MSATransformer, pretrained


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )

    parser.add_argument(
        "model_location",
        type=str,
        default="esm2_t36_3B_UR50D",
        help="PyTorch model file OR name of pretrained model to download (see README for models)",
    )
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )

    parser.add_argument("--toks_per_batch", type=int, default=4096, help="maximum batch size")
    parser.add_argument(
        "--repr_layers",
        type=int,
        default=[-1],
        nargs="+",
        help="layers indices from which to extract representations (0 to num_layers, inclusive)",
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        choices=["all_layers", "last_layer"],
        help="specify which representations to return",
        required=True,
    )
    parser.add_argument(
        "--truncation_seq_length",
        type=int,
        default=10000,
        help="truncate sequences longer than the given value",
    )

    parser.add_argument("--nogpu", action="store_true", help="Do not use GPU even if available")
    return parser


def run(args):
    model, alphabet = pretrained.load_model_and_alphabet(args.model_location)
    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError("This script currently does not handle models with MSA input (MSA Transformer).")
    if torch.cuda.is_available() and not args.nogpu:
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(args.fasta_file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches
    )
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in args.include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in args.repr_layers)
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    representations = {}
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            if "all_layers" in args.include:
                for sequence_hash, representation in zip(labels, out["representations"][model.num_layers]):
                    # save float32 to save disk memory
                    representations[sequence_hash] = torch.stack(
                        [out["representations"][i] for i in range(1, model.num_layers + 1)]
                    ).to(dtype=torch.float32)
            elif "last_layer" in args.include:
                for sequence_hash, representation in zip(labels, out["representations"][model.num_layers]):
                    representations[sequence_hash] = representation[1:-1]
            else:
                raise ValueError

        # store the representations into 2-layer dict
        for sequence_hash, representation in representations.items():
            output_dir = os.path.join(args.output_dir, sequence_hash[:2], sequence_hash[2:4], sequence_hash + ".pt")
            if not os.path.exists(os.path.dirname(output_dir)):
                if not os.path.exists(os.path.dirname(os.path.dirname(output_dir))):
                    os.makedirs(os.path.dirname(os.path.dirname(output_dir)))
            torch.save(representation, output_dir)


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
