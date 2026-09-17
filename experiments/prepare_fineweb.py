"""Download and tokenize FineWeb into the sharded .bin format the dataloader reads.

Writes shards of 100M GPT-2 tokens each into $DEFMUON_DATA_DIR/<name>-gpt2/
(shard 0 is the val split, the rest are train).  The default --shards 11 gives
the fineweb1B dataset used by the main benchmark: 1 val shard + 10 train shards
= 1B training tokens.

Usage:
    python experiments/prepare_fineweb.py [--shards 11] [--name fineweb1B] [--nprocs 0]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import tiktoken
from datasets import load_dataset

from defmuon.data_utils import process_and_save_docs
from defmuon.dataloader import DATA_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=11,
                        help="number of 100M-token shards to write (shard 0 = val)")
    parser.add_argument("--name", type=str, default="fineweb1B",
                        help="dataset name; shards go to $DEFMUON_DATA_DIR/<name>-gpt2/")
    parser.add_argument("--nprocs", type=int, default=0,
                        help="tokenizer processes; 0 = all cores minus two")
    args = parser.parse_args()

    out_dir = os.path.join(DATA_DIR, f"{args.name}-gpt2")
    os.makedirs(out_dir, exist_ok=True)

    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                           split="train", streaming=True)
    encoding = tiktoken.get_encoding("gpt2")
    process_and_save_docs(dataset, out_dir, encoding, shard_size=int(1e8),
                          nprocs=args.nprocs, max_shards=args.shards)
    print(f"Done: wrote up to {args.shards} shards to {out_dir}")


if __name__ == "__main__":
    main()
