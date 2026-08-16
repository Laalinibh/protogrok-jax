#!/usr/bin/env python
"""Turn UNSW-NB15 Payload-Bytes parquet files into compact per-flow shards.

Each source file is ~2.7 GB of parquet with 1485 columns, but the model only
needs ``payload_max_len`` of the 1476 byte columns plus seven metadata columns.
Parquet is columnar, so ``--source hf`` streams just those column chunks over
HTTPS and never stages the full file on disk. Use ``--source local`` if you
have already downloaded the parquet files.

Output per shard (``PAYLOAD_MAX_LEN=128``, ``MAX_PACKETS_PER_FLOW=16``)::

    payload.npy    uint8 [N, 16, 128]   raw bytes      (2048 B / flow)
    lengths.npy    uint8 [N, 16]        valid lengths    (16 B / flow)
    headers.npy    int32 [N, 5]
    proto_id.npy   int32 [N]
    label_bin.npy  int32 [N]
    label_cls.npy  int32 [N]

Examples
--------
    # Stream file 1 from the Hub, write a shard, never store the parquet:
    python tools/prepare_packets.py --files 1 --out data/shards

    # Preprocess local downloads instead:
    python tools/prepare_packets.py --local /path/Payload_Bytes_File_1.parquet \
        --out data/shards

    # Peek at what one file contains without writing anything:
    python tools/prepare_packets.py --files 1 --max-flows 5000 --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from models.config import MAX_PACKETS_PER_FLOW, PAYLOAD_MAX_LEN  # noqa: E402
from data.packets import FlowShard, hf_url, load_parquet_flows  # noqa: E402


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare Protogrok packet shards")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--files", type=int, nargs="+",
                     help="Payload-Bytes file numbers to stream from the Hub (1-18)")
    src.add_argument("--local", nargs="+", help="local .parquet paths")
    p.add_argument("--out", default=None, help="output directory for shards")
    p.add_argument("--max-packets", type=int, default=MAX_PACKETS_PER_FLOW)
    p.add_argument("--payload-len", type=int, default=PAYLOAD_MAX_LEN)
    p.add_argument("--proto-vocab", type=int, default=132,
                   help="must match the model config used for training")
    p.add_argument("--max-flows", type=int, default=None,
                   help="cap flows per file (useful for a quick smoke run)")
    p.add_argument("--merge", action="store_true",
                   help="write one merged shard instead of one per file")
    p.add_argument("--dry-run", action="store_true",
                   help="report statistics and write nothing")
    args = p.parse_args()

    if not args.dry_run and not args.out:
        p.error("--out is required unless --dry-run is given")

    sources = args.local if args.local else [hf_url(n) for n in args.files]
    labels = args.local if args.local else [f"File_{n}" for n in args.files]

    shards = []
    for source, label in zip(sources, labels):
        print(f"\n=== {label} ===")
        print(f"  reading {source if len(source) < 90 else source[:87] + '...'}")
        t0 = time.time()
        shard = load_parquet_flows(
            source, max_packets=args.max_packets, payload_max_len=args.payload_len,
            proto_vocab=args.proto_vocab, max_flows=args.max_flows)
        dt = time.time() - t0

        nbytes = sum(getattr(shard, n).nbytes for n in FlowShard._ARRAYS)
        pkts = int(shard.headers[:, 3].sum())
        print(f"  flows            : {len(shard):,}")
        print(f"  packets kept     : {pkts:,} (mean {pkts/max(len(shard),1):.1f}/flow)")
        print(f"  attack rate      : {shard.label_bin.mean():.3f}")
        print(f"  class balance    : {shard.class_balance()}")
        print(f"  shard size       : {human(nbytes)}   ({dt:.0f}s)")

        if args.dry_run:
            continue
        if args.merge:
            shards.append(shard)
        else:
            out = os.path.join(args.out, label.replace(".parquet", "").split("/")[-1])
            shard.save(out)
            print(f"  wrote            : {out}")

    if args.merge and shards and not args.dry_run:
        merged = FlowShard.concat(shards)
        merged.save(args.out)
        print(f"\nmerged {len(shards)} shards -> {args.out} "
              f"({len(merged):,} flows, attack rate {merged.label_bin.mean():.3f})")


if __name__ == "__main__":
    main()
