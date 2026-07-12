"""Bin-packing dynamic batcher (the Operations-Research component).

Network flows have highly variable *real* packet counts, yet the naive pipeline
pads every flow to ``MAX_PACKETS`` (=16) before the packet-level Transformer,
whose self-attention cost is O(T^2) in the padded length ``T``. Most of that
compute is spent on padding.

This module reduces the waste with two classic OR techniques:

1. **DP-optimal length bucketing** (:func:`optimal_buckets`) -- given the
   histogram of real packet counts, a dynamic program chooses ``K`` bucket
   upper-edges that *minimise total padding*. Bucketing also bounds the number
   of distinct tensor shapes JAX must compile (one per bucket).

2. **First-Fit-Decreasing bin packing** (:func:`pack_ffd`) -- the classic
   FFD approximation (<= 11/9 * OPT + 1) packs flows into batches under a
   per-batch token budget and cardinality cap, padding each batch only to its
   own (bucketed) max length rather than to the global maximum.

:class:`BinPackedBatcher` applies either strategy and yields ``Batch`` objects
trimmed to ``[B, T_bucket, L]``; the model consumes them unchanged (its packet
axis is already dynamic). :func:`padding_report` quantifies the win.
"""
from __future__ import annotations

import dataclasses
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from models.config import PAD_ID, MAX_PACKETS_PER_FLOW
from data.tokenizer import Batch

Example = Tuple[np.ndarray, np.ndarray, int, int]  # (payload[T,L], header[5], proto_id, label)


@dataclasses.dataclass
class PackingConfig:
    """Configuration for the bin-packing batcher."""
    strategy: str = "bucket"              # "bucket" | "ffd"
    buckets: Optional[Sequence[int]] = None   # explicit bucket upper-edges
    num_buckets: int = 4                  # if buckets is None: DP-optimal this many
    max_batch: int = 64                   # max flows per batch (cardinality cap)
    token_budget: Optional[int] = None    # FFD: max sum of real packets per batch
    max_packets: int = MAX_PACKETS_PER_FLOW
    shuffle: bool = True
    seed: int = 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def real_packet_count(payload: np.ndarray) -> int:
    """Number of non-PAD packet rows in a flow's payload ``[T, L]``."""
    return int(np.any(payload != PAD_ID, axis=1).sum()) or 1


def counts_of(examples: Sequence[Example]) -> np.ndarray:
    return np.asarray([real_packet_count(ex[0]) for ex in examples], dtype=np.int64)


def bucket_for(n: int, buckets: Sequence[int]) -> int:
    for b in buckets:
        if n <= b:
            return b
    return buckets[-1]


# --------------------------------------------------------------------------- #
# DP-optimal length bucketing
# --------------------------------------------------------------------------- #
def optimal_buckets(counts: Sequence[int], num_buckets: int,
                    max_len: int = MAX_PACKETS_PER_FLOW) -> List[int]:
    """Choose ``num_buckets`` upper-edges minimising total padded packet-tokens.

    Padding cost with edge assignment = sum over flows of (edge_of_flow - n).
    Since sum(n) is constant, we minimise sum(edge_of_flow), solved exactly by a
    DP over contiguous length partitions. Returns sorted edges ending at max_len.
    """
    hist = np.zeros(max_len + 1, dtype=np.int64)
    for n in counts:
        hist[min(int(n), max_len)] += 1
    # cnt[a..b] and cost of assigning all lengths in (a, b] to edge b = b * items
    prefix = np.cumsum(hist)  # prefix[j] = #flows with length <= j

    def items(a: int, b: int) -> int:  # flows with a < length <= b
        return int(prefix[b] - prefix[a])

    K = max(1, min(num_buckets, max_len))
    INF = float("inf")
    # dp[k][e] = min cost covering lengths 1..e with k buckets, last edge == e
    dp = [[INF] * (max_len + 1) for _ in range(K + 1)]
    choice = [[0] * (max_len + 1) for _ in range(K + 1)]
    for e in range(1, max_len + 1):
        dp[1][e] = e * items(0, e)
    for k in range(2, K + 1):
        for e in range(k, max_len + 1):
            for prev in range(k - 1, e):
                c = dp[k - 1][prev] + e * items(prev, e)
                if c < dp[k][e]:
                    dp[k][e] = c
                    choice[k][e] = prev
    # backtrack from dp[K][max_len]
    edges: List[int] = []
    k, e = K, max_len
    while k > 0:
        edges.append(e)
        e = choice[k][e]
        k -= 1
    edges = sorted(set(edges))
    if edges[-1] != max_len:
        edges.append(max_len)
    return edges


# --------------------------------------------------------------------------- #
# Packing strategies -> lists of (indices, bucket_length)
# --------------------------------------------------------------------------- #
def pack_bucketed(examples: Sequence[Example], cfg: PackingConfig,
                  buckets: Sequence[int]) -> List[Tuple[List[int], int]]:
    """Length bucketing: assign each flow to smallest bucket >= n, then split
    each bucket into fixed-cardinality batches. Bounded #shapes = len(buckets)."""
    rng = np.random.default_rng(cfg.seed)
    order = np.arange(len(examples))
    if cfg.shuffle:
        rng.shuffle(order)
    groups = {b: [] for b in buckets}
    for i in order:
        groups[bucket_for(real_packet_count(examples[i][0]), buckets)].append(int(i))
    out: List[Tuple[List[int], int]] = []
    for b, idxs in groups.items():
        for s in range(0, len(idxs), cfg.max_batch):
            out.append((idxs[s:s + cfg.max_batch], b))
    return out


def pack_ffd(examples: Sequence[Example], cfg: PackingConfig,
             buckets: Sequence[int]) -> List[Tuple[List[int], int]]:
    """First-Fit-Decreasing bin packing under a token budget + cardinality cap.

    Bins are batches; a flow of real length n consumes n from the bin's budget.
    Each finished bin is padded up to the smallest bucket >= its max length."""
    budget = cfg.token_budget or (cfg.max_batch * buckets[-1])
    ns = counts_of(examples)
    order = np.argsort(-ns)  # decreasing (FFD)
    bins: List[dict] = []    # {"idx": [...], "load": int, "maxn": int}
    for i in order:
        n = int(ns[i])
        placed = False
        for bn in bins:
            if bn["load"] + n <= budget and len(bn["idx"]) < cfg.max_batch:
                bn["idx"].append(int(i))
                bn["load"] += n
                bn["maxn"] = max(bn["maxn"], n)
                placed = True
                break
        if not placed:
            bins.append({"idx": [int(i)], "load": n, "maxn": n})
    return [(bn["idx"], bucket_for(bn["maxn"], buckets)) for bn in bins]


# --------------------------------------------------------------------------- #
# Batcher
# --------------------------------------------------------------------------- #
class BinPackedBatcher:
    """Iterable of padding-minimised ``Batch`` objects (trimmed to [B, T_b, L])."""

    def __init__(self, examples: Sequence[Example], cfg: PackingConfig):
        self.examples = examples
        self.cfg = cfg
        if cfg.buckets is not None:
            self.buckets = sorted(cfg.buckets)
        else:
            self.buckets = optimal_buckets(counts_of(examples), cfg.num_buckets, cfg.max_packets)
        packer = pack_ffd if cfg.strategy == "ffd" else pack_bucketed
        self.plan = packer(examples, cfg, self.buckets)

    def __len__(self) -> int:
        return len(self.plan)

    def __iter__(self) -> Iterator[Batch]:
        for idxs, Tb in self.plan:
            payload = np.stack([self.examples[i][0][:Tb] for i in idxs]).astype(np.int32)
            headers = np.stack([self.examples[i][1] for i in idxs]).astype(np.int32)
            protos = np.asarray([self.examples[i][2] for i in idxs], dtype=np.int32)
            labels = np.asarray([self.examples[i][3] for i in idxs], dtype=np.int32)
            yield Batch(payload, headers, protos, labels)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def padding_report(examples: Sequence[Example], cfg: PackingConfig) -> dict:
    """Compare packed vs naive (pad-to-max_packets) padded packet-token totals."""
    batcher = BinPackedBatcher(examples, cfg)
    real_tokens = int(counts_of(examples).sum())
    packed_padded = sum(len(idxs) * Tb for idxs, Tb in batcher.plan)
    naive_padded = len(examples) * cfg.max_packets
    return {
        "n_examples": len(examples),
        "buckets": batcher.buckets,
        "strategy": cfg.strategy,
        "n_batches": len(batcher.plan),
        "distinct_shapes": len({Tb for _, Tb in batcher.plan}),
        "real_packet_tokens": real_tokens,
        "naive_padded_tokens": naive_padded,
        "packed_padded_tokens": packed_padded,
        "naive_efficiency": round(real_tokens / max(naive_padded, 1), 4),
        "packed_efficiency": round(real_tokens / max(packed_padded, 1), 4),
        "padding_reduction": round(1.0 - packed_padded / max(naive_padded, 1), 4),
    }


# --------------------------------------------------------------------------- #
# CLI: measure the padding win on a real dataset in one line
#   python -m optimization.batching --csv UNSW_NB15_training-set.csv
# --------------------------------------------------------------------------- #
def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Report bin-packing padding reduction on a labelled flow CSV.")
    ap.add_argument("--csv", required=True, help="path to a UNSW-NB15-style CSV")
    ap.add_argument("--num-buckets", type=int, default=4)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--token-budget", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap rows for a quick estimate")
    ap.add_argument("--proto-col", default="proto")
    ap.add_argument("--sport-col", default="sport")
    ap.add_argument("--dport-col", default="dsport")
    ap.add_argument("--label-col", default="label")
    args = ap.parse_args()

    from data.tokenizer import load_unsw_nb15
    train, _, meta = load_unsw_nb15(
        args.csv, None, label_col=args.label_col, proto_col=args.proto_col,
        sport_col=args.sport_col, dport_col=args.dport_col, max_rows=args.max_rows)

    print(f"loaded {len(train)} flows | proto_vocab={meta['proto_vocab']}")
    hist = np.bincount(np.clip(counts_of(train), 0, MAX_PACKETS_PER_FLOW),
                       minlength=MAX_PACKETS_PER_FLOW + 1)
    print("real packet-count histogram (1..16):",
          " ".join(f"{i}:{int(hist[i])}" for i in range(1, MAX_PACKETS_PER_FLOW + 1) if hist[i]))
    print()
    for strat in ("bucket", "ffd"):
        cfg = PackingConfig(strategy=strat, num_buckets=args.num_buckets,
                            max_batch=args.max_batch, token_budget=args.token_budget)
        r = padding_report(train, cfg)
        print(f"[{strat:6s}] buckets={r['buckets']} | batches={r['n_batches']} "
              f"| JIT shapes={r['distinct_shapes']} | naive_eff={r['naive_efficiency']} "
              f"| packed_eff={r['packed_efficiency']} "
              f"| PADDING REDUCED {r['padding_reduction']*100:.1f}%")


if __name__ == "__main__":
    _main()
