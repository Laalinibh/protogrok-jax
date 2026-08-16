"""Tests for the real packet-byte pipeline (data/packets.py).

Builds a small parquet file in the exact Payload-Bytes schema, so the whole
path is exercised without touching the network or the 2.7 GB source files.
"""
from __future__ import annotations

import numpy as np
import pytest

from models.config import BYTE_OFFSET, PAD_ID
from data.packets import (
    ATTACK_CLASSES, FlowShard, iterate_flow_batches, load_parquet_flows,
    payload_columns, proto_to_number, tokenize, train_val_split,
)
from data.tokenizer import payload_to_token_ids

pq = pytest.importorskip("pyarrow.parquet")
pa = pytest.importorskip("pyarrow")

L, T = 128, 16


def make_parquet(path, n_flows=40, packets_per_flow=5, seed=0):
    """Write a synthetic file with the real Payload-Bytes schema.

    Attack flows carry a fixed byte signature so a model can actually learn
    something -- that makes the fixture usable as a learning smoke test too.
    """
    rng = np.random.default_rng(seed)
    rows = {c: [] for c in
            ["flow_id", "packet_id", "source_port", "destination_port",
             "protocol", "payload_length", "attack_label"]}
    payload = []
    pid = 0
    for f in range(n_flows):
        is_attack = f % 2 == 1
        for _ in range(packets_per_flow):
            plen = int(rng.integers(8, L))
            body = rng.integers(0, 256, L, dtype=np.uint8)
            if is_attack:
                body[:4] = [0xDE, 0xAD, 0xBE, 0xEF]
            body[plen:] = 0
            payload.append(body)
            rows["flow_id"].append(f)
            rows["packet_id"].append(pid)
            rows["source_port"].append(int(rng.integers(1024, 65535)))
            rows["destination_port"].append(80 if not is_attack else 4444)
            rows["protocol"].append("tcp")
            rows["payload_length"].append(plen)
            rows["attack_label"].append("exploits" if is_attack else "normal")
            pid += 1
    table = {k: pa.array(v) for k, v in rows.items()}
    arr = np.stack(payload)
    for i, name in enumerate(payload_columns(L)):
        table[name] = pa.array(arr[:, i].astype(np.int64))
    pq.write_table(pa.table(table), path)
    return arr


def test_tokenize_matches_inference_tokenizer():
    """tokenize() must be byte-identical to the path data/pcap.py uses."""
    rng = np.random.default_rng(0)
    for plen in (0, 1, 7, 64, L):
        raw = rng.integers(0, 256, L, dtype=np.uint8)
        raw[plen:] = 0
        got = tokenize(raw[None, None, :], np.array([[plen]]))[0, 0]
        want = np.array(payload_to_token_ids(bytes(raw[:plen]), L), np.int32)
        assert np.array_equal(got, want), f"mismatch at payload_length={plen}"


def test_zero_byte_is_not_padding():
    """A real 0x00 byte must tokenize to BYTE_OFFSET, not PAD_ID."""
    raw = np.zeros((1, 1, L), np.uint8)
    toks = tokenize(raw, np.array([[4]]))[0, 0]
    assert list(toks[:4]) == [BYTE_OFFSET] * 4
    assert list(toks[4:8]) == [PAD_ID] * 4


def test_proto_numbering_matches_pcap_inference():
    assert proto_to_number("tcp") == 6
    assert proto_to_number("udp") == 17
    assert proto_to_number(6) == 6      # already numeric
    assert proto_to_number("6") == 6    # numeric string


def test_load_parquet_flows(tmp_path):
    path = tmp_path / "p.parquet"
    make_parquet(path, n_flows=40, packets_per_flow=5)
    shard = load_parquet_flows(str(path), max_packets=T, payload_max_len=L)

    assert len(shard) == 40
    assert shard.payload.shape == (40, T, L)
    assert shard.lengths.shape == (40, T)
    assert shard.label_bin.mean() == pytest.approx(0.5)
    # 5 packets per flow, so slots 5..15 stay empty
    assert (shard.headers[:, 3] == 5).all()
    assert (shard.lengths[:, 5:] == 0).all()
    # protocol "tcp" -> 6, ports bucketed, class label resolved
    assert (shard.proto_id == 6).all()
    assert set(np.unique(shard.label_cls)) == {0, ATTACK_CLASSES.index("exploits")}


def test_flow_truncation_keeps_first_packets(tmp_path):
    """A flow longer than T keeps its first T packets, in packet_id order."""
    path = tmp_path / "long.parquet"
    make_parquet(path, n_flows=4, packets_per_flow=T + 9)
    shard = load_parquet_flows(str(path), max_packets=T, payload_max_len=L)
    assert (shard.headers[:, 3] == T).all()
    assert (shard.lengths > 0).all()


def test_batches_and_split(tmp_path):
    path = tmp_path / "p.parquet"
    make_parquet(path, n_flows=64, packets_per_flow=4)
    shard = load_parquet_flows(str(path), max_packets=T, payload_max_len=L)
    tr, va = train_val_split(shard, val_fraction=0.25, seed=0)
    assert len(tr) + len(va) == len(shard) and len(va) == 16

    batches = list(iterate_flow_batches(tr, 8, task="anomaly", seed=0))
    assert len(batches) == 6
    b = batches[0]
    assert b.payload.shape == (8, T, L) and b.payload.dtype == np.int32
    assert b.payload.max() <= 258 and b.payload.min() >= 0
    assert set(np.unique(b.labels)) <= {0, 1}

    cls = next(iter(iterate_flow_batches(tr, 8, task="class", seed=0)))
    assert set(np.unique(cls.labels)) <= set(range(len(ATTACK_CLASSES)))


def test_shard_roundtrip(tmp_path):
    path = tmp_path / "p.parquet"
    make_parquet(path, n_flows=16, packets_per_flow=3)
    shard = load_parquet_flows(str(path), max_packets=T, payload_max_len=L)
    shard.save(str(tmp_path / "shard"))
    back = FlowShard.load(str(tmp_path / "shard"), mmap=True)
    assert len(back) == len(shard)
    assert np.array_equal(np.asarray(back.payload), shard.payload)
    assert np.array_equal(np.asarray(back.lengths), shard.lengths)
