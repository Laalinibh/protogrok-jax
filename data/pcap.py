"""PCAP -> flow tensors -> anomaly scores (JAX inference).

Faithful port of the notebook's ``evaluate_pcap``: parse a .pcap into 5-tuple
flows, tokenize real payload bytes, and score each flow with the JAX model.
Requires ``scapy`` (``pip install scapy``); imported lazily so the rest of the
package has no hard dependency on it.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from models.config import PAD_ID, PAYLOAD_MAX_LEN, ProtogrokConfig
from data.tokenizer import Batch, payload_to_token_ids, port_bucket
from models.inference import anomaly_scores


def parse_pcap(pcap_path: str, max_packets: int) -> Tuple[Dict[str, List[bytes]], Dict[str, dict]]:
    """Group packets into flows keyed by 5-tuple; return payloads + metadata."""
    from scapy.all import IP, TCP, UDP, Raw, rdpcap  # lazy import

    packets = rdpcap(pcap_path)
    flows: Dict[str, List[bytes]] = defaultdict(list)
    meta: Dict[str, dict] = {}
    for pkt in packets:
        if IP not in pkt:
            continue
        src, dst, proto = pkt[IP].src, pkt[IP].dst, pkt[IP].proto
        sport = dport = 0
        payload = b""
        if TCP in pkt:
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
            if Raw in pkt:
                payload = bytes(pkt[TCP].payload)
        elif UDP in pkt:
            sport, dport = pkt[UDP].sport, pkt[UDP].dport
            if Raw in pkt:
                payload = bytes(pkt[UDP].payload)
        elif Raw in pkt:
            payload = bytes(pkt[IP].payload)
        key = f"{src}:{sport} -> {dst}:{dport} (proto {proto})"
        flows[key].append(payload)
        meta.setdefault(key, {"sport": sport, "dport": dport, "proto": proto})
    return flows, meta


def flows_to_batch(flows: Dict[str, List[bytes]], meta: Dict[str, dict],
                   cfg: ProtogrokConfig) -> Tuple[Batch, List[str]]:
    keys = list(flows.keys())
    payloads = np.full((len(keys), cfg.max_packets, cfg.payload_max_len), PAD_ID, np.int32)
    headers = np.zeros((len(keys), 5), np.int32)
    protos = np.zeros((len(keys),), np.int32)
    for i, k in enumerate(keys):
        pkts = flows[k][:cfg.max_packets]
        for t, pl in enumerate(pkts):
            payloads[i, t] = np.asarray(payload_to_token_ids(pl, cfg.payload_max_len), np.int32)
        m = meta[k]
        pid = int(m["proto"]) % cfg.proto_vocab
        headers[i] = [pid, port_bucket(m["sport"]), port_bucket(m["dport"]), len(pkts), 0]
        protos[i] = pid
    labels = np.zeros((len(keys),), np.int32)
    return Batch(payloads, headers, protos, labels), keys


def evaluate_pcap(pcap_path: str, cfg: ProtogrokConfig, params,
                  threshold: float = 0.5) -> List[dict]:
    """Return per-flow anomaly results for a PCAP file."""
    flows, meta = parse_pcap(pcap_path, cfg.max_packets)
    if not flows:
        return []
    batch, keys = flows_to_batch(flows, meta, cfg)
    scores = anomaly_scores(cfg, params, batch)
    return [{"flow": k, "score": float(s), "anomaly": bool(s > threshold),
             "packets": len(flows[k])} for k, s in zip(keys, scores)]
