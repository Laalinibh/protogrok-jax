"""Packet / protocol mismatch detection.

Where ``models.protogrok`` answers a *statistical* question ("does this flow
look like the anomalous class the network learned?"), this module answers a
*structural* one: does each packet actually respect the protocol rules it
claims to follow? It works directly on the parsed packets (via ``scapy``),
independently of the model, so its findings hold even for an untrained or
poorly-trained model.

Checks implemented
-------------------
1. **Checksum mismatch** — recompute the IP/TCP/UDP checksum and compare to
   the one on the wire (classic on-path corruption / injection / spoofing
   signal).
2. **TCP flag-state violations** — flag combinations that cannot legally
   occur in the TCP state machine (e.g. SYN+FIN, SYN carrying a payload).
3. **Port/payload signature mismatch** — the payload's byte signature does
   not match the protocol conventionally associated with the port it was
   sent on (e.g. non-HTTP bytes on port 80, an oversized/garbled response on
   port 53).
4. **TTL inconsistency** — wildly varying TTLs from what claims to be a
   single source, a classic spoofing / route-flap indicator.
"""
from __future__ import annotations

import dataclasses
import re
from collections import defaultdict
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class Finding:
    code: str            # short machine-readable tag, e.g. "checksum_mismatch"
    severity: str         # "info" | "low" | "medium" | "high"
    packet_index: int     # index within the flow (0-based), or -1 for flow-level
    message: str          # human-readable explanation
    evidence: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class PacketRecord:
    index: int
    src: str
    dst: str
    sport: int
    dport: int
    proto: int            # IP protocol number (6=TCP, 17=UDP)
    flags: str             # TCP flag string, "" for non-TCP
    ttl: int
    payload: bytes
    ip_checksum_ok: bool
    l4_checksum_ok: bool


# --------------------------------------------------------------------------- #
# PCAP -> detailed per-flow packet records
# --------------------------------------------------------------------------- #
def parse_pcap_detailed(pcap_path: str, max_packets: int) -> Dict[str, List[PacketRecord]]:
    """Group packets into the same 5-tuple flow keys as ``data.pcap.parse_pcap``,
    but retain the structural fields mismatch detection needs."""
    from scapy.all import IP, TCP, UDP, Raw, rdpcap  # lazy import

    packets = rdpcap(pcap_path)
    flows: Dict[str, List[PacketRecord]] = defaultdict(list)
    for pkt in packets:
        if IP not in pkt:
            continue
        ip = pkt[IP]
        src, dst, proto = ip.src, ip.dst, ip.proto
        sport = dport = 0
        flags = ""
        payload = b""
        l4_ok = True
        if TCP in pkt:
            tcp = pkt[TCP]
            sport, dport, flags = tcp.sport, tcp.dport, str(tcp.flags)
            if Raw in pkt:
                payload = bytes(tcp.payload)
            l4_ok = _checksum_ok(pkt, TCP)
        elif UDP in pkt:
            udp = pkt[UDP]
            sport, dport = udp.sport, udp.dport
            if Raw in pkt:
                payload = bytes(udp.payload)
            l4_ok = _checksum_ok(pkt, UDP)
        elif Raw in pkt:
            payload = bytes(ip.payload)

        key = f"{src}:{sport} -> {dst}:{dport} (proto {proto})"
        rec_list = flows[key]
        if len(rec_list) >= max_packets:
            continue
        rec_list.append(PacketRecord(
            index=len(rec_list), src=src, dst=dst, sport=sport, dport=dport,
            proto=proto, flags=flags, ttl=int(ip.ttl),
            payload=payload,
            ip_checksum_ok=_checksum_ok(pkt, IP),
            l4_checksum_ok=l4_ok,
        ))
    return dict(flows)


def aggregate_by_src(detailed: Dict[str, List["PacketRecord"]]) -> Dict[str, dict]:
    """Cross-flow view keyed by source IP.

    A single 5-tuple "flow" can never show a port scan by construction (the
    destination port is fixed within a flow), so reconnaissance/flood
    detection needs to look *across* flows. This groups every packet from
    every flow by its source IP so :mod:`diagnostics.rca` can see how many
    distinct destination ports one source touched.
    """
    by_src: Dict[str, List["PacketRecord"]] = defaultdict(list)
    for records in detailed.values():
        for r in records:
            by_src[r.src].append(r)

    out = {}
    for src, recs in by_src.items():
        dst_ports = {(r.dst, r.dport) for r in recs}
        syn_only = sum(1 for r in recs if r.proto == 6 and set(r.flags) == {"S"})
        out[src] = {
            "n_packets": len(recs),
            "n_distinct_dst_ports": len(dst_ports),
            "syn_only_ratio": syn_only / len(recs) if recs else 0.0,
        }
    return out


def aggregate_by_dst(detailed: Dict[str, List["PacketRecord"]]) -> Dict[tuple, dict]:
    """Cross-flow view keyed by (destination IP, destination port).

    Used to detect SYN floods / connection-exhaustion attacks, which show up
    as many *different* 5-tuple flows (one per spoofed/varied source port)
    converging on the same target rather than as one flow.
    """
    by_dst: Dict[tuple, List["PacketRecord"]] = defaultdict(list)
    for records in detailed.values():
        for r in records:
            by_dst[(r.dst, r.dport)].append(r)

    out = {}
    for key, recs in by_dst.items():
        src_ports = {(r.src, r.sport) for r in recs}
        syn_only = sum(1 for r in recs if r.proto == 6 and set(r.flags) == {"S"})
        out[key] = {
            "n_packets": len(recs),
            "n_distinct_sources": len(src_ports),
            "syn_only_ratio": syn_only / len(recs) if recs else 0.0,
        }
    return out


def _checksum_ok(pkt, layer_cls) -> bool:
    """Recompute ``layer_cls``'s checksum on a copy and compare to the original.

    Returns True (benefit of the doubt) if the layer is absent or scapy
    cannot recompute it (e.g. fragmented packets).
    """
    try:
        if layer_cls not in pkt:
            return True
        original = pkt[layer_cls].chksum
        clone = pkt.copy()
        del clone[layer_cls].chksum
        # rebuilding from raw bytes forces scapy to recompute every checksum
        rebuilt = clone.__class__(bytes(clone))
        return rebuilt[layer_cls].chksum == original
    except Exception:
        return True


# --------------------------------------------------------------------------- #
# Port -> expected payload signature heuristics
# --------------------------------------------------------------------------- #
_HTTP_METHODS = (b"GET ", b"POST ", b"PUT ", b"HEAD ", b"DELETE ", b"OPTIONS ", b"HTTP/1.")
# TLS record layer: ContentType (20=change_cipher_spec, 21=alert,
# 22=handshake, 23=application_data) followed by a 0x03 major version byte.
# Only checking for a Handshake record (the old, narrower check) misclassifies
# perfectly normal encrypted Application Data records -- which make up the
# bulk of any real HTTPS capture -- as a protocol mismatch.
_TLS_CONTENT_TYPES = {0x14, 0x15, 0x16, 0x17}
_SSH_BANNER = b"SSH-"


def _looks_like_http(payload: bytes) -> bool:
    return payload[:8].startswith(_HTTP_METHODS) or any(payload.startswith(m) for m in _HTTP_METHODS)


def _looks_like_tls(payload: bytes) -> bool:
    if len(payload) < 3:
        return False
    return payload[0] in _TLS_CONTENT_TYPES and payload[1] == 0x03


def _looks_like_ssh(payload: bytes) -> bool:
    return payload.startswith(_SSH_BANNER)


def _looks_like_dns(payload: bytes) -> bool:
    # A DNS message needs >= 12-byte header; QDCOUNT/ANCOUNT etc. should be
    # small (<= a few hundred) for ordinary queries/responses.
    if len(payload) < 12:
        return False
    qdcount = int.from_bytes(payload[4:6], "big")
    ancount = int.from_bytes(payload[6:8], "big")
    return qdcount <= 16 and ancount <= 64


# port -> (name, signature_fn); only ports we have a confident signature for
_PORT_SIGNATURES = {
    80: ("http", _looks_like_http),
    8080: ("http", _looks_like_http),
    443: ("tls", _looks_like_tls),
    22: ("ssh", _looks_like_ssh),
    53: ("dns", _looks_like_dns),
}

# DNS responses/queries beyond this are unusual for ordinary lookups and are
# commonly associated with amplification or tunneling.
_DNS_OVERSIZE_BYTES = 300


# --------------------------------------------------------------------------- #
# Per-flow inspection
# --------------------------------------------------------------------------- #
def inspect_flow(records: List[PacketRecord]) -> List[Finding]:
    findings: List[Finding] = []
    if not records:
        return findings

    ttls = []
    for r in records:
        # 1. checksum mismatch
        if not r.ip_checksum_ok:
            findings.append(Finding(
                code="ip_checksum_mismatch", severity="high", packet_index=r.index,
                message=f"IP header checksum does not match the recomputed value "
                        f"for packet {r.index} ({r.src} -> {r.dst}); packet is "
                        f"corrupted, was tampered with in transit, or was crafted.",
                evidence={"src": r.src, "dst": r.dst}))
        if not r.l4_checksum_ok:
            findings.append(Finding(
                code="l4_checksum_mismatch", severity="high", packet_index=r.index,
                message=f"Transport-layer (TCP/UDP) checksum mismatch on packet "
                        f"{r.index}; segment content does not match what its own "
                        f"header claims.",
                evidence={"sport": r.sport, "dport": r.dport}))

        # 2. TCP flag-state violations
        if r.proto == 6 and r.flags:
            flags = set(r.flags)
            if {"S", "F"} <= flags:
                findings.append(Finding(
                    code="tcp_syn_fin_violation", severity="high", packet_index=r.index,
                    message=f"Packet {r.index} sets both SYN and FIN — not a legal "
                            f"TCP state transition; classic scanner/evasion signature.",
                    evidence={"flags": r.flags}))
            if "S" in flags and "A" not in flags and len(r.payload) > 0:
                findings.append(Finding(
                    code="tcp_syn_with_payload", severity="medium", packet_index=r.index,
                    message=f"Packet {r.index} carries a payload on a bare SYN "
                            f"(no ACK) — a connection has not been established yet.",
                    evidence={"flags": r.flags, "payload_len": len(r.payload)}))
            if flags == {"R", "S"}:
                findings.append(Finding(
                    code="tcp_syn_rst_violation", severity="medium", packet_index=r.index,
                    message=f"Packet {r.index} sets both SYN and RST simultaneously.",
                    evidence={"flags": r.flags}))

        # 3. Port/payload signature mismatch
        sig = _PORT_SIGNATURES.get(r.dport) or _PORT_SIGNATURES.get(r.sport)
        if sig and r.payload:
            name, fn = sig
            if name == "dns":
                if len(r.payload) > _DNS_OVERSIZE_BYTES or not fn(r.payload):
                    findings.append(Finding(
                        code="dns_payload_anomalous", severity="medium", packet_index=r.index,
                        message=f"Packet {r.index} on DNS port ({r.sport}/{r.dport}) has "
                                f"a payload that doesn't look like a well-formed DNS "
                                f"message and/or is oversized ({len(r.payload)} bytes) — "
                                f"consistent with DNS tunneling or amplification abuse.",
                        evidence={"payload_len": len(r.payload)}))
            elif not fn(r.payload):
                findings.append(Finding(
                    code="port_protocol_mismatch", severity="low", packet_index=r.index,
                    message=f"Packet {r.index} on port {r.dport or r.sport} (expected "
                            f"'{name}') has a payload that doesn't match the expected "
                            f"'{name}' signature — traffic may be tunneled, "
                            f"misconfigured, or mislabeled.",
                    evidence={"expected_protocol": name, "port": r.dport or r.sport}))

        ttls.append(r.ttl)

    # 4. TTL inconsistency across the flow (possible spoofing / route flap)
    if len(ttls) >= 3 and (max(ttls) - min(ttls)) > 10:
        findings.append(Finding(
            code="ttl_inconsistency", severity="low", packet_index=-1,
            message=f"TTL varies by {max(ttls) - min(ttls)} across packets claiming "
                    f"to be the same flow (min={min(ttls)}, max={max(ttls)}) — "
                    f"possible source-address spoofing or asymmetric routing.",
            evidence={"ttl_min": min(ttls), "ttl_max": max(ttls)}))

    return findings
