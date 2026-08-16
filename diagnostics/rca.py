"""Root-cause analysis: turn a model anomaly score + structural findings +
flow-level statistics into a single, explainable diagnosis.

This is a transparent rule engine on purpose. A learned root-cause classifier
would need a labelled root-cause dataset that does not exist for this
project; a rule engine gives an inspectable "reasoning path" (every fired
rule is recorded) which is both more trustworthy for a v1 diagnostic tool and
easy to extend or replace with a learned component later without changing
the interface (``analyze`` -> :class:`RootCause`).
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from typing import Dict, List, Optional

from diagnostics.mismatch import Finding, PacketRecord

CODE_TO_CAUSE = {
    "ip_checksum_mismatch": "packet_corruption_or_tampering",
    "l4_checksum_mismatch": "packet_corruption_or_tampering",
    "tcp_syn_fin_violation": "scanner_or_evasion_tooling",
    "tcp_syn_rst_violation": "scanner_or_evasion_tooling",
    "tcp_syn_with_payload": "protocol_state_violation",
    "dns_payload_anomalous": "dns_tunneling_or_amplification",
    "port_protocol_mismatch": "protocol_payload_mismatch",
    "ttl_inconsistency": "possible_source_spoofing",
}

CAUSE_LABELS = {
    "packet_corruption_or_tampering": "Packet corruption or on-path tampering",
    "scanner_or_evasion_tooling": "Scanner / evasion tooling (illegal TCP flag combinations)",
    "protocol_state_violation": "TCP protocol state-machine violation",
    "dns_tunneling_or_amplification": "Possible DNS tunneling or amplification abuse",
    "protocol_payload_mismatch": "Payload does not match the protocol expected on this port",
    "possible_source_spoofing": "Possible source-address spoofing / asymmetric routing",
    "port_scan_reconnaissance": "Port-scan reconnaissance (many destination ports, one source)",
    "syn_flood_dos": "SYN flood / connection exhaustion (many source ports, incomplete handshakes)",
    "model_flagged_anomaly": "Model flagged this flow as anomalous; no specific structural cause matched",
    "benign": "No anomaly indicators found",
}


@dataclasses.dataclass
class RootCause:
    tag: str
    label: str
    confidence: float                # 0..1, heuristic
    reasoning: List[str]             # the rules that fired, in order
    contributing_findings: List[str]  # Finding.code values that contributed

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _flow_stats(records: List[PacketRecord]) -> Dict:
    if not records:
        return {}
    src_ports = {r.sport for r in records}
    dst_ports = {r.dport for r in records}
    syn_only = sum(1 for r in records if r.proto == 6 and set(r.flags) == {"S"})
    with_ack = sum(1 for r in records if r.proto == 6 and "A" in r.flags)
    return {
        "n_packets": len(records),
        "n_distinct_src_ports": len(src_ports),
        "n_distinct_dst_ports": len(dst_ports),
        "syn_only_count": syn_only,
        "ack_count": with_ack,
        "syn_only_ratio": syn_only / len(records) if records else 0.0,
    }


def analyze(
    anomaly_score: float,
    findings: List[Finding],
    records: Optional[List[PacketRecord]] = None,
    *,
    src_context: Optional[Dict] = None,
    dst_context: Optional[Dict] = None,
    anomaly_threshold: float = 0.5,
    port_scan_min_ports: int = 8,
    syn_flood_min_packets: int = 10,
    syn_flood_min_sources: int = 8,
    syn_flood_min_ratio: float = 0.8,
) -> RootCause:
    """Combine the model's anomaly score with structural findings and flow
    statistics into one explainable root-cause verdict.

    ``src_context`` / ``dst_context`` are the *cross-flow* aggregates from
    :func:`diagnostics.mismatch.aggregate_by_src` / ``aggregate_by_dst`` for
    this flow's source IP / (destination IP, destination port). A single
    5-tuple flow can never show a port scan or a distributed SYN flood by
    itself (ports are fixed within a flow) — those patterns only appear once
    you look across flows, which is what these two dicts capture. ``records``
    (this flow's own packets) is still used for structural/flag checks and
    remains supported standalone for direct rule testing.
    """
    records = records or []
    reasoning: List[str] = []

    # --- Cross-flow rules (reconnaissance / flooding patterns) --------------
    if src_context and src_context["n_distinct_dst_ports"] >= port_scan_min_ports:
        reasoning.append(
            f"Source {(records[0].src if records else 'this host')} touched "
            f"{src_context['n_distinct_dst_ports']} distinct destination ports "
            f"across the capture (>= {port_scan_min_ports}) => reconnaissance / "
            f"port-scan pattern.")
        return RootCause(
            tag="port_scan_reconnaissance", label=CAUSE_LABELS["port_scan_reconnaissance"],
            confidence=min(0.95, 0.5 + 0.05 * src_context["n_distinct_dst_ports"]),
            reasoning=reasoning, contributing_findings=[f.code for f in findings])

    if (dst_context and dst_context["n_distinct_sources"] >= syn_flood_min_sources
            and dst_context["n_packets"] >= syn_flood_min_packets
            and dst_context["syn_only_ratio"] >= syn_flood_min_ratio):
        reasoning.append(
            f"Target received {dst_context['n_packets']} packets from "
            f"{dst_context['n_distinct_sources']} distinct source (ip, port) pairs, "
            f"{dst_context['syn_only_ratio']:.0%} bare-SYN with no completed "
            f"handshake => SYN-flood / connection-exhaustion pattern.")
        return RootCause(
            tag="syn_flood_dos", label=CAUSE_LABELS["syn_flood_dos"],
            confidence=min(0.95, 0.6 + 0.3 * dst_context["syn_only_ratio"]),
            reasoning=reasoning, contributing_findings=[f.code for f in findings])

    # --- Single-flow shape rules (kept for flows/tests with multi-port
    # records directly supplied, and as a second line of defense) -----------
    stats = _flow_stats(records)
    if stats:
        if stats["n_distinct_dst_ports"] >= port_scan_min_ports and stats["n_packets"] >= port_scan_min_ports:
            reasoning.append(
                f"{stats['n_distinct_dst_ports']} distinct destination ports probed "
                f"from a single source within one flow window (>= {port_scan_min_ports}) "
                f"=> reconnaissance / port-scan pattern.")
            return RootCause(
                tag="port_scan_reconnaissance", label=CAUSE_LABELS["port_scan_reconnaissance"],
                confidence=min(0.95, 0.5 + 0.05 * stats["n_distinct_dst_ports"]),
                reasoning=reasoning, contributing_findings=[f.code for f in findings])

        if (stats["n_packets"] >= syn_flood_min_packets
                and stats["n_distinct_src_ports"] >= syn_flood_min_packets * 0.8
                and stats["syn_only_ratio"] >= syn_flood_min_ratio):
            reasoning.append(
                f"{stats['n_packets']} packets, {stats['n_distinct_src_ports']} distinct "
                f"source ports, {stats['syn_only_ratio']:.0%} bare-SYN with no completed "
                f"handshake => SYN-flood / connection-exhaustion pattern.")
            return RootCause(
                tag="syn_flood_dos", label=CAUSE_LABELS["syn_flood_dos"],
                confidence=min(0.95, 0.6 + 0.3 * stats["syn_only_ratio"]),
                reasoning=reasoning, contributing_findings=[f.code for f in findings])

    # --- Structural-finding rules, ranked by severity ------------------------
    severity_rank = {"high": 3, "medium": 2, "low": 1, "info": 0}
    if findings:
        by_cause = Counter(CODE_TO_CAUSE.get(f.code, "model_flagged_anomaly") for f in findings)
        # pick the cause with the highest-severity supporting finding, tie-break by frequency
        best_cause, best_key = None, (-1, -1)
        for f in findings:
            cause = CODE_TO_CAUSE.get(f.code)
            if cause is None:
                continue
            key = (severity_rank.get(f.severity, 0), by_cause[cause])
            if key > best_key:
                best_key, best_cause = key, cause
        if best_cause:
            supporting = [f for f in findings if CODE_TO_CAUSE.get(f.code) == best_cause]
            reasoning.append(
                f"{len(supporting)} structural finding(s) map to "
                f"'{CAUSE_LABELS[best_cause]}': " + "; ".join(f.message for f in supporting[:3]))
            confidence = min(0.97, 0.5 + 0.12 * len(supporting) + 0.1 * best_key[0])
            return RootCause(
                tag=best_cause, label=CAUSE_LABELS[best_cause], confidence=confidence,
                reasoning=reasoning, contributing_findings=[f.code for f in supporting])

    # --- Fall back to the model's own verdict --------------------------------
    if anomaly_score >= anomaly_threshold:
        reasoning.append(
            f"No structural rule matched, but the Protogrok model's P(anomaly) = "
            f"{anomaly_score:.3f} >= threshold {anomaly_threshold:.2f}; flagging for "
            f"manual review — this is the model's learned judgement, not a rule.")
        return RootCause(
            tag="model_flagged_anomaly", label=CAUSE_LABELS["model_flagged_anomaly"],
            confidence=float(anomaly_score), reasoning=reasoning,
            contributing_findings=[])

    reasoning.append(
        f"No structural findings and P(anomaly) = {anomaly_score:.3f} is below "
        f"threshold {anomaly_threshold:.2f}.")
    return RootCause(
        tag="benign", label=CAUSE_LABELS["benign"], confidence=1.0 - float(anomaly_score),
        reasoning=reasoning, contributing_findings=[])
