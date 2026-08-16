"""Suggested fixing actions, keyed off the :class:`diagnostics.rca.RootCause` tag.

Deliberately a plain lookup table rather than another model: the actions are
standard network-operations playbook items, and keeping them declarative
makes it trivial for an operator to edit/extend without touching model code.
"""
from __future__ import annotations

from typing import List

from diagnostics.rca import RootCause

_ACTIONS = {
    "port_scan_reconnaissance": [
        "Rate-limit or temporarily block the source IP at the edge firewall/ACL.",
        "Check whether the destination host(s) expose any of the probed ports "
        "unnecessarily and close/firewall the unused ones.",
        "Correlate the source IP against threat-intel feeds and internal asset "
        "inventory (is this an authorized scanner, e.g. a vuln-management job?).",
        "Enable/verify port-scan detection thresholds on the IDS/IPS in front of "
        "this segment.",
    ],
    "syn_flood_dos": [
        "Enable SYN cookies on the target host/load-balancer.",
        "Rate-limit new connections per source IP at the firewall or L4 load balancer.",
        "Lower the SYN-RECEIVED timeout / increase the backlog queue temporarily "
        "to absorb the burst.",
        "If sustained, engage upstream DDoS scrubbing / blackhole the offending "
        "source ranges.",
    ],
    "packet_corruption_or_tampering": [
        "Check the physical/link layer on the path (bad cable, failing NIC, "
        "duplex mismatch) — checksum errors are often hardware, not malicious.",
        "If corruption is isolated to one path, capture at both ends to localize "
        "the failing hop.",
        "If checksums are intentionally wrong (some crafted-packet tools do "
        "this), treat as a potential evasion/fuzzing attempt and inspect the "
        "source further.",
    ],
    "scanner_or_evasion_tooling": [
        "Illegal TCP flag combinations (SYN+FIN, SYN+RST) are a strong scanner/"
        "evasion signature — block the source IP and open an investigation.",
        "Verify your IDS/IPS has signatures for stealth-scan flag patterns enabled.",
        "Check the destination host's logs for any resulting connection attempts.",
    ],
    "protocol_state_violation": [
        "Inspect the sending host/application for a broken TCP implementation "
        "(payload on a bare SYN is invalid) — this can also be crafted traffic.",
        "If reproducible from a known-good client, check for a NIC offload / "
        "TCP stack bug.",
    ],
    "dns_tunneling_or_amplification": [
        "Inspect the DNS payload for encoded/exfiltrated data (tunneling tools "
        "typically use TXT/NULL records with high-entropy subdomains).",
        "Apply DNS response-size limits and enable response-rate limiting (RRL) "
        "on authoritative/recursive resolvers to blunt amplification abuse.",
        "Restrict which hosts are allowed to query external DNS directly "
        "(force internal resolvers) to make tunneling easier to detect.",
    ],
    "protocol_payload_mismatch": [
        "Confirm whether this is intentional (e.g. a service legitimately "
        "running on a non-standard port) and update asset documentation if so.",
        "If unexpected, treat as possible protocol tunneling/evasion and "
        "inspect the payload contents more closely.",
        "Consider deep-packet-inspection rules that key on payload signature "
        "rather than port number for this segment.",
    ],
    "possible_source_spoofing": [
        "Enable/verify ingress and egress filtering (BCP38/uRPF) at your "
        "network edge to drop spoofed-source traffic.",
        "Cross-check the TTL/path against known-good baselines for this source; "
        "large swings can also just mean route changes, not spoofing — confirm "
        "before blocking.",
    ],
    "model_flagged_anomaly": [
        "No specific rule fired, but the model's learned score is high — queue "
        "for manual analyst review rather than auto-remediating.",
        "If this repeats for the same flow signature, consider adding a "
        "dedicated structural rule for it in diagnostics/mismatch.py.",
        "Re-run with a model checkpoint trained on real labelled traffic "
        "(e.g. UNSW-NB15) if you are still on a smoke-test/synthetic checkpoint "
        "— untrained models over-flag indiscriminately.",
    ],
    "benign": [
        "No action needed.",
    ],
}


def suggest_actions(cause: RootCause) -> List[str]:
    return list(_ACTIONS.get(cause.tag, _ACTIONS["model_flagged_anomaly"]))
