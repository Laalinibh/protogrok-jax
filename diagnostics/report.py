"""End-to-end diagnostic report: anomaly score + mismatch findings +
root-cause + suggested fixes, per flow.

This is the module both ``eval.py --diagnose`` and ``handler.py`` (the
Hugging Face inference handler) call.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from models.config import ProtogrokConfig
from diagnostics.mismatch import (
    inspect_flow, parse_pcap_detailed, aggregate_by_src, aggregate_by_dst, PacketRecord,
)
from diagnostics.rca import analyze
from diagnostics.remediation import suggest_actions


def diagnose_flow(
    flow_key: str,
    anomaly_score: float,
    records: List[PacketRecord],
    *,
    n_packets: Optional[int] = None,
    threshold: float = 0.5,
    src_context: Optional[Dict] = None,
    dst_context: Optional[Dict] = None,
) -> Dict:
    """Build the full diagnostic report for a single already-scored flow.

    ``src_context`` / ``dst_context`` are optional cross-flow aggregates (see
    :func:`diagnostics.rca.analyze`); pass them when scoring a whole capture
    via :func:`diagnose_pcap` so scan/flood patterns spanning multiple flows
    are detected. Omit them to analyze a flow in isolation.
    """
    findings = inspect_flow(records)
    cause = analyze(anomaly_score, findings, records, src_context=src_context,
                    dst_context=dst_context, anomaly_threshold=threshold)
    actions = suggest_actions(cause)
    return {
        "flow": flow_key,
        "packets": n_packets if n_packets is not None else len(records),
        "anomaly_score": float(anomaly_score),
        "anomaly": bool(anomaly_score > threshold),
        "mismatch_findings": [f.to_dict() for f in findings],
        "mismatch_detected": len(findings) > 0,
        "root_cause": cause.to_dict(),
        "suggested_actions": actions,
    }


def diagnose_pcap(
    pcap_path: str,
    cfg: ProtogrokConfig,
    params,
    *,
    threshold: float = 0.5,
) -> List[Dict]:
    """Full pipeline: parse -> score with the Protogrok model -> structural
    mismatch inspection -> root-cause analysis -> remediation suggestions,
    one report per network flow found in the capture.
    """
    from data.pcap import evaluate_pcap  # neural anomaly scoring path

    anomaly_results = evaluate_pcap(pcap_path, cfg, params, threshold=threshold)
    if not anomaly_results:
        return []

    detailed = parse_pcap_detailed(pcap_path, cfg.max_packets)
    by_src = aggregate_by_src(detailed)
    by_dst = aggregate_by_dst(detailed)

    reports = []
    for r in anomaly_results:
        records = detailed.get(r["flow"], [])
        src_ctx = by_src.get(records[0].src) if records else None
        dst_ctx = by_dst.get((records[0].dst, records[0].dport)) if records else None
        reports.append(diagnose_flow(
            r["flow"], r["score"], records, n_packets=r["packets"], threshold=threshold,
            src_context=src_ctx, dst_context=dst_ctx))
    return reports
