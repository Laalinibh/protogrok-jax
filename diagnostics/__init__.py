"""Diagnostics layer: packet-mismatch detection, root-cause analysis, and
remediation suggestions built on top of the Protogrok anomaly-detection model.

This package is intentionally rule-based / explainable rather than learned:
Protogrok's neural model answers "is this flow anomalous?"; the modules here
answer the follow-up questions a network operator actually needs answered —
"what specifically looks wrong on the wire?", "what's the likely cause?", and
"what should I do about it?" — with a transparent, inspectable reasoning
trace for each verdict.
"""
from diagnostics.mismatch import Finding, inspect_flow
from diagnostics.rca import RootCause, analyze
from diagnostics.remediation import suggest_actions
from diagnostics.report import diagnose_pcap, diagnose_flow

__all__ = [
    "Finding", "inspect_flow",
    "RootCause", "analyze",
    "suggest_actions",
    "diagnose_pcap", "diagnose_flow",
]
