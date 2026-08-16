"""Tests for the diagnostics layer: mismatch detection, root-cause analysis,
remediation suggestions, and the end-to-end pcap report."""
import jax
import pytest

scapy = pytest.importorskip("scapy.all", reason="diagnostics tests need scapy")
from scapy.all import IP, TCP, UDP, Raw, wrpcap  # noqa: E402

from diagnostics.mismatch import (
    inspect_flow, parse_pcap_detailed, aggregate_by_src, aggregate_by_dst, PacketRecord,
)
from diagnostics.rca import analyze
from diagnostics.remediation import suggest_actions
from diagnostics.report import diagnose_pcap, diagnose_flow
from models.config import ProtogrokConfig
from models.protogrok import init_params
from optimization import checkpointing


def _tiny_cfg(**overrides):
    """A tiny config so tests exercise the real forward pass without the cost
    of a full 124M/300M model."""
    base = dict(d_model=32, payload_dim=16, header_dim=16, packet_layers=1,
                session_layers=1, nhead=2, mlp_ratio=2, memory_slots=2,
                num_classes=4, proto_vocab=132)
    base.update(overrides)
    return ProtogrokConfig(**base)


# --------------------------------------------------------------------------- #
# Mismatch detection
# --------------------------------------------------------------------------- #
def test_ip_checksum_mismatch_detected(tmp_path):
    pkt = IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1234, dport=80, flags="S")
    pkt[IP].chksum = 0x1234  # deliberately wrong, and fixed so scapy won't recompute
    pcap = tmp_path / "bad_checksum.pcap"
    wrpcap(str(pcap), [pkt])

    flows = parse_pcap_detailed(str(pcap), max_packets=16)
    assert len(flows) == 1
    records = next(iter(flows.values()))
    findings = inspect_flow(records)
    codes = {f.code for f in findings}
    assert "ip_checksum_mismatch" in codes


def test_tcp_syn_fin_violation_detected(tmp_path):
    pkt = IP(src="10.0.0.1", dst="10.0.0.2") / TCP(sport=1234, dport=80, flags="SF")
    pcap = tmp_path / "syn_fin.pcap"
    wrpcap(str(pcap), [pkt])

    flows = parse_pcap_detailed(str(pcap), max_packets=16)
    records = next(iter(flows.values()))
    findings = inspect_flow(records)
    codes = {f.code for f in findings}
    assert "tcp_syn_fin_violation" in codes


def test_dns_oversized_payload_flagged(tmp_path):
    pkt = IP(src="10.0.0.1", dst="8.8.8.8") / UDP(sport=33445, dport=53) / Raw(load=b"A" * 500)
    pcap = tmp_path / "dns_oversize.pcap"
    wrpcap(str(pcap), [pkt])

    flows = parse_pcap_detailed(str(pcap), max_packets=16)
    records = next(iter(flows.values()))
    findings = inspect_flow(records)
    codes = {f.code for f in findings}
    assert "dns_payload_anomalous" in codes


def test_benign_http_flow_has_no_findings(tmp_path):
    req = IP(src="10.0.0.5", dst="10.0.0.10") / TCP(sport=54321, dport=80, flags="PA") / \
        Raw(load=b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
    pcap = tmp_path / "benign.pcap"
    wrpcap(str(pcap), [req])

    flows = parse_pcap_detailed(str(pcap), max_packets=16)
    records = next(iter(flows.values()))
    findings = inspect_flow(records)
    assert findings == []


def test_aggregate_by_src_and_dst_across_flows(tmp_path):
    scan = [IP(src="10.0.0.66", dst="10.0.0.10") / TCP(sport=40000 + i, dport=20 + i, flags="S")
            for i in range(10)]
    pcap = tmp_path / "scan.pcap"
    wrpcap(str(pcap), scan)

    detailed = parse_pcap_detailed(str(pcap), max_packets=16)
    assert len(detailed) == 10  # each probe is its own 5-tuple flow

    by_src = aggregate_by_src(detailed)
    assert by_src["10.0.0.66"]["n_distinct_dst_ports"] == 10

    by_dst = aggregate_by_dst(detailed)
    # each flood-style destination port only saw 1 source here (it's a scan,
    # not a flood) -- confirms the two aggregations measure different things.
    assert all(v["n_distinct_sources"] == 1 for v in by_dst.values())


# --------------------------------------------------------------------------- #
# Root-cause analysis
# --------------------------------------------------------------------------- #
def _syn_record(i, dport, sport=40000):
    return PacketRecord(index=i, src="10.0.0.66", dst="10.0.0.10", sport=sport,
                        dport=dport, proto=6, flags="S", ttl=64, payload=b"",
                        ip_checksum_ok=True, l4_checksum_ok=True)


def test_rca_detects_port_scan():
    records = [_syn_record(i, dport=20 + i) for i in range(12)]
    cause = analyze(anomaly_score=0.9, findings=[], records=records)
    assert cause.tag == "port_scan_reconnaissance"
    assert cause.reasoning  # explainable: at least one reason recorded


def test_rca_detects_syn_flood():
    records = [_syn_record(i, dport=22, sport=40000 + i) for i in range(15)]
    cause = analyze(anomaly_score=0.9, findings=[], records=records)
    assert cause.tag == "syn_flood_dos"


def test_rca_benign_when_no_signal():
    cause = analyze(anomaly_score=0.1, findings=[], records=[])
    assert cause.tag == "benign"


def test_rca_falls_back_to_model_when_no_rule_fires():
    cause = analyze(anomaly_score=0.95, findings=[], records=[])
    assert cause.tag == "model_flagged_anomaly"
    assert cause.confidence == pytest.approx(0.95)


# --------------------------------------------------------------------------- #
# Remediation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tag", [
    "port_scan_reconnaissance", "syn_flood_dos", "packet_corruption_or_tampering",
    "scanner_or_evasion_tooling", "protocol_state_violation",
    "dns_tunneling_or_amplification", "protocol_payload_mismatch",
    "possible_source_spoofing", "model_flagged_anomaly", "benign",
])
def test_every_root_cause_has_suggested_actions(tag):
    from diagnostics.rca import RootCause
    cause = RootCause(tag=tag, label=tag, confidence=0.5, reasoning=[], contributing_findings=[])
    actions = suggest_actions(cause)
    assert isinstance(actions, list) and len(actions) >= 1


# --------------------------------------------------------------------------- #
# End-to-end report (real forward pass through a tiny model)
# --------------------------------------------------------------------------- #
def test_diagnose_pcap_end_to_end(tmp_path):
    cfg = _tiny_cfg()
    model, variables = init_params(cfg, jax.random.PRNGKey(0))
    ckpt_dir = tmp_path / "ckpt"
    checkpointing.save(str(ckpt_dir), variables["params"], cfg, step=0)
    params, restored_cfg = checkpointing.restore(str(ckpt_dir), params_template=variables["params"])

    benign = IP(src="10.0.0.5", dst="10.0.0.10") / TCP(sport=54321, dport=80, flags="PA") / \
        Raw(load=b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
    scan = [IP(src="10.0.0.66", dst="10.0.0.10") / TCP(sport=40000 + i, dport=20 + i, flags="S")
            for i in range(10)]
    pcap = tmp_path / "trace.pcap"
    wrpcap(str(pcap), [benign] + scan)

    reports = diagnose_pcap(str(pcap), restored_cfg, params)
    # 1 benign flow + 10 scan-probe flows (each probe is its own 5-tuple flow
    # since the destination port differs on each one -- that's exactly why
    # port-scan detection needs the cross-flow src_context, not a single flow).
    assert len(reports) == 11
    for r in reports:
        assert set(r.keys()) >= {
            "flow", "packets", "anomaly_score", "anomaly", "mismatch_findings",
            "mismatch_detected", "root_cause", "suggested_actions"}
        assert isinstance(r["suggested_actions"], list) and r["suggested_actions"]

    scan_reports = [r for r in reports if "10.0.0.66" in r["flow"]]
    assert len(scan_reports) == 10
    assert all(r["root_cause"]["tag"] == "port_scan_reconnaissance" for r in scan_reports)


def test_diagnose_flow_matches_report_shape():
    cause_records = [_syn_record(i, dport=20 + i) for i in range(10)]
    report = diagnose_flow("10.0.0.66:x -> 10.0.0.10:y (proto 6)", 0.8, cause_records)
    assert report["root_cause"]["tag"] == "port_scan_reconnaissance"
    assert report["anomaly"] is True
    assert report["mismatch_detected"] in (True, False)
