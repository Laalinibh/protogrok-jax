"""Tests for the Hugging Face inference handler (handler.py)."""
import base64
import json

import jax
import pytest

scapy = pytest.importorskip("scapy.all", reason="handler tests need scapy")
from scapy.all import IP, TCP, Raw, wrpcap  # noqa: E402

from handler import EndpointHandler
from models.config import ProtogrokConfig
from models.protogrok import init_params
from optimization import checkpointing


def _tiny_cfg():
    return ProtogrokConfig(d_model=32, payload_dim=16, header_dim=16, packet_layers=1,
                           session_layers=1, nhead=2, mlp_ratio=2, memory_slots=2,
                           num_classes=4, proto_vocab=132)


@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory):
    cfg = _tiny_cfg()
    _, variables = init_params(cfg, jax.random.PRNGKey(0))
    d = tmp_path_factory.mktemp("handler_ckpt")
    checkpointing.save(str(d), variables["params"], cfg, step=0)
    return str(d)


@pytest.fixture(scope="module")
def pcap_bytes(tmp_path_factory):
    pkt = IP(src="10.0.0.5", dst="10.0.0.10") / TCP(sport=1234, dport=80, flags="PA") / \
        Raw(load=b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
    p = tmp_path_factory.mktemp("pcaps") / "t.pcap"
    wrpcap(str(p), [pkt])
    return p.read_bytes()


def test_handler_loads_from_checkpoint_dir(ckpt_dir):
    handler = EndpointHandler(ckpt_dir)
    assert handler.cfg.d_model == 32
    assert handler._n_params > 0


def test_handler_missing_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        EndpointHandler(str(tmp_path))


def test_handler_call_raw_bytes(ckpt_dir, pcap_bytes):
    handler = EndpointHandler(ckpt_dir)
    out = handler({"inputs": pcap_bytes})
    assert "error" not in out
    assert out["n_flows"] == 1
    r = out["results"][0]
    assert set(r.keys()) >= {"flow", "anomaly_score", "anomaly", "root_cause", "suggested_actions"}


def test_handler_call_base64_string(ckpt_dir, pcap_bytes):
    handler = EndpointHandler(ckpt_dir)
    b64 = base64.b64encode(pcap_bytes).decode("ascii")
    out = handler({"inputs": b64})
    assert "error" not in out
    assert out["n_flows"] == 1


def test_handler_call_dict_pcap_base64(ckpt_dir, pcap_bytes):
    handler = EndpointHandler(ckpt_dir)
    b64 = base64.b64encode(pcap_bytes).decode("ascii")
    out = handler({"inputs": {"pcap_base64": b64}, "parameters": {"threshold": 0.9}})
    assert "error" not in out
    assert out["n_flows"] == 1


def test_handler_call_non_diagnose_mode(ckpt_dir, pcap_bytes):
    handler = EndpointHandler(ckpt_dir)
    out = handler({"inputs": pcap_bytes, "parameters": {"diagnose": False}})
    assert "error" not in out
    r = out["results"][0]
    assert set(r.keys()) == {"flow", "score", "anomaly", "packets"}


def test_handler_call_bad_input_returns_error_not_exception(ckpt_dir):
    handler = EndpointHandler(ckpt_dir)
    out = handler({"inputs": 12345})
    assert "error" in out


def test_handler_output_is_json_serializable(ckpt_dir, pcap_bytes):
    handler = EndpointHandler(ckpt_dir)
    out = handler({"inputs": pcap_bytes})
    json.dumps(out)  # must not raise
