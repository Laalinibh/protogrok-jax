"""Tokenizer parity and checkpoint round-trip tests."""
import os

import jax
import numpy as np

from models import ProtogrokConfig, init_params
from data import payload_to_token_ids, port_bucket, hash_bytes_from_fields
from optimization import checkpointing
from models.config import BYTE_OFFSET, PAD_ID, PAYLOAD_MAX_LEN


def test_tokenizer_parity():
    toks = payload_to_token_ids(b"\x00\x01\xff", max_len=8)
    assert toks[:3] == [BYTE_OFFSET + 0, BYTE_OFFSET + 1, BYTE_OFFSET + 255]
    assert toks[3:] == [PAD_ID] * 5
    assert len(payload_to_token_ids(b"x" * 300)) == PAYLOAD_MAX_LEN


def test_port_bucket():
    assert port_bucket(80) == 1
    assert port_bucket(1024) == 2
    assert port_bucket(50000) == 3
    assert port_bucket("bad") == 0


def test_hash_bytes_deterministic():
    a = hash_bytes_from_fields([1, "tcp", 80], 16)
    b = hash_bytes_from_fields([1, "tcp", 80], 16)
    c = hash_bytes_from_fields([1, "tcp", 81], 16)
    assert a == b and a != c and len(a) == 16


def test_checkpoint_roundtrip(tmp_path):
    cfg = ProtogrokConfig.base_124m()
    _, variables = init_params(cfg, jax.random.PRNGKey(0))
    params = variables["params"]
    path = os.path.join(tmp_path, "ckpt")
    checkpointing.save(path, params, cfg, step=1)
    restored, cfg2 = checkpointing.restore(path, params_template=params)
    assert cfg2.d_model == cfg.d_model
    leaves_a = jax.tree_util.tree_leaves(params)
    leaves_b = jax.tree_util.tree_leaves(restored)
    for a, b in zip(leaves_a, leaves_b):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-6)
