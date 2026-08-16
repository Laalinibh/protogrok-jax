"""Configuration and tokenizer constants for Protogrok-JAX.

Faithful port of the constants and hyper-parameters from the original
PyTorch notebook (``modeling_protogrok.py`` / ``protogrok_config.json``).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict

# --------------------------------------------------------------------------- #
# Tokenizer / byte-vocabulary constants (must match the PyTorch data pipeline)
# --------------------------------------------------------------------------- #
PAD_ID: int = 0
BOS_ID: int = 1
EOS_ID: int = 2
BYTE_OFFSET: int = 3
BYTE_VOCAB: int = 256 + 3          # 259
PAYLOAD_MAX_LEN: int = 128         # L: bytes kept per packet
MAX_PACKETS_PER_FLOW: int = 16     # T: packets kept per flow
HEADER_FIELDS: int = 5             # [proto_id, sport_bucket, dport_bucket, s0, s1]
NUM_PORT_BUCKETS: int = 4


@dataclasses.dataclass(frozen=True)
class ProtogrokConfig:
    """Model hyper-parameters.

    The defaults reproduce the ~300M-parameter network from the notebook;
    :meth:`base_124m` returns the ~124M configuration.
    """

    d_model: int = 1024
    payload_dim: int = 512
    header_dim: int = 384
    packet_layers: int = 16
    session_layers: int = 8
    nhead: int = 16
    mlp_ratio: int = 4
    memory_slots: int = 8
    dropout_rate: float = 0.1

    # data-derived / task
    proto_vocab: int = 132
    num_classes: int = 20          # traffic-class head width (anomaly head is fixed at 2)
    byte_vocab: int = BYTE_VOCAB
    payload_max_len: int = PAYLOAD_MAX_LEN
    max_packets: int = MAX_PACKETS_PER_FLOW

    # numerics
    dtype: str = "float32"         # compute dtype; "bfloat16" for TPU/GPU training
    param_dtype: str = "float32"

    #: How PayloadEncoder's two 1D convolutions are implemented. "conv" uses
    #: ``nn.Conv``; "matmul" uses mathematically identical shifted matmuls with
    #: the same parameter shapes. Some XLA:GPU versions expand the convolution
    #: over the [B*T, L, E] payload tensor into a buffer orders of magnitude
    #: larger than the model needs -- "matmul" avoids that path. Checkpoints are
    #: interchangeable between the two.
    payload_conv_impl: str = "conv"

    #: Gradient checkpointing on the transformer blocks. Trades ~30% extra
    #: compute for a large drop in peak activation memory, and prevents XLA
    #: from holding the whole layer stack live in one fusion. Recommended on
    #: GPU; harmless on CPU.
    remat: bool = False

    @property
    def bottleneck(self) -> int:
        return max(128, self.d_model // 8)

    @classmethod
    def base_124m(cls, **overrides: Any) -> "ProtogrokConfig":
        cfg = dict(
            d_model=768, payload_dim=384, header_dim=256,
            packet_layers=12, session_layers=4, nhead=12,
        )
        cfg.update(overrides)
        return cls(**cfg)

    @classmethod
    def large_300m(cls, **overrides: Any) -> "ProtogrokConfig":
        return cls(**overrides)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProtogrokConfig":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def from_yaml(cls, path: str, **overrides: Any) -> "ProtogrokConfig":
        """Load a config from a YAML (or JSON) file in ``configs/``."""
        with open(path) as f:
            text = f.read()
        try:
            import yaml
            d = yaml.safe_load(text)
        except ImportError:
            import json
            d = json.loads(text)
        d = dict(d or {})
        d.update(overrides)
        return cls.from_dict(d)
