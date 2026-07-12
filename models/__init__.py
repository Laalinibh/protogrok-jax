"""Core Protogrok model: architecture, config, and inference."""
from models.config import ProtogrokConfig
from models.protogrok import (
    ProtogrokModel, PayloadEncoder, HeaderEncoder, TransformerBlock,
    ProtocolAdapter, MemoryModule, init_params, count_params,
)
from models.inference import make_apply, anomaly_scores, evaluate, macro_f1

__all__ = [
    "ProtogrokConfig", "ProtogrokModel", "PayloadEncoder", "HeaderEncoder",
    "TransformerBlock", "ProtocolAdapter", "MemoryModule", "init_params",
    "count_params", "make_apply", "anomaly_scores", "evaluate", "macro_f1",
]
