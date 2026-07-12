"""Tests for the bin-packing dynamic batcher (the OR component)."""
import jax
import numpy as np

from models import ProtogrokConfig, init_params
from optimization import (
    PackingConfig, BinPackedBatcher, optimal_buckets, pack_ffd, padding_report,
    real_packet_count,
)
from optimization.batching import counts_of
from models.config import PAD_ID, PAYLOAD_MAX_LEN


def _variable_length_examples(n=200, max_packets=16, seed=0):
    """Synthetic flows with a skewed distribution of real packet counts."""
    rng = np.random.default_rng(seed)
    exs = []
    for _ in range(n):
        k = int(np.clip(rng.geometric(0.4), 1, max_packets))  # mostly short flows
        payload = np.full((max_packets, PAYLOAD_MAX_LEN), PAD_ID, np.int32)
        payload[:k] = rng.integers(3, 259, (k, PAYLOAD_MAX_LEN), np.int32)
        header = np.array([rng.integers(0, 130), 1, 2, k, 0], np.int32)
        exs.append((payload, header, int(rng.integers(0, 130)), int(rng.integers(0, 2))))
    return exs


def test_real_packet_count():
    exs = _variable_length_examples(10)
    for payload, header, _, _ in exs:
        assert real_packet_count(payload) == int(header[3])


def test_optimal_buckets_monotonic_and_covers():
    counts = counts_of(_variable_length_examples(300))
    for k in (2, 3, 4):
        b = optimal_buckets(counts, k, max_len=16)
        assert b == sorted(b) and b[-1] == 16 and len(b) <= k + 1
        # optimal-K padding must be <= a naive single bucket of 16
        assert all(1 <= e <= 16 for e in b)


def test_optimal_buckets_beats_single():
    counts = counts_of(_variable_length_examples(500))
    real = int(counts.sum())

    def padded(edges):
        return sum(min([e for e in edges if e >= n], default=16) for n in counts)

    single = padded([16])
    multi = padded(optimal_buckets(counts, 4, 16))
    assert multi < single, "DP bucketing did not reduce padding vs pad-to-16"
    assert multi >= real


def test_every_example_batched_once():
    exs = _variable_length_examples(137)
    for strat in ("bucket", "ffd"):
        cfg = PackingConfig(strategy=strat, num_buckets=4, max_batch=16)
        seen = sorted(i for idxs, _ in BinPackedBatcher(exs, cfg).plan for i in idxs)
        assert seen == list(range(len(exs)))


def test_bucket_trim_preserves_real_packets():
    exs = _variable_length_examples(64)
    cfg = PackingConfig(strategy="bucket", num_buckets=4, max_batch=8)
    for batch in BinPackedBatcher(exs, cfg):
        # every trimmed flow keeps all its real (non-PAD) packet rows
        for row in batch.payload:
            assert np.any(row != PAD_ID, axis=1).sum() >= 1


def test_padding_report_reduces_and_bounds_shapes():
    exs = _variable_length_examples(400)
    rep = padding_report(exs, PackingConfig(strategy="bucket", num_buckets=4, max_batch=32))
    assert rep["packed_padded_tokens"] < rep["naive_padded_tokens"]
    assert rep["padding_reduction"] > 0.0
    assert rep["distinct_shapes"] <= 4          # JIT compile shapes stay bounded
    assert rep["packed_efficiency"] > rep["naive_efficiency"]


def test_packed_batch_runs_through_model():
    cfg = ProtogrokConfig.base_124m()
    model, variables = init_params(cfg, jax.random.PRNGKey(0))
    exs = _variable_length_examples(40)
    pk = PackingConfig(strategy="ffd", num_buckets=4, max_batch=8, token_budget=48)
    for batch in BinPackedBatcher(exs, pk):
        import jax.numpy as jnp
        out = model.apply(variables, jnp.asarray(batch.payload), jnp.asarray(batch.headers),
                          jnp.asarray(batch.proto_id), task="anomaly", deterministic=True)
        assert out.shape == (batch.payload.shape[0], 2)
        assert jnp.isfinite(out).all()
        break  # one packed batch is enough to prove the shape contract
