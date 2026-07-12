"""Shape, parameter-count, and train-step tests for Protogrok-JAX."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from models import ProtogrokConfig, count_params, init_params
from data import synthetic_batch
from optimization import (
    build_optimizer, create_train_state, train_step, eval_step, batch_to_device,
)


def test_param_count_124m():
    cfg = ProtogrokConfig.base_124m()
    _, variables = init_params(cfg, jax.random.PRNGKey(0))
    n = count_params(variables["params"]) / 1e6
    # PyTorch reported 125.1M for this config; allow a small tolerance.
    assert 115 < n < 135, f"124m config gave {n:.1f}M"


def test_param_count_300m():
    cfg = ProtogrokConfig.large_300m()
    _, variables = init_params(cfg, jax.random.PRNGKey(0))
    n = count_params(variables["params"]) / 1e6
    assert 250 < n < 360, f"300m config gave {n:.1f}M"


@pytest.mark.parametrize("task,width", [("anomaly", 2), ("class", 20), ("protocol", 259)])
def test_forward_shapes(task, width):
    cfg = ProtogrokConfig.base_124m(num_classes=20)
    model, variables = init_params(cfg, jax.random.PRNGKey(1))
    b = synthetic_batch(3, cfg)
    out = model.apply(variables, jnp.asarray(b.payload), jnp.asarray(b.headers),
                      jnp.asarray(b.proto_id), task=task, deterministic=True)
    assert out.shape == (3, width)
    assert jnp.isfinite(out).all()


def test_pooled_and_determinism():
    cfg = ProtogrokConfig.base_124m()
    model, variables = init_params(cfg, jax.random.PRNGKey(2))
    b = synthetic_batch(2, cfg)
    args = (jnp.asarray(b.payload), jnp.asarray(b.headers), jnp.asarray(b.proto_id))
    o1 = model.apply(variables, *args, task="pooled", deterministic=True)
    o2 = model.apply(variables, *args, task="pooled", deterministic=True)
    assert o1.shape == (2, cfg.d_model)
    np.testing.assert_allclose(np.asarray(o1), np.asarray(o2), rtol=1e-6, atol=1e-6)


def test_train_step_reduces_loss():
    cfg = ProtogrokConfig.base_124m()
    opt = build_optimizer(total_steps=30, peak_lr=1e-3, warmup_steps=2)
    state = create_train_state(cfg, jax.random.PRNGKey(3), opt)
    batch = batch_to_device(synthetic_batch(8, cfg, num_labels=2, seed=7))
    first = None
    for _ in range(15):
        state, m = train_step(state, batch, task="anomaly")
        if first is None:
            first = float(m["loss"])
        assert np.isfinite(float(m["loss"]))
    assert float(m["loss"]) < first, "loss did not decrease on a fixed batch"


def test_eval_step():
    cfg = ProtogrokConfig.base_124m()
    opt = build_optimizer(total_steps=10, peak_lr=1e-3, warmup_steps=2)
    state = create_train_state(cfg, jax.random.PRNGKey(4), opt)
    out = eval_step(state, batch_to_device(synthetic_batch(4, cfg)), task="anomaly")
    assert out["logits"].shape == (4, 2)
    assert 0.0 <= float(out["accuracy"]) <= 1.0
