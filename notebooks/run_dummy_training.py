"""Compatibility launcher for the upstream AlphaStar dummy learner on JAX 0.4."""

import runpy
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp

# AlphaStar Unplugged was released against an older JAX/Acme combination.
jax.xla = types.SimpleNamespace(Device=type(jax.devices()[0]))
jax.tree_map = jax.tree_util.tree_map
jax.random.KeyArray = jax.Array
jnp.DeviceArray = jax.Array

# The dummy learner only uses Reverb in a type annotation. Avoid requiring the
# platform-specific dm-reverb extension just to run the synthetic-data smoke test.
reverb = types.ModuleType("reverb")
reverb.ReplaySample = object
sys.modules["reverb"] = reverb

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
runpy.run_path(str(root / "alphastar/unplugged/scripts/train.py"), run_name="__main__")
