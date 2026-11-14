"""Bounded NPZ numeric-pool materialization; the original codec is unchanged."""
from contextlib import contextmanager
import numpy as np

@contextmanager
def cached_unpack(module):
    original = module._unpack
    def materialized(arrays):
        if isinstance(arrays, np.lib.npyio.NpzFile):
            arrays = {name: arrays[name] for name in arrays.files}
        return original(arrays)
    module._unpack = materialized
    try:
        yield
    finally:
        module._unpack = original
