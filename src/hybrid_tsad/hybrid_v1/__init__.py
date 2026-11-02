"""Complete Hybrid V1 candidate; upstream code stays external and immutable."""
from .config import HybridV1Config
from .provenance import OfficialSources

__all__ = ["HybridV1", "HybridV1Config", "OfficialSources"]


def __getattr__(name):
    if name == "HybridV1":
        from .model import HybridV1
        return HybridV1
    raise AttributeError(name)
