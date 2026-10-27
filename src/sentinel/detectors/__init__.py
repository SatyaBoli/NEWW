"""Sentinel Detector Zoo."""
from .base import BaseDetector
from .ewma import EWMADetector
from .isolation_forest import StreamingIsolationForestDetector
from .rolling_mad import RollingMedianMADDetector

try:
    from .tcn_autoencoder import CausalTCNAEDetector, TCNAEModule
    from .ensemble import HybridEnsembleDetector
except ImportError:
    class CausalTCNAEDetector:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required to run CausalTCNAEDetector. "
                "Install PyTorch via `pip install torch`."
            )

    class TCNAEModule:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to instantiate TCNAEModule.")

    class HybridEnsembleDetector:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch is required to run HybridEnsembleDetector. "
                "Install PyTorch via `pip install torch`."
            )

__all__ = [
    "BaseDetector",
    "CausalTCNAEDetector",
    "EWMADetector",
    "HybridEnsembleDetector",
    "RollingMedianMADDetector",
    "StreamingIsolationForestDetector",
    "TCNAEModule",
]
