"""Sentinel Detector Zoo."""
from .base import BaseDetector
from .rolling_mad import RollingMedianMADDetector

__all__ = ["BaseDetector", "RollingMedianMADDetector"]
