"""Model entry points."""

from .clip_detector import CLIPBinaryDetector
from .dire_residual import ResidualEncoder
from .fusion import ClipDIREFusionDetector
from .abstention import predict_with_abstention

__all__ = [
    "CLIPBinaryDetector",
    "ResidualEncoder",
    "ClipDIREFusionDetector",
    "predict_with_abstention",
]
