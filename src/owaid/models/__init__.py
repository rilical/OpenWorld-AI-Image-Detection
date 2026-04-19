"""Model entry points."""

from .clip_detector import CLIPBinaryDetector
from .dire_residual import ResidualEncoder
from .fusion import ClipDIREFusionDetector
from .sgf_net import SGFNet
from .abstention import predict_with_abstention
from .domain_adversarial import DomainAdversarialWrapper, grad_reverse, GradReverse

__all__ = [
    "CLIPBinaryDetector",
    "ResidualEncoder",
    "ClipDIREFusionDetector",
    "SGFNet",
    "predict_with_abstention",
    "DomainAdversarialWrapper",
    "grad_reverse",
    "GradReverse",
]
