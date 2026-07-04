from importlib.metadata import PackageNotFoundError, version

from metricsifter.evaluation import SelectionMetrics, evaluate_selection
from metricsifter.sifter import Sifter
from metricsifter.transformer import SifterTransformer
from metricsifter.types import Segment, SegmentCandidate, SegmentInfo, SiftResult

try:
    __version__ = version("metricsifter")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "Sifter",
    "SifterTransformer",
    "Segment",
    "SegmentCandidate",
    "SegmentInfo",
    "SiftResult",
    "SelectionMetrics",
    "evaluate_selection",
    "__version__",
]
