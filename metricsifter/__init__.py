from importlib.metadata import PackageNotFoundError, version

from metricsifter.sifter import Sifter
from metricsifter.transformer import SifterTransformer
from metricsifter.types import Segment, SegmentInfo, SiftResult

try:
    __version__ = version("metricsifter")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["Sifter", "SifterTransformer", "Segment", "SegmentInfo", "SiftResult", "__version__"]
