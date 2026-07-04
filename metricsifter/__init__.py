from importlib.metadata import PackageNotFoundError, version

from metricsifter.sifter import Sifter
from metricsifter.types import Segment

try:
    __version__ = version("metricsifter")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["Sifter", "Segment", "__version__"]
