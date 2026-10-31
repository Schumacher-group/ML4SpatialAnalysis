"""Patient-level spatial modeling pipeline.

The legacy code in this repository predicts one label per ROI.  This package
uses the clinical unit of observation instead: one patient is one sample and
their ROIs form a bag of spatial cell graphs.
"""

from .schema import CACHE_VERSION, DEFAULT_MARKERS

__all__ = ["CACHE_VERSION", "DEFAULT_MARKERS"]

