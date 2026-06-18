"""Deprecated compatibility shim for the old classification module.

Per-feature classification has been removed from the package surface. Import
``filter_act_density`` from ``summarization.prune`` in new code.
"""

from summarization.prune import filter_act_density

__all__ = ["filter_act_density"]
