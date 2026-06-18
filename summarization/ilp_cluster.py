"""Deprecated compatibility alias for ``summarization.cluster``.

The ILP implementation is now the canonical clustering stage in
``summarization.cluster``.
"""

import sys as _sys

from summarization import cluster as _cluster
from summarization.cluster import *  # noqa: F401,F403

_sys.modules[__name__] = _cluster
