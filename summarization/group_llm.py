"""Deprecated compatibility alias for ``summarization.label``."""

import sys as _sys

from summarization import label as _label
from summarization.label import *  # noqa: F401,F403

_sys.modules[__name__] = _label
