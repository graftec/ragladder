"""Store adapters. Import registers the built-in types."""

from ragladder.adapters.stores import inmemory_exact as _inmemory  # noqa: F401
from ragladder.adapters.stores.base import Store

__all__ = ["Store"]
