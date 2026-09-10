"""Adapters: the swappable pieces (embedders, stores, and later rerankers).

Importing this package registers the built-in adapters with the registry so
their config `type:` names resolve.
"""

from ragladder.adapters import embedders as embedders
from ragladder.adapters import rerankers as rerankers
from ragladder.adapters import stores as stores
