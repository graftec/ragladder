"""Name -> adapter class registry.

Everything swappable in a study (embedder, store, and later reranker) is a named
adapter chosen in config. This module maps the config `type:` string to a class.
Adapters register themselves on import via the decorators below; tests (or third
parties) can register additional types programmatically.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")

_EMBEDDERS: dict[str, type] = {}
_STORES: dict[str, type] = {}
_RERANKERS: dict[str, type] = {}


def register_embedder(name: str) -> Callable[[type[_T]], type[_T]]:
    def deco(cls: type[_T]) -> type[_T]:
        _EMBEDDERS[name] = cls
        return cls

    return deco


def register_store(name: str) -> Callable[[type[_T]], type[_T]]:
    def deco(cls: type[_T]) -> type[_T]:
        _STORES[name] = cls
        return cls

    return deco


def get_embedder(name: str) -> type:
    if name not in _EMBEDDERS:
        raise KeyError(
            f"unknown embedder type {name!r}; registered: {sorted(_EMBEDDERS)}"
        )
    return _EMBEDDERS[name]


def get_store(name: str) -> type:
    if name not in _STORES:
        raise KeyError(f"unknown store type {name!r}; registered: {sorted(_STORES)}")
    return _STORES[name]


def register_reranker(name: str) -> Callable[[type[_T]], type[_T]]:
    def deco(cls: type[_T]) -> type[_T]:
        _RERANKERS[name] = cls
        return cls

    return deco


def get_reranker(name: str) -> type:
    if name not in _RERANKERS:
        raise KeyError(
            f"unknown reranker type {name!r}; registered: {sorted(_RERANKERS)}"
        )
    return _RERANKERS[name]
