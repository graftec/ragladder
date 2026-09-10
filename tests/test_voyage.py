"""Voyage embedder adapter: registration, config, and lazy failure modes.

The `voyageai` package isn't a test dependency, so these tests exercise the
adapter's contract without hitting the API — construction is cheap, and using it
fails with a clear, actionable error.
"""

import numpy as np
import pytest

from ragladder.config import EmbedderConfig
from ragladder.registry import get_embedder


def _cfg(**kw):
    base = {"name": "voyage-law-2", "type": "voyage", "model": "voyage-law-2"}
    base.update(kw)
    return EmbedderConfig(
        name=base["name"], type=base["type"], model=base["model"],
        api_key_env=base.get("api_key_env"),
        options=base.get("options", {}),
    )


def test_registered():
    cls = get_embedder("voyage")
    emb = cls(_cfg())
    assert emb.name == "voyage-law-2"
    assert emb.dim == 1024  # voyage-law-2 native dim
    assert emb.max_input_tokens == 16000


def test_bad_kind_rejected():
    emb = get_embedder("voyage")(_cfg())
    with pytest.raises(ValueError, match="query.*document"):
        emb.embed(["x"], kind="passage")


def test_missing_key_or_package_raises(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    emb = get_embedder("voyage")(_cfg())
    # Either voyageai isn't installed (ImportError) or the key is missing
    # (RuntimeError) — both are clear, actionable failures, not an AttributeError.
    with pytest.raises((ImportError, RuntimeError)):
        emb.embed(["hello"], kind="document")


def test_stubbed_client_batches(monkeypatch):
    """With a stub client, embed() batches and returns an (n, dim) float array."""
    emb = get_embedder("voyage")(_cfg())

    class _Resp:
        def __init__(self, texts):
            self.embeddings = [[0.1, 0.2, 0.3] for _ in texts]

    class _StubClient:
        def __init__(self):
            self.calls = 0

        def embed(self, batch, model, input_type):
            self.calls += 1
            assert input_type == "document"
            return _Resp(batch)

    emb._client = _StubClient()  # bypass _ensure_client (no package/key needed)
    out = emb.embed(["a", "b", "c"], kind="document")
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 3)
    assert out.dtype == np.float32
