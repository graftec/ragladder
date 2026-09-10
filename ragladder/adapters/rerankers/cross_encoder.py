"""Cross-encoder reranker via sentence-transformers (BAAI/bge-reranker-v2-m3).

Off-the-shelf at launch; a LoRA-fine-tuned reranker is a documented follow-up.
The model is loaded lazily so importing the package never triggers a download.
"""

from __future__ import annotations

from ragladder.adapters.rerankers.base import Candidate
from ragladder.registry import register_reranker


@register_reranker("cross_encoder")
class CrossEncoderReranker:
    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3", name: str | None = None, **options):
        self.model_id = model
        self.name = name or model
        self._options = options
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for reranker type 'cross_encoder'. "
                "Install with: pip install sentence-transformers"
            ) from e
        device = self._options.get("device")
        self._model = CrossEncoder(self.model_id, device=device)

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        self._ensure_model()
        scores = self._model.predict(
            [(query, c.text) for c in candidates], show_progress_bar=False
        )
        rescored = [
            Candidate(id=c.id, text=c.text, score=float(s))
            for c, s in zip(candidates, scores)
        ]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored
