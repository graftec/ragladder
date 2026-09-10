"""ragladder — RAG retrieval-layer evaluation with stage attribution and wrong-context metrics.

Thin public API for notebook/library use; the CLI (`ragladder.cli`) sits on top.

    from ragladder import Study
    result = Study.from_yaml("study.yaml").run()

Milestone 1 implements the dense rung of the ablation ladder; +bm25 / +rerank
rungs are reported as pending. See the README for status.
"""

from __future__ import annotations

from pathlib import Path

import ragladder.adapters as _adapters  # noqa: F401  (registers built-in adapters)
from ragladder.config import StudyConfig
from ragladder.pipeline.runner import RunResult, run_study

__version__ = "0.1.0"


class Study:
    """A study loaded from config, runnable to a RunResult."""

    def __init__(self, config: StudyConfig):
        self.config = config

    @classmethod
    def from_yaml(cls, path: str | Path) -> Study:
        return cls(StudyConfig.from_yaml(path))

    def run(self, limit: int | None = None, cache_dir: str | None = ".cache/embeddings") -> RunResult:
        return run_study(self.config, limit=limit, cache_dir=cache_dir)


__all__ = ["RunResult", "Study", "StudyConfig", "__version__", "run_study"]
