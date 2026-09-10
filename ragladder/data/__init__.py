"""Dataset loading — the BEIR JSONL contract (corpus / queries / qrels)."""

from ragladder.data.beir import Corpus, Dataset, Query, load_dataset

__all__ = ["Corpus", "Dataset", "Query", "load_dataset"]
