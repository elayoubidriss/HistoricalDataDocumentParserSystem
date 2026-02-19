import abc
from typing import List

class BaseEmbeddings(abc.ABC):
    """Abstract base class for an embeddings model."""

    @abc.abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Convert text into a vector (list of floats).
        """
        raise NotImplementedError()

    def embed_parallel(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError()