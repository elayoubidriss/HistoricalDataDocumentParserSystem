from typing import List

from services.embeddings_services.base_embeddings import BaseEmbeddings
from models.settings import DummyEmbeddingsSettings

class DummyEmbeddings(BaseEmbeddings):
    """A dummy embeddings model that converts text into a list of floats."""

    def __init__(self, settings: DummyEmbeddingsSettings):
        self.settings = settings
        print(f"Initializing DummyEmbeddings with settings: {self.settings.json()}")
        self.dimension = settings.dimension

    def embed(self, text: str) -> List[float]:
        vec = [float(ord(c)) for c in text]
        if len(vec) < self.dimension:
            vec.extend([0.0] * (self.dimension - len(vec)))
        else:
            vec = vec[:self.dimension]
        return vec
    
    def embed_parallel(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(text) for text in texts]
    