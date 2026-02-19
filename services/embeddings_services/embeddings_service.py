from typing import Type, Dict

from services.embeddings_services.base_embeddings import BaseEmbeddings
from services.embeddings_services.dummy_embeddings import DummyEmbeddings
from services.embeddings_services.vllm_embeddings import VLLMEmbeddings
from models.settings import BaseEmbeddingsSettings

class EmbeddingsFactory:
    _mapping: Dict[str, Type[BaseEmbeddings]] = {
        "dummy_embeddings": DummyEmbeddings,
        "vllm_embeddings": VLLMEmbeddings
    }

    @classmethod
    def create(cls, name: str, settings: BaseEmbeddingsSettings) -> BaseEmbeddings:
        if name not in cls._mapping:
            raise ValueError(f"Embeddings model '{name}' is not recognized. Options: {list(cls._mapping.keys())}")
        return cls._mapping[name](settings=settings)
    
class EmbeddingsService:
    _instance: BaseEmbeddings = None

    @classmethod
    def init(cls, settings: BaseEmbeddingsSettings) -> None:
        cls._instance = EmbeddingsFactory.create(settings.type, settings=settings)
        print("EmbeddingsService initialized.")

    @classmethod
    def get(cls) -> BaseEmbeddings:
        if cls._instance is None:
            raise RuntimeError("EmbeddingsService is not initialized.")
        return cls._instance
