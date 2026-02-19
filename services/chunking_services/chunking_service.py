from typing import Type, Dict

from services.chunking_services.base_chunker import BaseChunker
from services.chunking_services.simple_chunker import SimpleChunker
from services.chunking_services.unstructured_chunker import UnstructuredChunker
from services.chunking_services.docling_chunker import DoclingChunker
from models.settings import BaseChunkerSettings
from models.settings import (
    BaseChunkerSettings,
    DoclingChunkerSettings,
    UnstructuredChunkerSettings,
    SimpleChunkerSettings
)


class ChunkerFactory:
    _mapping: Dict[str, Type[BaseChunker]] = {
        "simple_chunker": SimpleChunker,
        "unstructured_chunker": UnstructuredChunker,
        "docling_chunker": DoclingChunker
    }

    @classmethod
    def create(cls, name: str, settings: BaseChunkerSettings) -> BaseChunker:
        # Map to the specific settings class based on chunker type
        if name == "docling_chunker":
            validated_settings = DoclingChunkerSettings(**settings.dict())
        elif name == "unstructured_chunker":
            validated_settings = UnstructuredChunkerSettings(**settings.dict())
        elif name == "simple_chunker":
            validated_settings = SimpleChunkerSettings(**settings.dict())
        else:
            raise ValueError(f"Invalid chunker type: {name}")

        return cls._mapping[name](settings=validated_settings)


class ChunkerService:
    _instance: BaseChunker = None

    @classmethod
    def init(cls, settings: BaseChunkerSettings) -> None:
        cls._instance = ChunkerFactory.create(settings.type, settings=settings)
        print("ChunkerService initialized.")

    @classmethod
    def get(cls) -> BaseChunker:
        if cls._instance is None:
            raise RuntimeError("ChunkerService is not initialized.")
        return cls._instance