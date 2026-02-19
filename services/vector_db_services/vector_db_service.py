from typing import Type, Dict

from services.vector_db_services.base_vector_db import BaseVectorDB
from services.vector_db_services.dummy_vector_db import DummyVectorDB
from services.vector_db_services.pg_vector_db import PGVectorService # Added import
from services.vector_db_services.chroma_vector_db import ChromaVectorDBService # Added import
from models.settings import BaseVectorDBSettings

class VectorDBFactory:
    _mapping: Dict[str, Type[BaseVectorDB]] = {
        "dummy_vector_db": DummyVectorDB,
        "pgvector_db": PGVectorService, # Added pgvector mapping
        "chroma_db": ChromaVectorDBService, # Added chroma mapping
    }

    @classmethod
    def create(cls, name: str, settings: BaseVectorDBSettings) -> BaseVectorDB:
        if name not in cls._mapping:
            raise ValueError(f"VectorDB '{name}' is not recognized. Options: {list(cls._mapping.keys())}")
        return cls._mapping[name](settings=settings)
    
class VectorDBService:
    _instance: BaseVectorDB = None

    @classmethod
    def init(cls, settings: BaseVectorDBSettings) -> None:
        cls._instance = VectorDBFactory.create(settings.type, settings=settings)
        print("VectorDBService initialized.")

    @classmethod
    def get(cls) -> BaseVectorDB:
        if cls._instance is None:
            raise RuntimeError("VectorDBService is not initialized.")
        return cls._instance
