from typing import List, Dict, Any

from services.vector_db_services.base_vector_db import BaseVectorDB
from models.settings import BaseVectorDBSettings


class DummyVectorDB(BaseVectorDB):
    """A simple in-memory vector database implementation."""

    def __init__(self, settings: BaseVectorDBSettings):
        self.settings = settings
        print(f"Initializing DummyVectorDB with settings: {self.settings.model_dump_json()}")
        self.documents: List[Dict[str, Any]] = []

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        self.documents.extend(documents)
        print(f"Added {len(documents)} documents to the vector database.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        print(f"Searching for query: '{query}' and returning top {top_k} documents.")
        return self.documents[:top_k]
    
    
    def list_collections(self) -> List[str]:
        """
        Returns the list of collection names from vector table.
        """
        pass

    
    def get_document_ids(self, collection_name: str) -> List[str]:
        """
        Retrieve all document custom_ids from collection.
        """
        pass

    
    def delete_documents(self, collection_name: str, source_file_ids: List[str]) -> None:
        """
        Delete documents from a collection by source_file IDs.
        """
        pass