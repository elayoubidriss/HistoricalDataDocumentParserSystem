import abc
from typing import List, Dict, Any
class BaseVectorDB(abc.ABC):
    """Abstract base class for a vector database used for document retrieval."""

    @abc.abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the database.
        Each document should be a dict with keys like 'text' and 'embedding'.
        """
        pass

    @abc.abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k documents relevant to the query.
        """
        pass
    
    @abc.abstractmethod
    def list_collections(self) -> List[str]:
        """
        Returns the list of collection names from vector table.
        """
        pass

    @abc.abstractmethod
    def get_document_ids(self, collection_name: str) -> List[str]:
        """
        Retrieve all document custom_ids from collection.
        """
        pass

    @abc.abstractmethod
    def delete_documents(self, collection_name: str, source_file_ids: List[str]) -> None:
        """
        Delete documents from a collection by source_file IDs.
        """
        pass
