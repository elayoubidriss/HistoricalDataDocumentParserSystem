from typing import List, Dict, Any

from langchain_community.vectorstores import Chroma
from langchain.schema.vectorstore import VectorStore

from services.vector_db_services.base_vector_db import BaseVectorDB
from services.embeddings_services.base_embeddings import BaseEmbeddings
from models.settings import ChromaVectorDBSettings


import chromadb
from chromadb.config import Settings

class ChromaVectorDBService(BaseVectorDB):
    """
    Vector database implementation using ChromaDB.
    """
    def __init__(self, settings: ChromaVectorDBSettings):
        self.settings = settings

        self.client = chromadb.HttpClient(
            host=settings.host,
            port=settings.port,
            settings=Settings(allow_reset=True)
        )

        print(f"Initializing ChromaDB with settings: {self.settings.model_dump_json()}")

    def get_langchain_server_vectorstore(self, collection_name: str, embedding_function: BaseEmbeddings) -> VectorStore:
        """
        Get a Langchain server Chroma instance for a specific collection.
        """
        return Chroma(
            client=self.client,
            embedding_function=embedding_function,
            collection_name=collection_name,
        ) 
    
    def get_langchain_inmemory_vectorstore(self, collection_name: str, embedding_function: BaseEmbeddings) -> VectorStore:
        """
        Get a Langchain inmemory Chroma instance for a specific collection.
        """
        return Chroma(
            embedding_function=embedding_function,
            collection_name=collection_name,
        ) 

    def list_collections(self):
        """
        Returns the list of collection names from Chroma.
        """
        collections = self.client.list_collections()
        return collections
    
    def get_document_ids(self, collection_name: str) -> List[str]:
        """
        Retrieve all document ids from a Chroma collection.
        """

        # Check if collection exists
        collection_names = self.client.list_collections()
        if collection_name not in [c.name for c in collection_names]:
            return []
            
        # Get the collection
        collection = self.client.get_collection(name=collection_name)
        
        # Get all IDs from the collection
        result = collection.get()
        
        # Return the IDs list or empty list if none found
        return result.get("ids", [])
    
    def delete_documents(self, collection_name: str, source_file_ids: List[str]) -> None:
        """
        Delete documents from a Chroma collection by source_file IDs.
        """
        try:
            # Check if the collection exists
            existing_collections = [c.name for c in self.client.list_collections()]
            if collection_name not in existing_collections:
                raise ValueError(f"Collection '{collection_name}' does not exist.")

            # Get the collection
            collection = self.client.get_collection(name=collection_name)

            # Perform deletion using metadata filter
            collection.delete(where={"source_file": {"$in": source_file_ids}})

            print(f"Deleted documents with source_file in {source_file_ids} from collection '{collection_name}'.")

        except Exception as e:
            print(f"Error deleting documents from ChromaDB: {e}")
            raise
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the database.
        Each document should be a dict with keys like 'text' and 'embedding'.
        """
        pass

   
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k documents relevant to the query.
        """
        pass 


    

