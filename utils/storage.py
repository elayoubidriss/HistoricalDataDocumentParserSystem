import tempfile
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import uuid # Added for UUID conversion
from utils.document import Document
from utils.vector_storage import fetch_documents_from_document_list_db_by_ids # Added import


temp_dir = None

def document_get_many(doc_ids: List[str]) -> List[Document]: 
    """
    Loads multiple Document instances from the database by their IDs.
    
    Parameters:
    - doc_ids: A list of Document IDs (UUID strings).

    Returns:
    - A list of Document instances in the same order as requested doc_ids.
    """
    if not doc_ids:
        return []

    # Convert string IDs to UUID objects for the database query
    try:
        uuid_doc_ids = [uuid.UUID(doc_id_str) for doc_id_str in doc_ids]
    except ValueError as e:
        raise ValueError(f"Invalid UUID format in doc_ids: {e}")

    fetched_docs_data_list = fetch_documents_from_document_list_db_by_ids(doc_ids=uuid_doc_ids)

    fetched_docs_map: Dict[str, Dict[str, Any]] = {str(doc_data['id']): doc_data for doc_data in fetched_docs_data_list}
    
    documents = []
    for doc_id_str in doc_ids:
        doc_data = fetched_docs_map.get(doc_id_str)
        if doc_data is None:
            raise ValueError(f"Document with ID '{doc_id_str}' not found in the database.")
        
        try:
            if 'id' in doc_data and isinstance(doc_data['id'], uuid.UUID):
                doc_data['id'] = str(doc_data['id'])
            document = Document(**doc_data)
            documents.append(document)
        except Exception as e: # Catch PydanticValidationError or other model instantiation errors
            print(f"Error instantiating Document model for ID {doc_id_str} with data {doc_data}: {e}")
            raise # Re-raise the error

    return documents

def working_directory():
    global temp_dir
    #MZ : Old code : dir = tempfile.TemporaryDirectory(dir="/tmp", ignore_cleanup_errors=True)
    dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    temp_dir = dir.name
    return dir


def get_working_directory() -> str:
    global temp_dir
    if not temp_dir:
        raise ValueError("Working directory not initialized !")
    return temp_dir
