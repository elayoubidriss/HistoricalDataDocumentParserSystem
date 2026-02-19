import json
import logging
import uuid
from typing import List
from pathlib import Path
from datetime import datetime
from langchain.schema.document import Document
from utils.document import Document as storageDocument

from services.embeddings_services.embeddings_service import EmbeddingsService
from services.vector_db_services.vector_db_service import VectorDBService

def list_collections():
    vector_service = VectorDBService.get()
    collections = vector_service.list_collections()
    return collections

def delete_documents(collection_name: str, source_file_ids: List[str]):
    vector_service = VectorDBService.get()
    vector_service.delete_documents(collection_name=collection_name,source_file_ids=source_file_ids)

def get_server_vectorstore(collection_name:str):
    embedding_function = EmbeddingsService.get()
    vector_service = VectorDBService.get()
    return vector_service.get_langchain_server_vectorstore(collection_name=collection_name,embedding_function=embedding_function)

def get_document_ids_from_service(collection_name: str):
    vector_service = VectorDBService.get()
    return vector_service.get_document_ids(collection_name)

def get_inmemory_vectorstore(collection_name:str):
    embedding_function = EmbeddingsService.get()
    vector_service = VectorDBService.get()
    collections = vector_service.list_collections()
    print(collections)
    return vector_service.get_langchain_inmemory_vectorstore(collection_name=collection_name,embedding_function=embedding_function)


def store_document_list(document: storageDocument):
    vector_service = VectorDBService.get()
    vector_service.insert_document_list_item(document=document)

def store_datastorage(
        entry_id: uuid.UUID, # Changed from doc_id
        shared_id: uuid.UUID, # Added shared_id
        page_content: str,
        source_file: uuid.UUID,
        media_type: str,
        processing_type: str):
    
    vector_service = VectorDBService.get()
    vector_service.insert_datastorage_item(
        entry_id=entry_id, # Pass entry_id
        shared_id=shared_id, # Pass shared_id
        page_content=page_content,
        source_file=source_file,
        media_type=media_type,
        processing_type=processing_type
    )

def query_document_by_sharepoint_id(sharepoint_id: int):
    vector_service = VectorDBService.get()
    return vector_service.get_document_metadata_by_sharepoint_id(sharepoint_id=sharepoint_id)


def remove_document_from_db_tables(document_id: uuid.UUID) -> bool:
    vector_service = VectorDBService.get()
    return vector_service.delete_document_from_document_list(document_id=document_id)
    

def fetch_documents_from_document_list_db_by_ids(doc_ids: List[uuid.UUID]) -> List[dict]:
    vector_service = VectorDBService.get()
    return vector_service.fetch_documents_from_document_list_db_by_ids(doc_ids=doc_ids)

def db_fetch_datastorage_items_by_criteria(
    source_file_ids: List[uuid.UUID],
    media_types: List[str] = None, # Made Optional explicit, though List implies it can be None if default is None
    processing_types: List[str] = None
) -> List[dict]:
    """
    Fetches records from the datastorage table based on criteria.
    """
    vector_service = VectorDBService.get()

    return vector_service.fetch_datastorage_items_by_criteria(
            source_file_ids=source_file_ids,
            media_types=media_types,
            processing_types=processing_types
        )


def get_user_organization(user_id: str) -> str | None:
    """Get the organization for a specific user."""
    vector_service = VectorDBService.get()
    return vector_service.get_user_organization(uuid.UUID(user_id))


def get_user_conversations(user_id: str) -> list[dict]:
    """Get all conversations associated with a user, including organization information."""
    vector_service = VectorDBService.get()
    return vector_service.get_user_conversations(uuid.UUID(user_id))


def get_conversation_messages(conversation_id: str) -> list[dict]:
    """Get all messages for a specific conversation."""
    vector_service = VectorDBService.get()
    return vector_service.get_conversation_messages(uuid.UUID(conversation_id))


def create_new_conversation(conversation_id: str, user_id: str, title: str = None) -> bool:
    """Create a new conversation with the provided ID."""
    vector_service = VectorDBService.get()
    return vector_service.create_new_conversation(uuid.UUID(conversation_id), uuid.UUID(user_id), title)


def save_message_to_conversation(
    message_id: str,
    conversation_id: str, 
    question: str, 
    answer: str, 
    sources: List[str], 
    status: str,
    timestamp: datetime,
    metadata: dict = None
) -> bool:
    """Save a message (question and answer) to a conversation."""
    print(f"DEBUG save_message_to_conversation called with:")
    print(f"  message_id: {message_id}")
    print(f"  conversation_id: {conversation_id}")
    print(f"  question: '{question}'")
    print(f"  answer: '{answer}' (length: {len(answer) if answer else 'None'})")
    print(f"  sources: {sources}")
    print(f"  status: {status}")
    print(f"  timestamp: {timestamp}")
    print(f"  metadata: {metadata}")
    
    vector_service = VectorDBService.get()
    result = vector_service.save_message_to_conversation(
        message_id,
        uuid.UUID(conversation_id), 
        question, 
        answer, 
        sources, 
        status, 
        timestamp,
        metadata
    )
    
    print(f"DEBUG: vector_service.save_message_to_conversation returned: {result}")
    return result


def get_message_details_by_id(message_id: str) -> dict | None:
    """Get message sources, organization, and organization_id by message UUID."""
    vector_service = VectorDBService.get()
    return vector_service.get_message_details_by_id(message_id)


def get_embeddings_metadata_by_organization_and_sources(organization_id: str, source_ids: List[str]) -> List[dict]:
    """Get metadata from langchain_pg_embedding table by organization_id and source IDs."""
    vector_service = VectorDBService.get()
    return vector_service.get_embeddings_metadata_by_organization_and_sources(organization_id, source_ids)

    
def store_data_vectorstore(
    collection_name,
    document_id,
    document_name,
    texts,
    text_summaries,
    tables,
    table_summaries,
    images,
    image_summaries,
    ):
    
    data_info = [
        ("text", texts, text_summaries),
        ("table", tables, table_summaries),
        ("image", images, image_summaries),
    ]

    for docs_type, docs, docs_summary in data_info:
        if docs:  
            store_vectorstore(
                collection_name,
                document_id,
                document_name,
                docs, 
                docs_summary,
                file_type="pdf",
                docs_type=docs_type)
            


def load_docs_multimodal(parent_doc_ids: List[str], content_type: List[str]) -> tuple[List[Document], List[str]]:
    """
    Loads Langchain Document objects from the datastorage table based on parent document IDs
    and requested content types (e.g., "texts_raw", "tables_summaries").
    """

    parent_doc_uuids = [uuid.UUID(pid) for pid in parent_doc_ids]
    logging.info(f'ids : {parent_doc_ids}')

    # Parse requested_combined_types into db_media_types and db_processing_types
    db_media_types = set()
    db_processing_types = set()

    for combined_type in content_type:
        parts = combined_type.split('_')
        if len(parts) == 2:
            media_type_part, processing_type_part = parts[0], parts[1]
            if media_type_part and processing_type_part in ["raw", "summaries"]:
                db_media_types.add(media_type_part)
                db_processing_types.add(processing_type_part)
            else:
                print(f"Warning: Could not parse combined_type '{combined_type}'. Skipping.")
        else:
            print(f"Warning: Could not parse combined_type '{combined_type}'. Skipping.")

    if not db_media_types or not db_processing_types:
        print("Warning: No valid media_types or processing_types derived from requested_combined_types.")
        return [], []

    fetched_items_data = db_fetch_datastorage_items_by_criteria(
        source_file_ids=parent_doc_uuids,
        media_types=list(db_media_types),
        processing_types=list(db_processing_types)
    )

    langchain_docs = []
    ids_for_retriever = []

    for item_data in fetched_items_data:  

        current_item_combined_type = f"{item_data['media_type']}_{item_data['processing_type']}"

        if current_item_combined_type not in content_type:
            continue # Skip if this specific combination was not requested

        page_content = item_data.get('page_content', '')
        metadata = {
            "source_file": str(item_data.get('source_file')),
        }

        langchain_docs.append(Document(page_content=page_content, metadata=metadata))
        ids_for_retriever.append(str(item_data.get('shared_id')))
        
    return langchain_docs, ids_for_retriever



def load_docs_json(filenames):
    docs = []
    doc_ids = []

    for filename in filenames:
        print(f"Loading {filename}_processed")
        with open(f"{filename}_processed.json", "r") as file:
            loaded_docs_with_ids = json.load(file)

            # Extend the docs list with Document objects created from the loaded data
            docs.extend([Document(**doc["document"]) for doc in loaded_docs_with_ids])

            # Extend the doc_ids list with doc_id from the loaded data
            doc_ids.extend([doc["doc_id"] for doc in loaded_docs_with_ids])
    return docs, doc_ids


def store_docs(docs, doc_ids, filename, type):
    if type == "Transcript":
        docs_as_dicts = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]

        docs_with_ids = [{"doc_id": doc_id, "document": doc} for doc_id, doc in zip(doc_ids, docs_as_dicts)]

        with open(f"{filename}_processed.json", "w") as file:
            json.dump(docs_with_ids, file)


def store_vectorstore(collection_name, doc_id, document_name, docs, docs_summaries, file_type ,docs_type):
    vectorstore = get_server_vectorstore(collection_name=collection_name)
    
    if file_type == "pdf":

        summary_docs = []
        vector_db_ids = [f"{doc_id}_{docs_type}_{i}" for i, _ in enumerate(docs, start=1)]
    
        for i,doc in enumerate(docs):
            
            if docs_type == "text" :
                if docs_summaries:
                    doc_to_embed = Document(
                    page_content=docs_summaries[i], metadata={"id": vector_db_ids[i], "source_file": doc_id, "file_name" : document_name, "content_type": docs_type, "raw_content" : doc[1], "coords" : doc[2].model_dump(), "page_no" : doc[3]}
                    )
                else : 
                    if i == 0 :
                        print(f'false {doc}')
                    doc_to_embed = Document(
                    page_content=doc[1], metadata={"id": vector_db_ids[i], "source_file": doc_id, "file_name" : document_name, "content_type": docs_type, "raw_content" : doc[1], "coords" : doc[2].model_dump(), "page_no" : doc[3]}
                    )
                summary_docs.append(doc_to_embed)
            else : 
                doc_to_embed = Document(
                    page_content=docs_summaries[i], metadata={"id": vector_db_ids[i], "source_file": doc_id, "file_name" : document_name, "content_type": docs_type, "raw_content" : doc[1], "coords" : doc[2].model_dump(), "page_no" : doc[3]}
                    )
                summary_docs.append(doc_to_embed)

        
        vectorstore.add_documents(documents=summary_docs,ids=vector_db_ids)


def construct_filter(selected_docs=None, selected_content_types=None):
    filter = {"$and": [{"source_file": {"$in": selected_docs}}, {"content_type": {"$in": selected_content_types}}]}
    return filter


def reset_collection(client, collection_name):
    client.delete_collection(collection_name)


def stuff(docs: list[Document]) -> str:
    all_text = ""

    for doc in docs:
        page_text = doc.page_content
        all_text += page_text + "\n"  # Add a newline character to separate pages

    return all_text
