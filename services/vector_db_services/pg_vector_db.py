from typing import List, Dict, Iterable, Optional, Any
import uuid
import json

from langchain.schema.vectorstore import VectorStore
from langchain_postgres import PGVector
from services.vector_db_services.base_vector_db import BaseVectorDB
from services.embeddings_services.base_embeddings import BaseEmbeddings
from models.settings import PGVectorDBSettings

from sqlalchemy import create_engine, Column, Integer, String, text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, Session, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from utils.document import Document
from datetime import datetime



Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    organization = Column(String(255), nullable=False)
    organization_id = Column(UUID(as_uuid=True))
    
    # Relationship to conversations
    conversations = relationship("Conversation", back_populates="user")

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    title = Column(String(255))
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to user
    user = relationship("User", back_populates="conversations")
    # Relationship to messages
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(UUID(as_uuid=True), primary_key=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False)
    question = Column(String)
    answer = Column(String)
    sources = Column(JSONB)
    status = Column(String(20))
    message_metadata = Column(JSONB) # Renamed from metadata
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to conversation
    conversation = relationship("Conversation", back_populates="messages")

class LangchainPgCollection(Base):
    __tablename__ = 'langchain_pg_collection'
    
    uuid = Column(UUID(as_uuid=True), primary_key=True)
    name = Column(String)
    cmetadata = Column(JSONB)
    
    # Relationship to embeddings
    embeddings = relationship("LangchainPgEmbedding", back_populates="collection")

class LangchainPgEmbedding(Base):
    __tablename__ = 'langchain_pg_embedding'
    
    id = Column(String, primary_key=True)
    collection_id = Column(UUID(as_uuid=True), ForeignKey('langchain_pg_collection.uuid'), nullable=False)
    embedding = Column(Vector)
    document = Column(String)
    cmetadata = Column(JSONB)
    
    # Relationship to collection
    collection = relationship("LangchainPgCollection", back_populates="embeddings")

class NativePGVectorStore:
    def __init__(
        self,
        engine: str,
        collection: str,
        embedding_function: BaseEmbeddings,  # <-- accepts the whole model
        embedding_dim: int
    ):
        self.collection = collection
        self.embedding_function = embedding_function
        self.embedding_dim = embedding_dim
        self.engine = engine
        self.Table = self._define_table()
        Base.metadata.create_all(self.engine)

    from sqlalchemy.dialects.postgresql import JSONB

    def _define_table(self):
        class VectorTable(Base):
            __tablename__ = self.collection
            id = Column(String, primary_key=True)  
            content = Column(String)
            embedding = Column(Vector(self.embedding_dim))
            cmetadata = Column(JSONB) 

        return VectorTable


    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add or update documents in the vector store.

        Each document must contain:
        - "text": the raw document text
        - "embedding": the embedding vector
        - Optional: "metadata": a dictionary of metadata

        Returns:
        - List of stringified IDs of added documents
        """
        ids = []
        with Session(self.engine) as session:
            for i, doc in enumerate(documents):
                content = doc["text"]
                embedding = doc["embedding"]
                metadata = doc.get("metadata", {})
                row = self.Table(content=content, embedding=embedding, cmetadata=metadata)
                for key, value in metadata.items():
                    setattr(row, key, value)
                session.add(row)
                session.flush()  # to get ID before commit
                ids.append(str(row.id))
            session.commit()
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run similarity search with optional metadata filtering.

        Parameters:
        - query: search string
        - k: top-k results
        - filter: dict-based metadata filter 

        Returns:
        - List of matching documents as dicts
        """
        query_emb = self.embedding_model.embed_query(query)

        with Session(self.engine) as session:
            q = session.query(self.Table)

            if filter:
                for key, value in filter.items():
                    column = getattr(self.Table, key, None)
                    if column is not None:
                        if isinstance(value, dict) and "$in" in value:
                            q = q.filter(column.in_(value["$in"]))
                        else:
                            q = q.filter(column == value)

            results = q.order_by(
                self.Table.embedding.l2_distance(query_emb)
            ).limit(k).all()

            return [
                {
                    "text": r.content,
                    "id": str(r.id),
                    **{col: getattr(r, col) for col in r.__table__.columns.keys() if col not in {"id", "content", "embedding"}}
                }
                for r in results
            ]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[str]:
        """
        Add texts to the vector store with optional metadata and custom IDs.

        Parameters:
        - texts: list of strings to embed and store
        - metadatas: optional metadata per text
        - ids: optional list of IDs
        - kwargs: ignored for now

        Returns:
        - list of IDs added to the store
        """
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]

        if len(metadatas) != len(texts):
            raise ValueError("Length of metadatas must match length of texts")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        elif len(ids) != len(texts):
            raise ValueError("Length of ids must match length of texts")

        embeddings = self.embedding_model.embed_documents(texts)

        inserted_ids = []
        with Session(self.engine) as session:
            for i, text in enumerate(texts):
                emb = embeddings[i]
                meta = metadatas[i]
                row = self.Table(id=ids[i], content=text, embedding=emb, cmetadata=meta)
                for key, value in meta.items():
                    if hasattr(self.Table, key):
                        setattr(row, key, value)
                session.add(row)
                inserted_ids.append(ids[i])
            session.commit()

        return inserted_ids


class PGVectorService(BaseVectorDB):
    """
    Vector database implementation using PGVector.
    """
    def __init__(self, settings: PGVectorDBSettings):
        self.settings = settings
        self.drivername = settings.drivername
        self.name=settings.name
        self.username=settings.username
        self.password=settings.password
        self.host=settings.host
        self.port=settings.port
        self.embedding_dim=settings.embedding_dim
        
        self.connection_string = f"{self.drivername}://{self.username}:{self.password}@{self.host}:{self.port}/{self.name}"
        self.engine = create_engine(self.connection_string)
        
        print(f"Initializing PGVector with settings: {self.settings.model_dump_json()}")

    def get_langchain_server_vectorstore(self, collection_name: str, embedding_function: BaseEmbeddings) -> VectorStore:
        """
        Get a Langchain PGVector instance for a specific collection.
        """
        return PGVector(
            connection=self.connection_string,
            embeddings=embedding_function,
            collection_name=collection_name,
            use_jsonb=True
        )
    
    def get_langchain_inmemory_vectorstore(self, collection_name: str, embedding_function: BaseEmbeddings) -> VectorStore:
        raise NotImplementedError(
            "The 'get_langchain_inmemory_vectorstore' method isn't implemented for the PGVector service."
            "Use a different service like Chroma if you need this functionality."
        )
    
    def get_native_vectorstore(
        self, collection: str, embedding_function: BaseEmbeddings ) -> NativePGVectorStore:
        """
        Get a PGVector Native instance for a specific collection.
        """

        return NativePGVectorStore(
            engine=self.engine,
            collection=collection,
            embedding_function=embedding_function,
            embedding_dim =self.embedding_dim
        )
    
    
    def list_collections(self) -> List[str]:
        """
        Returns the list of collection names from LangChain's internal PGVector table.
        """
        with self.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT name FROM public.langchain_pg_collection ORDER BY name ASC
            """))
            return [row[0] for row in result.fetchall()]



    def get_document_ids(self, collection_name: str) -> List[str]:
        """
        Retrieve all document custom_ids from a LangChain PGVector collection.
        """
        try:
            with self.engine.connect() as conn:
                # Step 1: Get collection UUID from langchain_pg_collection
                result = conn.execute(text("""
                    SELECT uuid FROM langchain_pg_collection
                    WHERE name = :collection_name
                """), {"collection_name": collection_name})

                row = result.fetchone()
                if row is None:
                    print(f"No collection found with name '{collection_name}'")
                    return []

                collection_uuid = row[0]

                # Step 2: Fetch custom_ids from langchain_pg_embedding for that collection
                result = conn.execute(text("""
                    SELECT id FROM langchain_pg_embedding
                    WHERE collection_id = :collection_id
                """), {"collection_id": collection_uuid})

                return [row[0] for row in result.fetchall()]
        
        except Exception as e:
            print(f"Error retrieving document IDs from PGVector: {e}")
            return []
        
    def delete_documents(self, collection_name: str, source_file_ids: List[str]) -> None:
        """
        Delete documents from a LangChain PGVector collection by source_file IDs.
        """
        try:
            with self.engine.begin() as conn:  
                # Step 1: Get collection UUID from langchain_pg_collection
                result = conn.execute(text("""
                    SELECT uuid FROM langchain_pg_collection
                    WHERE name = :collection_name
                """), {"collection_name": collection_name})

                row = result.fetchone()
                if row is None:
                    raise ValueError(f"No collection found with name '{collection_name}'")

                collection_uuid = row[0]

                # Step 2: Get Source File Id
                for source_file_id in source_file_ids:
                    preview = conn.execute(text("""
                        SELECT id FROM langchain_pg_embedding
                        WHERE cmetadata->>'source_file' = :source_file AND collection_id = :collection_id
                    """), {"source_file": source_file_id, "collection_id": collection_uuid})
                    print(f"Matching rows: {preview.fetchall()}")

                # Step 3: Delete Documents
                    deleted = conn.execute(text("""
                        DELETE FROM langchain_pg_embedding
                        WHERE cmetadata->>'source_file' = :source_file AND collection_id = :collection_id
                        RETURNING id
                    """), {"source_file": source_file_id, "collection_id": collection_uuid})
                    
                    deleted_ids = deleted.fetchall()
                    print(f"Deleted {len(deleted_ids)} rows for source_file: {source_file_id}")

        except Exception as e:
            print(f"Error deleting documents: {e}")


    def insert_document_list_item(self, document: Document):
        """
        Inserts a Document object into the document_list table.
        """

        sql = text("""
            INSERT INTO document_list (
                id, organization, path, sender, name, description,
                status, created, document_type, category, sub_category,
                source_url, sharepoint_id
            ) VALUES (
                :id, :organization, :path, :sender, :name, :description,
                :status, :created, :document_type, :category, :sub_category,
                :source_url, :sharepoint_id
            )
            RETURNING id;
        """)
        try:
            
            params = {
                "id": document.id,
                "organization": document.organization,
                "path": document.path,
                "sender": document.sender,
                "name": document.name,
                "description": document.description,
                "status": document.status.value if hasattr(document.status, 'value') else document.status, # Get enum value
                "created": document.created, 
                "document_type": document.document_type.value if hasattr(document.document_type, 'value') else document.document_type, # Get enum value
                "category": document.category.value if hasattr(document.category, 'value') else document.category, # Get enum value
                "sub_category": document.sub_category,
                "source_url": document.source_url,
                "sharepoint_id": document.sharepoint_id
            }

            with self.engine.connect() as conn:
                result = conn.execute(sql, params)
                inserted_id = result.scalar_one_or_none() 
                conn.commit()
                if inserted_id:
                    print(f"Successfully inserted document metadata into document_list with ID: {inserted_id}")
                return inserted_id
        except Exception as e:
            print(f"Error inserting document metadata into document_list: {e}")
            return None


    def get_document_metadata_by_sharepoint_id(self, sharepoint_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetches document id and organization from document_list table by sharepoint_id.
        """
        sql = text("""
            SELECT id, organization FROM document_list
            WHERE sharepoint_id = :sharepoint_id
            LIMIT 1;
        """)
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sql, {"sharepoint_id": sharepoint_id})
                row = result.fetchone()
                if row:
                    # Assuming row is a RowProxy or similar, access by index or name
                    return {"id": row[0], "organization": row[1]} 
            return None
        except Exception as e:
            print(f"Error fetching document by sharepoint_id {sharepoint_id}: {e}")
            return None

    def delete_document_from_document_list(self, document_id: uuid.UUID) -> bool:
        """
        Deletes a document from the document_list table by its id.
        This will also cascade delete related entries in the datastorage table.
        Returns True if deletion was successful (at least one row affected), False otherwise.
        """
        sql = text("""
            DELETE FROM document_list
            WHERE id = :document_id
            RETURNING id;
        """)
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sql, {"document_id": document_id})
                deleted_id = result.scalar_one_or_none()
                conn.commit()
                if deleted_id:
                    print(f"Successfully deleted document {document_id} from document_list.")
                    return True
                else:
                    print(f"Document with id {document_id} not found in document_list for deletion.")
                    return False
        except Exception as e:
            print(f"Error deleting document {document_id} from document_list: {e}")
            return False

    def fetch_documents_from_document_list_db_by_ids(self, doc_ids: List[uuid.UUID]) -> List[Dict[str, Any]]:
        """
        Fetches multiple full document records from the document_list table by a list of IDs.
        Returns a list of dictionaries, where each dictionary represents a document.
        """
        if not doc_ids:
            return []

        sql = text("""
            SELECT * FROM document_list WHERE id = ANY(:doc_ids_list);
        """)
        
        documents_data = []
        try:
            with self.engine.connect() as conn:
                # Convert UUIDs to strings if required by the DB driver or specific SQL syntax,
                # but psycopg3 usually handles UUID objects directly.
                result = conn.execute(sql, {"doc_ids_list": [str(uid) for uid in doc_ids]})
                rows = result.fetchall()
                column_names = result.keys()
                for row in rows:
                    documents_data.append(dict(zip(column_names, row)))
            return documents_data
        except Exception as e:
            print(f"Error fetching documents by IDs from document_list: {e}")
            return []

    def fetch_datastorage_items_by_criteria(
        self,
        source_file_ids: List[uuid.UUID],
        media_types: Optional[List[str]] = None,
        processing_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetches records from the datastorage table based on criteria.
        """

        conditions = ["source_file = ANY(:source_file_ids_list)"]
        params = {"source_file_ids_list": [str(uid) for uid in source_file_ids]}

        if media_types:
            conditions.append("media_type = ANY(:media_types_list)")
            params["media_types_list"] = media_types
        
        if processing_types:
            conditions.append("processing_type = ANY(:processing_types_list)")
            params["processing_types_list"] = processing_types

        where_clause = " AND ".join(conditions)
        
        sql_query_str = f"""
            SELECT shared_id, page_content, source_file, media_type, processing_type
            FROM datastorage
            WHERE {where_clause};
        """
        sql = text(sql_query_str)
        
        items_data = []
        try:
            with self.engine.connect() as conn:
                result = conn.execute(sql, params)
                rows = result.fetchall()
                column_names = result.keys()
                for row in rows:
                    items_data.append(dict(zip(column_names, row)))
            return items_data
        except Exception as e:
            print(f"Error fetching datastorage items by criteria: {e}")
            return []

    def get_user_organization(self, user_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """
        Get the organization information for a specific user.
        
        Parameters:
        - user_id: UUID of the user
        
        Returns:
        - Dictionary with organization and organization_id if user exists, None otherwise
        """
        try:
            with Session(self.engine) as session:
                user = session.query(User).filter(User.id == user_id).first()
                if user:
                    return {
                        "organization": user.organization,
                        "organization_id": str(user.organization_id) if user.organization_id else None
                    }
                return None
                
        except Exception as e:
            print(f"Error fetching organization for user {user_id}: {e}")
            return None

    def get_user_conversations(self, user_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        Get all conversations associated with a user, including organization information.
        
        Parameters:
        - user_id: UUID of the user
        
        Returns:
        - List of dictionaries containing conversation data with organization info
        """
        try:
            # Get user organization first
            organization_info = self.get_user_organization(user_id)
            if organization_info is None:
                print(f"User {user_id} not found")
                return []
            
            with Session(self.engine) as session:
                # Query conversations using SQLAlchemy ORM
                conversations_query = session.query(Conversation)\
                    .filter(Conversation.user_id == user_id)\
                    .order_by(Conversation.updated_at.desc())\
                    .all()
                
                conversations = []
                for conversation in conversations_query:
                    conversations.append({
                        'conversation_id': str(conversation.id),  # Convert UUID to string
                        'title': conversation.title,
                        'created_at': conversation.created_at.isoformat() if conversation.created_at else None,  # Convert datetime to ISO string
                        'updated_at': conversation.updated_at.isoformat() if conversation.updated_at else None,  # Convert datetime to ISO string
                        'organization': organization_info["organization"],
                        'organization_id': organization_info["organization_id"]
                    })
                
                return conversations
                
        except Exception as e:
            print(f"Error fetching conversations for user {user_id}: {e}")
            return []

    def get_conversation_messages(self, conversation_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        Retrieve all messages for a conversation from the database.
        
        Parameters:
        - conversation_id: UUID of the conversation
        
        Returns:
        - List of dictionaries containing message data
        """
        try:
            with Session(self.engine) as session:
                # First check if conversation exists
                conversation = session.query(Conversation).filter(Conversation.id == conversation_id).first()
                if not conversation:
                    print(f"Conversation {conversation_id} not found")
                    return []
                
                # Query messages using raw SQL since we don't have ORM model for messages yet
                result = session.execute(text("""
                    SELECT id, question, answer, sources, status, created_at, message_metadata
                    FROM messages 
                    WHERE conversation_id = :conversation_id 
                    ORDER BY created_at ASC
                """), {"conversation_id": conversation_id})
                
                messages = []
                for row in result:
                    messages.append({
                        'id': str(row[0]) if row[0] else None,  # Convert UUID to string
                        'question': row[1],
                        'answer': row[2],
                        'sources': row[3],
                        'status': row[4],
                        'created_at': row[5].isoformat() if row[5] else None,  # Convert datetime to ISO string
                        'message_metadata': row[6]  # Renamed from metadata
                    })
                
                return messages
                
        except Exception as e:
            print(f"Error fetching messages for conversation {conversation_id}: {e}")
            return []

    def create_new_conversation(self, conversation_id: uuid.UUID, user_id: uuid.UUID, title: str = None) -> bool:
        """
        Create a new conversation with the provided ID.
        
        Parameters:
        - conversation_id: UUID for the new conversation
        - user_id: UUID of the user
        - title: Optional title for the conversation
        
        Returns:
        - True if successful, False if failed
        """
        try:
            with Session(self.engine) as session:
                # Create new conversation with provided ID
                new_conversation = Conversation(
                    id=conversation_id,
                    user_id=user_id,
                    title=title
                )
                
                session.add(new_conversation)
                session.commit()
                
                print(f"Created new conversation {conversation_id} for user {user_id}")
                return True
                
        except Exception as e:
            print(f"Error creating new conversation {conversation_id} for user {user_id}: {e}")
            return False

    def save_message_to_conversation(
        self, 
        message_id: str,
        conversation_id: uuid.UUID, 
        question: str, 
        answer: str, 
        sources: List[str], 
        status: str,
        timestamp: datetime,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Save a message (question and answer) to a conversation.
        
        Parameters:
        - message_id: ID for the message
        - conversation_id: UUID of the conversation
        - question: The user's question
        - answer: The assistant's answer
        - sources: List of source references
        - status: Answer status (Answered, PartiallyAnswered, NotAnswered, OffTopic)
        - timestamp: When the message was created
        - metadata: Optional metadata dictionary
        
        Returns:
        - True if successful, False if failed
        """
        try:
            with Session(self.engine) as session:
                # Check if conversation exists
                conversation = session.query(Conversation).filter(Conversation.id == conversation_id).first()
                if not conversation:
                    print(f"Conversation {conversation_id} not found")
                    return False
                
                # Insert message using raw SQL with provided message_id
                session.execute(text("""
                    INSERT INTO messages (id, conversation_id, question, answer, sources, status, created_at, message_metadata)
                    VALUES (:id, :conversation_id, :question, :answer, :sources, :status, :created_at, :message_metadata)
                """), {
                    "id": message_id,
                    "conversation_id": conversation_id,
                    "question": question,
                    "answer": answer,
                    "sources": json.dumps(sources),  # Serialize sources list to JSON string
                    "status": status,
                    "created_at": timestamp,
                    "message_metadata": json.dumps(metadata) if metadata else None  # Renamed key from metadata
                })
                
                # Update conversation's updated_at timestamp
                session.execute(text("""
                    UPDATE conversations 
                    SET updated_at = :updated_at 
                    WHERE id = :conversation_id
                """), {
                    "updated_at": timestamp,
                    "conversation_id": conversation_id
                })
                
                session.commit()
                
                print(f"Saved message {message_id} to conversation {conversation_id}")
                return True
                
        except Exception as e:
            print(f"Error saving message {message_id} to conversation {conversation_id}: {e}")
            return False

    def get_message_details_by_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """
        Get message sources, organization, and organization_id by message UUID.
        
        Parameters:
        - message_id: UUID string of the message
        
        Returns:
        - Dictionary containing sources, organization, and organization_id, None if not found
        """
        try:
            with Session(self.engine) as session:
                # Query using SQLAlchemy ORM
                message = session.query(Message).filter(Message.id == message_id).first()
                
                if message:
                    # Extract organization info from message_metadata
                    current_message_metadata = message.message_metadata or {} # Renamed from metadata
                    organization = current_message_metadata.get('user_organization')
                    organization_id = current_message_metadata.get('user_organization_id')
                    
                    return {
                        'sources': message.sources,  # Already parsed as list from JSONB
                        'organization': organization,
                        'organization_id': organization_id
                    }
                return None
                
        except Exception as e:
            print(f"Error fetching message details for message {message_id}: {e}")
            return None

    def get_embeddings_metadata_by_organization_and_sources(
        self, 
        organization_id: str, 
        source_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get metadata from langchain_pg_embedding table by organization_id (collection_id) and source IDs.
        
        Parameters:
        - organization_id: Organization ID (used as collection_id)
        - source_ids: List of source IDs to filter by
        
        Returns:
        - List of dictionaries containing metadata for matching embeddings
        """
        try:
            with Session(self.engine) as session:
                # Query using SQLAlchemy ORM directly from langchain_pg_embedding
                results = session.query(
                    LangchainPgEmbedding.id,
                    LangchainPgEmbedding.cmetadata
                ).filter(
                    LangchainPgEmbedding.collection_id == organization_id,
                    LangchainPgEmbedding.id.in_(source_ids)
                ).all()
                
                embeddings_metadata = []
                for result in results:
                    embeddings_metadata.append({
                        'id': result[0],
                        'metadata': result[1]  # Already parsed as dict from JSONB
                    })
                
                return embeddings_metadata
                
        except Exception as e:
            print(f"Error fetching embeddings metadata for organization {organization_id} and sources {source_ids}: {e}")
            return []

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
