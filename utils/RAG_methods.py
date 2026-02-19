
import uuid
from typing import Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import LLMChain, RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import NumberedListOutputParser
from pydantic import Field
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema import BaseRetriever, BaseStore, Document
from langchain.schema.document import Document
from langchain.schema.vectorstore import VectorStore
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.vector_storage import get_server_vectorstore

"""
Implemented methods :
- Questions generation model
- Information retrieval
- Summarizing tasks
"""


class DocumentLoaderAndSplitter:
    def __init__(self, file_paths=[None], link=None, chunk_size=5000):
        self.file_paths = file_paths
        self.chunk_size = chunk_size
        self.link = link

    def load_and_split_from_disc_txt(self):
        docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size)
        for path in self.file_paths:
            loader = TextLoader(path)
            loaded_docs = loader.load()
            docs.extend(loaded_docs)

        docs = text_splitter.split_documents(docs)
        return docs

    def load_and_split_url(self):
        docs = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size)
        loader = WebBaseLoader(self.link)
        loaded_docs = loader.load()
        docs.extend(loaded_docs)

        docs = text_splitter.split_documents(docs)
        return docs

    def split_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size)
        docs = text_splitter.split_documents(docs)
        return docs

    def get_ids(self, docs):
        doc_ids = [str(uuid.uuid4()) for _ in docs]
        return doc_ids


class MultiVectorRetrieverCustom(BaseRetriever):
    vectorstore: VectorStore
    docstore: BaseStore[str, Document]
    id_key: str = "doc_id"
    filter: Optional[Dict] = None

    search_kwargs: dict = Field(default_factory=dict)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        sub_docs = self.vectorstore.similarity_search(query, k=4, filter=self.filter, **self.search_kwargs)
        # We do this to maintain the order of the ids that are returned
        ids = []
        for d in sub_docs:
            print(d)
            if d.metadata[self.id_key] not in ids:
                ids.append(d.metadata[self.id_key])
        docs = self.docstore.mget(ids)
        print(ids)
        return [d for d in docs if d is not None]


class CustomGenerationModel:
    def __init__(self, openai_api_key, model_name="gpt-3.5-turbo-16k"):
        self.model = ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key)
        self.model_name = model_name


class QuestionGenerationModel:
    def __init__(self, docs, openai_api_key, model_name="gpt-3.5-turbo-16k"):
        self.docs = docs
        self.llm = ChatOpenAI(model_name=model_name, openai_api_key=openai_api_key)
        self.chain = LLMChain.from_string(
            llm=self.llm,
            template="""
                Generate a numbered list of 3 hypothetical questions that the below document could be used to answer:
                {doc}
            """,
        )
        self.chain.verbose = True
        self.chain.output_parser = NumberedListOutputParser()

    def generate_questions(self):
        doc_ids = [str(uuid.uuid4()) for _ in self.docs]
        id_key = "doc_id"
        question_docs = []
        for i, doc in enumerate(self.docs):
            result = self.chain.run(doc.page_content)
            question_docs.extend([Document(page_content=s, metadata={id_key: doc_ids[i]}) for s in result])
        return question_docs


class RetrieverInMemory:
    def __init__(self, llm, embedding_function, chunk_size=400):
        self.vectorstore = get_server_vectorstore(collection_name="link_wiki",embedding_function=embedding_function)
        self.store = InMemoryStore()
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key="doc_id",
        )
        self.chunk_size = chunk_size
        self.llm = llm

    def index_documents(self, docs):
        doc_ids = [str(uuid.uuid4()) for _ in docs]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size)
        sub_docs = []
        for i, doc in enumerate(docs):
            _id = doc_ids[i]
            _sub_docs = text_splitter.split_documents([doc])
            for _doc in _sub_docs:
                _doc.metadata["doc_id"] = _id
            sub_docs.extend(_sub_docs)

        self.retriever.vectorstore.add_documents(sub_docs)
        self.retriever.docstore.mset(list(zip(doc_ids, docs)))

    def build_chain(self):
        llm = self.llm
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        chain = RetrievalQA.from_chain_type(llm=llm, retriever=self.retriever, verbose=False, memory=memory)
        return chain


class RetrieverFromServer:
    def __init__(self, llm, vectorstore, filter=None):
        self.vectorstore = vectorstore
        self.store = InMemoryStore()
        self.filter = filter
        self.retriever = MultiVectorRetrieverCustom(
            vectorstore=self.vectorstore, docstore=self.store, id_key="doc_id", filter=self.filter
        )
        self.llm = llm

    def retriever_docstore(self, docs, doc_ids):
        self.retriever.docstore.mset(list(zip(doc_ids, docs)))

    def build_chain(self):
        llm = self.llm
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        chain = RetrievalQA.from_chain_type(llm=llm, retriever=self.retriever, verbose=False, memory=memory)
        return chain
