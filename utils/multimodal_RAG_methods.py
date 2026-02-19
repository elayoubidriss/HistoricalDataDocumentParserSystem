import base64
from importlib import metadata
import io
import logging
import re
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import Field

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_community.document_transformers import LongContextReorder
from langchain.schema import BaseRetriever, BaseStore, Document
from langchain.schema.messages import HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.vectorstore import VectorStore
from langchain.storage import InMemoryStore


from PIL import Image


from services.embeddings_services.embeddings_service import EmbeddingsService
from utils.vector_storage import get_inmemory_vectorstore


from chat_history import ChatHistory, QuestionAndAnswer
from utils.vector_storage import save_message_to_conversation


class MultiVectorRetrieverCustom(BaseRetriever):
    vectorstore: VectorStore
    filter: Optional[Dict] = None
    k: Optional[int] = 4
    reorder: Optional[bool] = True
    search_kwargs: dict = Field(default_factory=dict)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        sub_docs = self.vectorstore.similarity_search(query,k=self.k,filter=self.filter, **self.search_kwargs)
        
        if self.reorder:
            reordering = LongContextReorder()
            sub_docs = reordering.transform_documents(sub_docs)
        
        sub_docs = [Document(page_content=doc.metadata.get("raw_content"),metadata=doc.metadata) for doc in sub_docs]
        logging.info(f"retrieved docs from vector_db :  {len([d for d in sub_docs if d is not None])}")
        return [d for d in sub_docs if d is not None]

def are_all_elements_in_list(sub_list, main_list):
    if sub_list is None or  main_list is None :
        return False
    else:
        # logging.info(f"sub_list :  {sub_list}")
        # logging.info(f"main_list :  {main_list}")
        return all(element in main_list for element in sub_list)


class MultimodalRetrieverFromServer:
    def __init__(self, vectorstore, k=5, filter=None, reorder=True):
        self.vectorstore = vectorstore
        self.filter = filter
        self.reorder = reorder
        
        self.retriever = MultiVectorRetrieverCustom(
            vectorstore=self.vectorstore,
            filter=self.filter,
            k=k,
            reorder=self.reorder,
        )

class MultimodalRetrieverInMemory:
    def __init__(self, collection_name, reorder=True):
        self.store = InMemoryStore()
        self.id_key = "doc_id"
        self.reorder = reorder
        self.retriever = None
        self.vectorstore = get_inmemory_vectorstore(collection_name=collection_name)


    def add_documents(self, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={self.id_key: doc_ids[i]}) for i, s in enumerate(doc_summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_docs)
        self.retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    def create_retriever(self, text_summaries, texts, table_summaries, tables, image_summaries, images):
        self.retriever = MultiVectorRetrieverCustom(
            vectorstore=self.vectorstore, docstore=self.store, id_key="doc_id", k=6, reorder=self.reorder
        )

        if text_summaries:
            self.add_documents(text_summaries, texts)
        if table_summaries:
            self.add_documents(table_summaries, tables)
        if image_summaries:
            self.add_documents(image_summaries, images)

        return self.retriever


class ChatMultiModalRAGChain:
    def __init__(self, retriever, llm, openai_api_key: str, chat_history: ChatHistory, user_organization_info: dict = None):
        self.retriever = retriever
        self.openai_api_key = openai_api_key
        self.model = llm
        self.chat_history = chat_history
        self.user_organization_info = user_organization_info
        self.chain = self.create_chain(self.retriever, self.model)
        self.answer_language = "English"
        self.retrieved_documents : List = []
        self.response_last = ""


    def plt_img_base64(self, img_base64):
        """Display base64 encoded string as image"""
        # Create an HTML img tag with the base64 string as the source
        image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
        # Display the image by rendering the HTML
        # display(HTML(image_html))


    def split_image_text_types(self, docs):
        """
        Split base64-encoded images and texts
        """
        """
        self.wipe_folder('tests/request')
        """
        b64_images = []
        texts = []
        self.retrieved_documents = []


        for i,doc in enumerate(docs):
            # Check if the document is of type Document and extract page_content if so
            if isinstance(doc, Document):
                page_content = doc.page_content
                content_type = doc.metadata.get("content_type")

                if content_type == "text":
                    texts.append(page_content)

                else:
                    base64_string = page_content
                    b64_images.append(base64_string)
                    """
                    img_data = base64.b64decode(base64_string)
                    with open(f"tests/request/{i}_{content_type}.png", "wb") as f:
                        f.write(img_data)
                    """
                self.retrieved_documents.append(doc.metadata.get("id"))
                
        return {"images": b64_images, "texts": texts}
    

    def update_chat_history(self, questionAndAnswer: QuestionAndAnswer, message_id: str = None):
        print(f"DEBUG update_chat_history called with:")
        print(f"  message_id: {message_id}")
        print(f"  question: '{questionAndAnswer.question}'")
        print(f"  answer: '{questionAndAnswer.answer.answer}' (length: {len(questionAndAnswer.answer.answer) if questionAndAnswer.answer.answer else 'None'})")
        print(f"  sources: {questionAndAnswer.answer.sources}")
        print(f"  status: {questionAndAnswer.status}")
        print(f"  conversation_id: {self.chat_history.session_id}")
        
        self.chat_history.history.append(questionAndAnswer)
        
        # Save the message to the database if message_id is provided
        if message_id:
            print(f"DEBUG: About to call save_message_to_conversation")
            
            # Prepare metadata with user organization
            metadata = {}
            if self.user_organization_info:
                metadata["user_organization"] = self.user_organization_info.get("organization")
                metadata["user_organization_id"] = self.user_organization_info.get("organization_id")
            elif self.chat_history.organization_id:
                metadata["user_organization"] = self.chat_history.organization_id
            
            success = save_message_to_conversation(
                message_id=message_id,
                conversation_id=self.chat_history.session_id,
                question=questionAndAnswer.question,
                answer=questionAndAnswer.answer.answer,
                sources=questionAndAnswer.answer.sources,
                status=questionAndAnswer.status.value,
                timestamp=questionAndAnswer.timestamp,
                metadata=metadata
            )
            
            print(f"DEBUG: save_message_to_conversation returned: {success}")
            if success:
                logging.info(f"Successfully saved message {message_id} to conversation {self.chat_history.session_id}")
            else:
                logging.error(f"Failed to save message {message_id} to conversation {self.chat_history.session_id}")
        else:
            logging.warning("No message_id provided, message not saved to database")
        
        

    def invoke_with_memory(self, question, timestamp: datetime, message_id: str = None, verbose=False):
        if verbose:
            from langchain.callbacks.tracers import ConsoleCallbackHandler
            response = self.chain.invoke(question, config={"callbacks": [ConsoleCallbackHandler()]})
        else:
            response = self.chain.invoke(question)
        status = "Answered"
        self.update_chat_history(
            QuestionAndAnswer(timestamp=timestamp, question=question, answer=response, status=status), message_id=message_id
        )
        return response

    def img_prompt_func(self, data_dict):
        """
        Join the context into a single string
        """
        retrieved_texts = "\n".join(data_dict["context"]["texts"])
        logging.info(f"Retrieved texts : {len(retrieved_texts)}")
        history_text = "\n".join([f"User: {q.question}\nAssistant: {q.answer.answer}" for q in self.chat_history.history])
        
        messages = []
        image_messages = []

        # Adding image(s) to the messages if present

        if data_dict["context"]["images"]:
            for image in data_dict["context"]["images"]:
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"},
                }
                image_messages.append(image_message)
        
        if len(image_messages) > 3 : 
            image_messages = image_messages[:3]
        
        logging.info(f"Answering in : {self.answer_language}")
        
        text_message = {
            "type": "text",
            "text": f""" You are an assistant with access to multiple documents. The following conversation history and user questions all relate to the content of those documents:

                    Conversation history: ####
                    {history_text}
                    ####

                    You must consider the conversation history to answer the user's question. You are a helpful assistant who provides friendly, professional answers and follow-up questions. Always answer as helpfully as possible, *without* providing unnecessary details or information the user didn’t request. 

                    If the user's question is unclear or factually incoherent, do your best to answer it. Do not offer a summary of the conversation or the document unless the user explicitly asks for one. If you lack the information to answer a question, admit it rather than guessing.

                    You will be given text, tables, and/or images (charts/graphs) that you may rely on to form your response. Do not reference or cite the document, and do not mention that your answer is based on the document. Focus solely on *tangible, verifiable information* relevant to the user's question.

                    Always provide your answer in markdown format. (Note: you will be penalized if you do not use markdown.)

                    User's query:
                    {data_dict['question']}

                    Documents:
                    ####
                    {retrieved_texts}
                    ####

                    You must answer in {self.answer_language}, strictly using {self.answer_language}. Do not summarize the document or conversation unless it is explicitly requested by the user. Simply provide a direct answer or clarification to the user's query or input.

                    Answer:
                    """,
        }

        messages.append(text_message)
        messages.extend(image_messages)
        
        """
        import os, json
        with open("tests/data.json", "w") as f: json.dump(messages, f, indent=2)
        """

        return [HumanMessage(content=messages)]

    def get_relevant_documents(self, query, limit=6):
        docs = self.retriever.get_relevant_documents(query, limit=limit)
        return docs

    def create_chain(self, retriever, model):
        chain = (
            {
                "context": retriever | RunnableLambda(self.split_image_text_types),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self.img_prompt_func)
            | model
            | StrOutputParser()
        )
        return chain


class MultiModalRAGChain:
    def __init__(self, retriever, llm, openai_api_key = ""):
        self.retriever = retriever
        self.openai_api_key = openai_api_key
        self.model = llm
        self.chain = self.create_chain(self.retriever, self.model)


    def plt_img_base64(self, img_base64):
        """Display base64 encoded string as image"""
        # Create an HTML img tag with the base64 string as the source
        image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
        # Display the image by rendering the HTML
        # display(HTML(image_html))

    def looks_like_base64(self, sb):
        """Check if the string looks like base64"""
        return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

    def is_image_data(self, b64data):
        """
        Check if the base64 data is an image by looking at the start of the data
        """
        image_signatures = {
            b"\xff\xd8\xff": "jpg",
            b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
            b"\x47\x49\x46\x38": "gif",
            b"\x52\x49\x46\x46": "webp",
        }
        try:
            header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
            for sig, format in image_signatures.items():
                if header.startswith(sig):
                    return True
            return False
        except Exception:
            return False

    def resize_base64_image(self, base64_string, size=(128, 128)):
        """
        Resize an image encoded as a Base64 string
        """
        # Decode the Base64 string
        img_data = base64.b64decode(base64_string)
        img = Image.open(io.BytesIO(img_data))

        # Resize the image
        resized_img = img.resize(size, Image.LANCZOS)

        # Save the resized image to a bytes buffer
        buffered = io.BytesIO()
        resized_img.save(buffered, format=img.format)

        # Encode the resized image to Base64
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def split_image_text_types(self, docs):
        """
        Split base64-encoded images and texts
        """
        b64_images = []
        texts = []
        for i,doc in enumerate(docs):
            # Check if the document is of type Document and extract page_content if so
            if isinstance(doc, Document):
                page_content = doc.page_content
                content_type = doc.metadata.get("content_type")
                logging.info(content_type)
            if content_type == "text":
                texts.append(page_content)
            else:
                base64_string = page_content
                base64_image = self.resize_base64_image(page_content, size=(1300, 600))
                b64_images.append(base64_image)
                
        return {"images": b64_images, "texts": texts}

    def img_prompt_func(self, data_dict):
        """
        Join the context into a single string
        """
        formatted_texts = "\n".join(data_dict["context"]["texts"])
        messages = []

        # Adding image(s) to the messages if present
        if data_dict["context"]["images"]:
            for image in data_dict["context"]["images"]:
                image_message = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
                messages.append(image_message)

        text_message = {
            "type": "text",
            "text": (
                "You are an assistant tasked with extracting information.\n"
                "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
                "Use this information to provide a precise response related to the user question. \n"
                "DO NOT reference unavailable details, unless they are explicitly mentioned in the documents. Focus on tangible and verifiable information.\n"
                "DO NOT add information that is not required or relevant to the User-provided question.\n"
                f"logging.infoUser-provided question: {data_dict['question']}\n\n"
                "Text and / or tables:\n"
                f"{formatted_texts}"
            ),
        }
        messages.append(text_message)
        return [HumanMessage(content=messages)]

    def get_relevant_documents(self, query, limit=6):
        docs = self.retriever.get_relevant_documents(query, limit=limit)
        return docs

    def create_chain(self, retriever, model):
        chain = (
            {
                "context": retriever | RunnableLambda(self.split_image_text_types),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self.img_prompt_func)
            | model
            | StrOutputParser()
        )
        return chain
