import logging
import re
from typing import Any, List
from langdetect import DetectorFactory, detect

from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Document

from utils.multimodal_RAG_methods import MultimodalRetrieverFromServer
from chat_history import ChatHistory

_follow_up_question_template = """Generate a numbered list of the top {number_questions} relevant follow-up question based on the following conversation and context. The questions must have their answer in the given context, if the question doesn't have the answer in the context below DO NOT return it, this is very important. Return the questions ONLY!
\n########################################################\n
Chat History: \n
{chat_history}
\n########################################################\n
context: \n
{context}
\n########################################################\n
You will answer in {answer_language}, it is very important that you answer in {answer_language}. \n
follow-up questions:"""

FOLLOW_UP_QUESTION_PROMPT = PromptTemplate.from_template(_follow_up_question_template)


_condense_question_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. If the follow up input is not a question, return it as it is without any explanation. you will be penalized if you give any explanation.

Chat History: \n
{chat_history} \n
Follow Up Input: {question} \n
You will answer in the same language of the Follow Up Input, it is very important that you answer in the same language of the Follow Up Input. \n
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_condense_question_template)

_correct_question_template = """Given the following user query, you will correct it if there is a mistake. If the input has no mistake, return it as it is without any explanation. you will be penalized if you give any explanation.

User query: {question} \n
You will answer in the same language of the User query, it is very important that you answer in the same language of the User query. \n
Corrected query:"""
CORRECT_QUESTION_PROMPT = PromptTemplate.from_template(_correct_question_template)

def chunk_message(message, chunk_size=50):
    return [message[i:i+chunk_size] for i in range(0, len(message), chunk_size)]



def check_filenames_in_vector_db(ids_vector_db: List[str], document_ids: List[str]) -> None:
    """
    Checks and logs whether each document ID in document_ids is present in ids_vector_db.

    Parameters:
    - ids_vector_db (List[str]): A list of complex IDs that may contain document IDs as substrings.
    - document_ids (List[str]): A list of document IDs to check against ids_vector_db.
    """
    for document_id in document_ids:
        is_present = any(document_id in vector_db_id for vector_db_id in ids_vector_db)

        if is_present:
            logging.info(f"{document_id} is present in vector_db.")
        else:
            logging.info(f"{document_id} is not present in vector_db.")


def check_filenames_in_docs(docs: List[Document], filenames: List[str]) -> None:
    """
    Checks if the specified filenames are present in the list of documents and prints the result.

    Parameters:
    - docs (List[Document]): A list of Document objects.
    - filenames (List[str]): A list of filenames to check against the documents' metadata.
    """
    # Extract 'source_file' values from each document's metadata
    extracted_filenames = [doc.metadata["source_file"] for doc in docs]

    # Check each filename against the extracted filenames
    for filename in filenames:
        if filename in extracted_filenames:
            logging.info(f"{filename} is present in datastore.")
        else:
            logging.info(f"{filename} is not present in datastore.")


def extract_and_clean_questions(suggested_queries: str) -> List[str]:
    """
    Extracts and cleans questions from a multiline string containing numbered questions.

    :param suggested_queries: Multiline string containing questions, each possibly prefixed with numbering.
    :return: List of cleaned questions without numbering.
    """
    lines = suggested_queries.split("\n")

    general_numbering_pattern = re.compile(r"^[0-9a-zA-Z]+[.\-)]\s*")

    question_lines = [line for line in lines if general_numbering_pattern.match(line)]

    cleaned_questions = [general_numbering_pattern.sub("", question.strip()) for question in question_lines]

    return cleaned_questions


async def condense_user_query(llm: Any, chat_history: ChatHistory, answer_language: str, user_query: str) -> str:
    """
    Condenses a user query based on the chat history and a specific prompt.
    :param chat_history: List of tuples representing the chat history, where each tuple is (user_query, assistant_response).
    :param user_query: The current user query to be condensed.
    :return: The condensed user query as a string.
    """
    # from utils.query_data import CONDENSE_QUESTION_PROMPT
    # from utils.query_data import CONDENSE_QUESTION_PROMPT

    if len(chat_history.history) == 0:
        condense_question_chain = LLMChain(
            llm=llm,
            prompt=CORRECT_QUESTION_PROMPT,
            verbose=False,
        )

        condense_question_response = await condense_question_chain.ainvoke(
            {"question": user_query, "answer_language": answer_language}
        )

    else:
        condense_question_chain = LLMChain(
            llm=llm,
            prompt=CONDENSE_QUESTION_PROMPT,
            verbose=False,
        )

        condense_question_response = await condense_question_chain.ainvoke(
            {"chat_history": str(chat_history), "question": user_query, "answer_language": answer_language}
        )

    condensed_query = condense_question_response["text"]

    return condensed_query


async def generate_follow_up_question(
    llm: Any, retrieval_model: MultimodalRetrieverFromServer, chat_history: ChatHistory, language: str, number_questions: int=2
) -> List[str]:
    """
    Generates a follow-up question based on the given chat history using a RAG model.

    :param chat_history: List of tuples representing the chat history, where each tuple is (user_query, assistant_response).
    :param chain: The RAG model chain used for generation.
    :return: A generated follow-up question as a string.

    """
    follow_up_question_chain = LLMChain(
        llm=llm,
        prompt=FOLLOW_UP_QUESTION_PROMPT,
        verbose=False,
    )

    """
    last_query = chat_history.history[-1].question
    docs = await retrieval_model.retriever.aget_relevant_documents(
        last_query,
    )
    context_text = "\n".join([doc.page_content for doc in docs])

    

    logging.info({"chat_history": str(chat_history), "context": context_text, "number_questions": number_questions,"answer_language": language})

    follow_up_question_response = await follow_up_question_chain.ainvoke(
        {"chat_history": str(chat_history), "context": context_text, "number_questions": number_questions,"answer_language": language}
    )
"""

    follow_up_question_response = await follow_up_question_chain.ainvoke(
        {"chat_history": str(chat_history), "context":"", "number_questions": number_questions,"answer_language": language}
    )

    follow_up_question = follow_up_question_response["text"]
    suggested_queries = extract_and_clean_questions(follow_up_question)

    return suggested_queries

def detect_document_language(list_of_texts):
    DetectorFactory.seed = 0
    combined_text = " ".join(list_of_texts)
    print("///////////////////////////////////////////////////////////")
    print(combined_text)
    detected_language = detect(combined_text)
    authorized_langs = ["en","fr", "es", "ja", "zh-cn", "zh-tw"]
    if detected_language not in authorized_langs:
        detected_language = "en"
    return detected_language