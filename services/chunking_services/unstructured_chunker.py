from typing import List

from services.chunking_services.base_chunker import BaseChunker
from models.settings import BaseChunkerSettings


import base64
import os
import logging

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from unstructured.partition.text import partition_text
from unstructured.partition.pdf import partition_pdf
from langchain_community.document_loaders import CSVLoader, Docx2txtLoader
from utils.RAG_methods import DocumentLoaderAndSplitter
from utils.loaders import UnstructuredPowerPointLoader
from services.llm_services.llm_service import LLMService
from services.chunking_services.utils.doc_processor import crop_and_encode_image,chunk_text,process_coordinates,split_texts_by_tokens,CoordType

class UnstructuredChunker(BaseChunker):
    """A simple chunker that splits text into fixed-size chunks."""

    def __init__(self, settings: BaseChunkerSettings):
        self.settings = settings
        self.max_tokens = settings.max_tokens
        self.overlap_tokens=settings.overlap_tokens
        self.tokenization_model = settings.tokenization_model
        self.crop_extension = settings.crop_extension
        self.use_custom_chunker = settings.use_custom_chunker
        print(f"Initializing Unstructured Chunker with settings: {self.settings.json()}")
        

    def process_text(self, file_path:str,file_name:str, summarize_texts: bool = True, extract_tables: bool = True, extract_images : bool = True):
        if file_name.endswith(".pdf"):
            logging.info("Extracting Elements")
            raw_pdf_elements = self.extract_pdf_elements(file_path=file_path,file_name=file_name)
            logging.info("Categorizing Elements")
            texts, tables, images = self.categorize_elements(raw_pdf_elements,extract_tables=extract_tables,extract_images=extract_images)
            logging.info("Processing Coordinates")
            texts = process_coordinates(texts,CoordType.UNSTRUCTURED,file_path,file_name)
            tables = process_coordinates(tables,CoordType.UNSTRUCTURED,file_path,file_name)
            images = process_coordinates(images,CoordType.UNSTRUCTURED,file_path,file_name)
            logging.info("Splitting Texts By Tokens")
            if self.use_custom_chunker:
                texts = split_texts_by_tokens(elements=texts,max_tokens=self.max_tokens,overlap_tokens=self.overlap_tokens,tokenization_model=self.tokenization_model)
            logging.info("Croping Images")
            images = [[image[0],crop_and_encode_image(os.path.join(file_path, file_name),image[2],self.crop_extension,image[3],300,True),image[2],image[3]] for image in images]
            tables = [[table[0],crop_and_encode_image(os.path.join(file_path, file_name),table[2],self.crop_extension,table[3],300,True),table[2],table[3]] for table in tables]
            logging.info("Generating summaries")
            texts_to_summarize = [element[1] for element in texts]
            tables_to_summarize = [element[1] for element in tables]
            images_to_summarize = [element[1] for element in images]
            text_summaries, table_summaries, image_summaries = [],[],[]
            text_summaries, table_summaries, image_summaries = self.generate_summaries(
                texts_to_summarize, tables_to_summarize, images_to_summarize, summarize_texts=summarize_texts, summarize_tables=extract_tables, summarize_images=extract_images
            )
            
            return text_summaries, table_summaries, image_summaries, texts, tables, images
        elif file_name.endswith(".txt"):
            with open(os.path.join(file_path, file_name), "r") as f:
                text = f.read()
            text_elements = partition_text(text)
            return text_elements, [], [], []
        elif file_name.endswith(".CSV") or file_name.endswith(".csv"):
            print("extracting pdf elements using CSVLoader...")
            chunk_size = 10
            loader = CSVLoader(file_path=os.path.join(file_path, file_name))
            data = loader.load()
            texts = [doc.page_content for doc in data]
            texts =  ['\n'.join(texts[i:i+chunk_size]) for i in range(0, len(texts), chunk_size)]
            tables = []
            images = []
            print("Generating summaries ...")
            text_summaries, table_summaries, image_summaries = self.generate_summaries(
                texts, tables, images, summarize_texts=summarize_texts, summarize_tables=extract_tables, summarize_images=extract_images
            )
            return text_summaries, table_summaries, image_summaries, texts, tables, images
        elif file_name.endswith(".docx"):
            loader = Docx2txtLoader(os.path.join(file_path, file_name))
            data = loader.load()
            texts = [doc.page_content for doc in data]
            joined_texts = " ".join(texts)
            texts =  self.chunk(max_tokens=self.max_tokens,overlap_tokens=self.overlap_tokens,text=joined_texts)
            tables = []
            images = []
            print("Generating summaries ...")
            text_summaries, table_summaries, image_summaries = self.generate_summaries(
                texts, tables, images, summarize_texts=summarize_texts, summarize_tables=extract_tables, summarize_images=extract_images
            )
            return text_summaries, table_summaries, image_summaries, texts, tables, images
        elif file_name.endswith(".ppt") or file_name.endswith(".pptx"):
            loader = UnstructuredPowerPointLoader(os.path.join(file_path, file_name),strategy="hi_res")
            data = loader.load()
            texts = [doc.page_content for doc in data]
            joined_texts = " ".join(texts)
            texts = self.chunk(max_tokens=self.max_tokens,overlap_tokens=self.overlap_tokens,text=joined_texts)
            tables = []
            images = []
            print("Generating summaries ...")
            text_summaries, table_summaries, image_summaries = self.generate_summaries(
                texts, tables, images, summarize_texts=summarize_texts, summarize_tables=extract_tables, summarize_images=extract_images
            )
            return text_summaries, table_summaries, image_summaries, texts, tables, images
        elif file_name.startswith(("http://", "https://", "ftp://")):
            document_loader = DocumentLoaderAndSplitter(link=file_name, chunk_size=1000)
            docs = document_loader.load_and_split_url()

            texts = [doc.page_content for doc in docs]
            tables = []
            images = []
            print("Generating summaries ...")
            text_summaries, table_summaries, image_summaries = self.generate_summaries(
                texts, tables, images, summarize_texts=summarize_texts, summarize_tables=extract_tables, summarize_images=extract_images
            )
            return text_summaries, table_summaries, image_summaries, texts, tables, images

        else:
            return [], [], [], []

    #####################################################################################""   
    # Helper Functions 
        
    def extract_pdf_elements(self,file_path:str,file_name:str):

        return partition_pdf(
            filename=os.path.join(file_path, file_name),
            strategy="hi_res",
            chunking_strategy="by_title",                       
            extract_image_block_types=["Image","Table"],          
            extract_image_block_to_payload=True,
            max_characters=4000,        
            )
    
    
    def categorize_elements(self, raw_pdf_elements,extract_images,extract_tables):
        elements_to_ignore = ["UncategorizedText","FigureCaption"]
        elements_with_base64 = ["Image","Table"]
        texts = []
        images = []
        tables = []
        for raw_element in raw_pdf_elements : 
            raw_element = raw_element.metadata
            orig_elements = raw_element.orig_elements
            for element in orig_elements : 
                element_dict = element.to_dict()
                element_type = element_dict.get("type")
                if element_type in elements_to_ignore : 
                    continue

                if element_type not in elements_with_base64:
                    element_text = element_dict.get("text")
                else :
                    element_text = None
                
                element_metadata = element_dict.get("metadata")
                element_base64string = element_metadata.get("image_base64")
                element_coordinates = element_metadata.get("coordinates")["points"]
                element_page_number = element_metadata.get("page_number")

                if element_type == "Image" and extract_images:
                    images.append([element_type, element_base64string, element_coordinates, element_page_number,])
                elif element_type == "Table" and extract_tables:
                    tables.append([element_type, element_base64string, element_coordinates, element_page_number])
                else:
                    texts.append(["Text",element_text,element_coordinates,element_page_number])
                
        return texts,tables,images
    
    def chunk(self, max_tokens, overlap_tokens, text) -> List[str]:
        return chunk_text(max_tokens, overlap_tokens, text, self.tokenization_model)
    
    

    def generate_summaries(self, texts, tables, images, summarize_texts: bool = True, summarize_tables: bool = True, summarize_images: bool = True):

        text_summaries = []
        table_summaries = []
        image_summaries = []

        # Text prompt
        prompt_text = """You are an assistant tasked with summarizing text for retrieval. 
        These summaries will be embedded and used to retrieve the raw text elements. 
        Give a concise summary of the text that is well optimized for retrieval.
        Start your response with "Summary:"
        text: {element} """
        prompt_template_text = ChatPromptTemplate.from_template(prompt_text)

        # Table prompt
        prompt_table = """You are an assistant tasked with summarizing table images for retrieval. 
        These summaries will be embedded and used to retrieve the raw table elements. 
        Give a concise summary of the image that is well optimized for retrieval. 
        Start your response with "Summary:"
        """

        prompt_template_table = ChatPromptTemplate.from_messages([
            ("human", [
                {"type": "text", "text": prompt_table},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{element}"}}
            ])
        ])
        
        
        # Image prompt
        prompt_image = """You are an assistant tasked with summarizing images for retrieval. 
        These summaries will be embedded and used to retrieve the raw image elements. 
        Give a concise summary of the image that is well optimized for retrieval. 
        Start your response with "Summary:"
        """
        
        prompt_template_image = ChatPromptTemplate.from_messages([
            ("human", [
                {"type": "text", "text": prompt_image},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{element}"}}
            ])
        ])
        
        # Use the same model for all - assuming it supports vision
        model = LLMService.get().get_langchain_chat()
        
        # Create chains
        summarize_chain_text = {"element": lambda x: x} | prompt_template_text | model | StrOutputParser()
        summarize_chain_table = {"element": lambda x: x} | prompt_template_table | model | StrOutputParser()
        summarize_chain_image = {"element": lambda x: x} | prompt_template_image | model | StrOutputParser()

        # Process each type
        if texts and summarize_texts:
            text_summaries = summarize_chain_text.batch(texts, {"max_concurrency": 5})

        if tables and summarize_tables:
            table_summaries = summarize_chain_table.batch(tables, {"max_concurrency": 5})
        
        if images and summarize_images:
            image_summaries = summarize_chain_image.batch(images, {"max_concurrency": 3})

        return text_summaries, table_summaries, image_summaries
    


    
