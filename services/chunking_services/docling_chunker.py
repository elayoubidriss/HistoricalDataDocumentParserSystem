from typing import List

from services.chunking_services.base_chunker import BaseChunker
from models.settings import BaseChunkerSettings


import base64
import os
import logging

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from services.llm_services.llm_service import LLMService
from services.chunking_services.utils.doc_processor import crop_and_encode_image,chunk_text,process_coordinates,split_texts_by_tokens,get_tokenizer,CoordType,remove_nested, extract_unknowns_as_images,extend_coordinates

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.chunking import HybridChunker


class DoclingChunker(BaseChunker):
    """A simple chunker that splits text into fixed-size chunks."""

    def __init__(self, settings: BaseChunkerSettings):
        self.settings = settings
        self.chunker_type = settings.type
        self.max_tokens = settings.max_tokens
        self.overlap_tokens=settings.overlap_tokens
        self.tokenization_model = settings.tokenization_model
        self.crop_extension = settings.crop_extension
        self.use_custom_chunker = settings.use_custom_chunker

        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.AUTO
        )
        self.pipeline_options.images_scale = 4.16
        self.pipeline_options.generate_picture_images = True
        self.pipeline_options.do_picture_classification = True
        self.pipeline_options.generate_table_images = True

        

        print(f"Initializing Docling Chunker with settings: {self.settings.json()}")

    def process_text(self, file_path: str, file_name: str, summarize_texts: bool = True, extract_tables: bool = True,
                     extract_images: bool = True):
        if file_name.endswith(".pdf"):

            converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=self.pipeline_options)})
            print("convert")
            doc = converter.convert(os.path.join(file_path, file_name)).document
            print("categorize_elements")
            texts, images = self.categorize_elements(doc,extract_tables=extract_tables,extract_images=extract_images)
            print("process_coordinates")
            texts = process_coordinates(texts,CoordType.DOCLING,file_path,file_name)
            images = process_coordinates(images,CoordType.DOCLING,file_path,file_name)
            print("extract_unknowns_as_images")
            texts,images_to_add = extract_unknowns_as_images(texts=texts,pdf_path=os.path.join(file_path, file_name))
            images.extend(images_to_add)
            print("extend_coordinates")
            images = extend_coordinates(images,file_path,file_name,self.crop_extension)
            print("remove_nested")
            images = remove_nested(images)
            print("split_texts_by_tokens")
            if self.use_custom_chunker and self.chunker_type != 'docling_chunker':
                print("Using custom chunker")
                texts = split_texts_by_tokens(elements=texts,max_tokens=self.max_tokens,overlap_tokens=self.overlap_tokens)
            print("crop_and_encode_image")
            images = [[image[0],crop_and_encode_image(os.path.join(file_path, file_name),image[2],image[3]),image[2],image[3]] for i,image in enumerate(images)]
            print("separating_images_and_tables")
            tables = [item for item in images if item[0] == "Table"]
            images = [item for item in images if item[0] == "Image"]
            logging.info("Generating summaries ...")
            texts_to_summarize = [element[1] for element in texts]
            tables_to_summarize = [element[1] for element in tables]
            images_to_summarize = [element[1] for element in images]
            text_summaries, table_summaries, image_summaries = [], [], []

            if summarize_texts or extract_tables or extract_images:
                logging.info("Generating summaries ...")
                text_summaries, table_summaries, image_summaries = self.generate_summaries(
                    texts_to_summarize, tables_to_summarize, images_to_summarize,
                    summarize_texts=summarize_texts,
                    summarize_tables=extract_tables,
                    summarize_images=extract_images
                )
            else:
                logging.info("Skipping summaries generation")

            return text_summaries, table_summaries, image_summaries, texts, tables, images
        else:
            return [], [], [], []

    #####################################################################################""   
    # Helper Functions 

    def classify_pic(self, pic, confidence_threshold=0.8):
        types_to_conserve = ["pie_chart","bar_chart","map","flow_chart","line_chart","other"]
        conserve_type = False
        confidence = 0
        picture_classification = []
        current_element = 0
        list_length = len(pic.annotations[0].predicted_classes)
        while confidence <= confidence_threshold and current_element < list_length: 
            picture_classification.append(pic.annotations[0].predicted_classes[current_element].class_name)
            confidence +=  pic.annotations[0].predicted_classes[current_element].confidence
            current_element += 1
        for type in types_to_conserve:
            if type in picture_classification:
                conserve_type = True
                break
        return conserve_type

    def get_pictures_to_conserve(self, pictures):
        picture_to_conserve = []
        for pic in pictures : 
            if self.classify_pic(pic) : 
                picture_to_conserve.append(pic)
        return picture_to_conserve
    
    def categorize_elements(self, doc, extract_tables, extract_images):
        texts = []
        images = []
        texts = self.get_chunked_texts(tokenization_model=self.tokenization_model, max_tokens=self.max_tokens, doc=doc)
        if extract_tables :
            images.extend([["Table","uri_placeholder",table.prov[0].bbox,table.prov[0].page_no] for table in doc.tables])
        if extract_images:
            images_to_conserve = self.get_pictures_to_conserve(doc.pictures)
            images.extend([["Image","uri_placeholder",image.prov[0].bbox,image.prov[0].page_no] for image in images_to_conserve])
        return texts, images
    
    def chunk(self, max_tokens, overlap_tokens, text) -> List[str]:
        return chunk_text(max_tokens, overlap_tokens, text,self.tokenization_model)
    
    def get_chunked_texts(self,tokenization_model,max_tokens,doc):
        tokenizer = get_tokenizer(tokenization_model)
        chunker = HybridChunker(tokenizer=tokenizer,max_tokens=max_tokens,merge_peers=True)
        chunk_iter = chunker.chunk(doc)
        texts = []
        for i,chunk in enumerate(chunk_iter):
            item_type = chunk.meta.doc_items[0].label
            if  item_type != "table" and item_type != "image" and item_type != "document_index":
                text = chunker.contextualize(chunk=chunk)
                metadata = chunk.meta.doc_items[0].prov[0]
                bbox = metadata.bbox
                page_no = metadata.page_no
                texts.append(["Text",text,bbox,page_no])
        return texts
            
    
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
        prompt_table = """
        You are an assistant tasked with summarizing table images for retrieval.
        These summaries will be embedded and used to retrieve the raw table image elements.

        Focus on the main table image.
        Ignore any extended canvas, crops, borders, watermarks, or incidental text/imagery
        unless it is essential for understanding the table image.

        Give a concise summary that is well-optimized for retrieval.
        Start your response with "Summary:"
        """

        prompt_template_table = ChatPromptTemplate.from_messages([
            ("human", [
                {"type": "text", "text": prompt_table},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,{element}"}}
            ])
        ])
        
        
        # Image prompt
        prompt_image = """
        You are an assistant tasked with summarizing images for retrieval.
        These summaries will be embedded and used to retrieve the raw image elements.

        Focus on the main image.
        Ignore any extended canvas, crops, borders, watermarks, or incidental text/imagery
        unless it is essential for understanding the image.

        Give a concise summary that is well-optimized for retrieval.
        Start your response with "Summary:"
        """
        
        prompt_template_image = ChatPromptTemplate.from_messages([
            ("human", [
                {"type": "text", "text": prompt_image},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,{element}"}}
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
    


    
