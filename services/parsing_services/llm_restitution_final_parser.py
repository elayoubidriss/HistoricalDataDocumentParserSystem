import re
import os
import logging
import json
from typing import List, Dict
from base_parser import BaseParser
from models.settings import RestitutionFinalParserSettings
from services.chunking_services.chunking_service import ChunkerService
from services.llm_services.llm_service import LLMService

# Configure logging
logger = logging.getLogger(__name__)


class RestitutionFinalParser(BaseParser):
    """Parser for 'Restitution Final' presentation files using LLM extraction."""

    def __init__(self, settings: RestitutionFinalParserSettings):
        super().__init__(settings)
        self.settings = settings

    def parse(self, file_path: str) -> List[Dict[str, str]]:
        """
        Extract data from Restitution Final presentation files using LLM extraction.
        """
        try:
            # For PPT files, we need to use a different approach than the base parser
            if not (file_path.endswith('.ppt') or file_path.endswith('.pptx')):
                logger.error(f"Unsupported file format for RestitutionFinalParser: {file_path}")
                return []

            # Process presentation with the chunker service
            dir_path = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)

            # Use the chunker to extract text from the presentation
            text_summaries, table_summaries, image_summaries, texts, tables, images = ChunkerService.get().process_text(
                dir_path, file_name,
                summarize_texts=False,
                extract_tables=False,
                extract_images=False
            )

            # For PPT files, we need to handle the text extraction differently
            # The texts array contains slide content, we need to identify the relevant slides
            full_text = self._extract_restitution_content(texts)

            if not full_text:
                logger.warning(f"No restitution content found in {file_path}")
                return []

            # Use LLM for extraction
            return self._extract_with_llm(full_text)

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            raise

    def _extract_restitution_content(self, texts: List) -> str:
        """
        Extract the restitution content from the presentation slides.
        This method identifies the slide(s) containing constats and related information.
        """
        restitution_text = ""

        # Look for slides that contain constat-related keywords
        keywords = ["constat n°", "CONSTAT N°", "CONSTAT",
                    "constats", "risques", "recommandations", "cotation des risques",
                    "nombre de constats", "nombre de recommandations", "NOMBRE DE CONSTATS", "NOMBRE DE RECOMMANDATIONS"]

        for text in texts:
            slide_content = text[1] if len(text) > 1 else ""
            slide_content_lower = slide_content.lower()

            # Check if this slide contains any of our keywords
            if any(keyword in slide_content_lower for keyword in keywords):
                restitution_text += slide_content + "\n\n"

        return restitution_text

    def _create_extraction_prompt(self, text: str) -> str:
        """Create prompt for LLM extraction."""
        return f"""
        You are a specialized AI document parser designed to extract structured data from French audit final restitution presentation documents. Your task is to analyze the provided presentation text and extract specific information elements without modifying, interpreting, or summarizing the original content.
        CRITICAL REQUIREMENTS

        PRESERVE ORIGINAL TEXT: Extract content exactly as written - do not rephrase, summarize, or interpret
        EXACT EXTRACTION: Copy text verbatim from the source document
        COMPLETE COVERAGE: Extract ALL instances of the specified data types
        STRUCTURED OUTPUT: Return data in the exact JSON format specified below

        EXTRACTION TARGETS
        You must identify and extract the following elements from the document:
        1. Document Metadata (Required for all entries)

        mission_name: The official name/title of the mission
        entity: The organization or entity being audited
        thematiques: The themes or subject areas covered in the mission

        2. Content Categories (Extract separately for each data type)

        nombre_constats: Number of constats (findings) - extract as integer
        nombre_reco: Number of recommendations - extract as integer
        constat: Complete constat information including constats, risques, recommandations, and cotation des risques

        OUTPUT FORMAT
        Return your findings as a JSON array containing objects for each data type. Each object must follow this exact structure:
        json{{
          "mission_name": "exact mission name from document",
          "entity": "exact entity name from document", 
          "thematiques": "exact themes/subject areas from document",
          "data_type": "nombre_constats" OR "nombre_reco" OR "constat",
          "content": "mission_name / entity / All content for this data type"
        }}

        DETAILED INSTRUCTIONS
        Step 1: Document Analysis

        Read the entire document text carefully
        Identify the mission name, entity, and thematiques
        Locate all sections containing the specified data types

        Step 2: Content Extraction
        For each data type, extract ALL relevant content:

        For nombre_constats: Extract the number of constats as an integer
        For nombre_reco: Extract the number of recommendations as an integer
        For constat: Extract complete constat information including:
          - Constats (findings)
          - Risques (risks)
          - Recommandations (recommendations)
          - Cotation des risques (risk rating)
        Preserve the original wording and structure
        Include all details, bullet points, and structured items
        Maintain the original formatting

        Step 3: JSON Construction

        Create one JSON object for nombre_constats
        Create one JSON object for nombre_reco
        Create one JSON object for each constat found
        Combine all content for each data type into a single content field
        Start the content with "mission_name / entity /"
        Then include all the extracted content for that data type

        VALIDATION CHECKLIST
        Before returning your response, verify:

         All extracted content is verbatim from the source
         Mission name, entity, and thematiques are accurate
         All content for each data type is combined into a single content field
         Data type classification is correct
         JSON structure is valid and complete
         No content has been summarized or interpreted
         All relevant content is captured

        ERROR HANDLING

        If mission_name, entity, or thematiques cannot be identified, use "Not specified" as the value
        If no relevant content is found for a data type, skip that data type
        If text is unclear or ambiguous, extract it as-is without interpretation

        EXAMPLE OUTPUT STRUCTURE
        json[
          {{
            "mission_name": "Audit de Projet d'investissement - Usine OCP Benguerir",
            "entity": "OCP Benguerir",
            "thematiques": "Sécurité",
            "data_type": "nombre_constats",
            "content": "Audit de Projet d'investissement - Usine OCP Benguerir / OCP Benguerir / 5"
          }},
          {{
            "mission_name": "Audit de Projet d'investissement - Usine OCP Benguerir",
            "entity": "OCP Benguerir",
            "thematiques": "Sécurité",
            "data_type": "nombre_reco",
            "content": "Audit de Projet d'investissement - Usine OCP Benguerir / OCP Benguerir / 8"
          }},
          {{
            "mission_name": "Audit de Projet d'investissement - Usine OCP Benguerir",
            "entity": "OCP Benguerir",
            "thematiques": "Sécurité",
            "data_type": "constat",
            "content": "Audit de Projet d'investissement - Usine OCP Benguerir / OCP Benguerir / CONSTAT N°1 : XXXX\\nCONSTATS :\\n- Xxxx ;\\n- Xxxx.\\nRISQUES :\\n- Xxxx.\\n- Xxxx.\\nRECOMMANDATIONS :\\n- Xxxx. (sigle entité concernée)\\n- Xxxx. (sigle entité concernée)\\nCOTATION DES RISQUES : XXXXX"
          }},
          {{
            "mission_name": "Audit de Projet d'investissement - Usine OCP Benguerir",
            "entity": "OCP Benguerir",
            "thematiques": "Sécurité",
            "data_type": "constat",
            "content": "Audit de Projet d'investissement - Usine OCP Benguerir / OCP Benguerir / CONSTAT N°2 : YYYY\\nCONSTATS :\\n- Yyyy ;\\n- Yyyy.\\nRISQUES :\\n- Yyyy.\\n- Yyyy.\\nRECOMMANDATIONS :\\n- Yyyy. (sigle entité concernée)\\n- Yyyy. (sigle entité concernée)\\nCOTATION DES RISQUES : YYYYY"
          }}
        ]
        DOCUMENT TEXT TO ANALYZE:
        {text[:15000]}
        """