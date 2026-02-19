import re
import os
import logging
import json
from typing import List, Dict
from base_parser import BaseParser
from models.settings import SupportKickoffParserSettings
from services.chunking_services.chunking_service import ChunkerService
from services.llm_services.llm_service import LLMService

# Configure logging
logger = logging.getLogger(__name__)


class SupportKickoffParser(BaseParser):
    """Parser for 'Support Kickoff' presentation files using LLM extraction."""

    def __init__(self, settings: SupportKickoffParserSettings):
        super().__init__(settings)
        self.settings = settings

    def parse(self, file_path: str) -> List[Dict[str, str]]:
        """
        Extract data from Support Kickoff presentation files using LLM extraction.
        """
        try:
            # For PPT files, we need to use a different approach than the base parser
            if not (file_path.endswith('.ppt') or file_path.endswith('.pptx')):
                logger.error(f"Unsupported file format for SupportKickoffParser: {file_path}")
                return []

            # Process presentation with the chunker service
            dir_path = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)

            # Use the chunker to extract text from the presentation
            # We need to ensure we're using UnstructuredChunker which supports PPT files
            text_summaries, table_summaries, image_summaries, texts, tables, images = ChunkerService.get().process_text(
                dir_path, file_name,
                summarize_texts=False,
                extract_tables=False,
                extract_images=False
            )

            # For PPT files, we need to handle the text extraction differently
            # The texts array contains slide content, we need to identify the "points_d_attention" slide
            full_text = self._extract_points_d_attention(texts)

            if not full_text:
                logger.warning(f"No points_d_attention found in {file_path}")
                return []

            # Use LLM for extraction
            return self._extract_with_llm(full_text)

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            raise

    def _extract_points_d_attention(self, texts: List) -> str:
        """
        Extract the points_d_attention content from the presentation slides.
        This method identifies the slide(s) containing points of attention.
        """
        points_text = ""

        # Look for slides that contain keywords related to points of attention
        keywords = ["points d'attention", "POINTS D'ATTENTION", "points d attention",
                    "attention", "key points", "focus areas", "considerations", "PRINCIPAUX POINTS D'ATTENTION"]

        for text in texts:
            slide_content = text[1] if len(text) > 1 else ""
            slide_content_lower = slide_content.lower()

            # Check if this slide contains any of our keywords
            if any(keyword in slide_content_lower for keyword in keywords):
                points_text += slide_content + "\n\n"

        return points_text

    def _create_extraction_prompt(self, text: str) -> str:
        """Create prompt for LLM extraction."""
        return f"""
        You are a specialized AI document parser designed to extract structured data from French audit support kickoff presentation documents. Your task is to analyze the provided presentation text and extract specific information elements without modifying, interpreting, or summarizing the original content.
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

        2. Content Categories (Extract separately)

        points_d_attention: Points of attention, key focus areas, or important considerations

        OUTPUT FORMAT
        Return your findings as a JSON array containing one object. The object must follow this exact structure:
        json{{
          "mission_name": "exact mission name from document",
          "entity": "exact entity name from document", 
          "thematiques": "exact themes/subject areas from document",
          "data_type": "points_d_attention",
          "content": "mission_name / entity / All points of attention for this entity"
        }}

        DETAILED INSTRUCTIONS
        Step 1: Document Analysis

        Read the entire document text carefully
        Identify the mission name, entity, and thematiques
        Locate all sections containing points of attention, key focus areas, or important considerations

        Step 2: Content Extraction
        For points_d_attention:

        Extract all points of attention, key focus areas, or important considerations
        Preserve the original wording and structure
        Include all details, bullet points, and structured items
        Maintain the original formatting

        Step 3: JSON Construction

        Create one JSON object containing all points of attention
        Combine all points of attention into a single content field
        Start the content with "mission_name / entity /"
        Then include all the extracted points of attention

        VALIDATION CHECKLIST
        Before returning your response, verify:

         All extracted content is verbatim from the source
         Mission name, entity, and thematiques are accurate
         All points of attention are combined into a single content field
         Data type classification is correct ("points_d_attention")
         JSON structure is valid and complete
         No content has been summarized or interpreted
         All points of attention are captured

        ERROR HANDLING

        If mission_name, entity, or thematiques cannot be identified, use "Not specified" as the value
        If no points of attention are found, return an empty array: []
        If text is unclear or ambiguous, extract it as-is without interpretation

        EXAMPLE OUTPUT STRUCTURE
        json[
          {{
            "mission_name": "Audit de Projet d'investissement - Usine OCP Benguerir",
            "entity": "OCP Benguerir",
            "thematiques": "Sécurité",
            "data_type": "points_d_attention",
            "content": "Audit de Projet d'investissement - Usine OCP Benguerir / OCP Benguerir / Point 1: Vérification de la conformité des procédures de sécurité\\n- S'assurer que toutes les procédures sont à jour\\n- Vérifier la formation du personnel aux nouvelles procédures\\n\\nPoint 2: Analyse des risques liés aux nouveaux équipements\\n- Identifier les risques potentiels\\n- Évaluer les mesures de mitigation existantes\\n\\nPoint 3: Revue de la documentation des incidents\\n- Vérifier la complétude des rapports d'incidents\\n- Analyser les tendances sur les 6 derniers mois"
          }}
        ]
        DOCUMENT TEXT TO ANALYZE:
        {text[:15000]}
        """