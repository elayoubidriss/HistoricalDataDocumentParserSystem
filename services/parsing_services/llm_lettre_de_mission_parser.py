from base_parser import BaseParser
from models.settings import LettreDeMissionParserSettings


class LettreDeMissionParser(BaseParser):
    """Parser for 'Lettre de Mission' document type using LLM extraction."""

    def __init__(self, settings: LettreDeMissionParserSettings):
        super().__init__(settings)

    def _create_extraction_prompt(self, text: str) -> str:
        """Create prompt for LLM extraction."""
        return f"""
        You are a specialized AI document parser designed to extract structured data from French audit documents called "Lettre de Mission" (Mission Letters). Your task is to analyze the provided document text and extract specific information elements without modifying, interpreting, or summarizing the original content.
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
        
        objectifs: Mission objectives, goals, purposes, or aims
        list_documents: Lists of documents, files, or materials mentioned
        
        OUTPUT FORMAT
        Return your findings as a JSON array containing multiple objects. Each object must follow this exact structure:
        json{
          "mission_name": "exact mission name from document",
          "entity": "exact entity name from document", 
          "thematiques": "exact themes/subject areas from document",
          "data_type": "objectifs" OR "list_documents",
          "content": "exact extracted content without modification BUT add beforehand the mission name and entity for context"
        }
        DETAILED INSTRUCTIONS
        Step 1: Document Analysis
        
        Read the entire document text carefully
        Identify the mission name, entity, and thematiques (these will be consistent across all extractions)
        Locate all sections containing objectives or document lists
        
        Step 2: Content Extraction
        For objectifs:
        
        Extract complete sentences or paragraphs describing mission goals
        Include bullet points, numbered lists, or structured objectives
        Preserve original formatting and structure
        Do not combine separate objective statements
        
        For list_documents:
        
        Extract any mentions of specific documents, files, reports, or materials
        Include document names, references, or descriptions
        Preserve list formatting (bullets, numbers, etc.)
        Include both formal document titles and informal references
        
        Step 3: JSON Construction
        
        Create separate JSON objects for each distinct piece of content
        If there are 3 objectives and 2 document lists, create 5 separate JSON objects
        Ensure each object contains all required metadata fields
        Use the exact text from the document without modification
        
        VALIDATION CHECKLIST
        Before returning your response, verify:
        
         All extracted content is verbatim from the source
         Mission name, entity, and thematiques are consistent across all objects
         Each content item has the correct data_type classification
         JSON structure is valid and complete
         No content has been summarized or interpreted
         All instances of objectives and document lists are captured
        
        ERROR HANDLING
        
        If mission_name, entity, or thematiques cannot be identified, use "Not specified" as the value
        If no objectives or document lists are found, return an empty array: []
        If text is unclear or ambiguous, extract it as-is without interpretation
        
        EXAMPLE OUTPUT STRUCTURE
        json[
          {
            "mission_name": "Audit de la gestion financière 2024",
            "entity": "Société ABC",
            "thematiques": "Contrôle interne et conformité réglementaire",
            "data_type": "objectifs",
            "content": "Audit de la gestion financière 2024 / Société ABC / Vérifier la conformité des procédures comptables aux normes IFRS"
          },
          {
            "mission_name": "Audit de la gestion financière 2024", 
            "entity": "Société ABC",
            "thematiques": "Contrôle interne et conformité réglementaire",
            "data_type": "list_documents",
            "content": "Audit de la gestion financière 2024 / Société ABC / États financiers consolidés de l'exercice 2023"
          }
        ]
        DOCUMENT TEXT TO ANALYZE:
        {text[:15000]}
        """