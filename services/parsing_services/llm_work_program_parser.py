from base_parser import BaseParser
from models.settings import WorkProgramParserSettings


class WorkProgramParser(BaseParser):
    """Parser for 'Work Program' document type using LLM extraction."""

    def __init__(self, settings: WorkProgramParserSettings):
        super().__init__(settings)

    def _create_extraction_prompt(self, text: str) -> str:
        """Create prompt for LLM extraction."""
        return f"""
        You are a specialized AI document parser designed to extract structured data from French audit work program documents. Your task is to analyze the provided document text and extract specific information elements without modifying, interpreting, or summarizing the original content.
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

        work_program: All audit tests and verification procedures for each entity

        OUTPUT FORMAT
        Return your findings as a JSON array containing one object per entity. Each object must follow this exact structure:
        json{{
          "mission_name": "exact mission name from document",
          "entity": "exact entity name from document", 
          "thematiques": "exact themes/subject areas from document",
          "data_type": "work_program",
          "content": "All audit tests for this entity, formatted as: mission_name / entity / Test 1: description\\nTest 2: description\\n..."
        }}

        DETAILED INSTRUCTIONS
        Step 1: Document Analysis

        Read the entire document text carefully
        Identify the mission name, entity, and thematiques
        Locate all sections containing audit tests, verification procedures, or work program items

        Step 2: Content Extraction
        For each entity, extract ALL audit tests:

        Group all tests for the same entity together
        Preserve the original wording and structure of each test
        Number each test sequentially (Test 1, Test 2, etc.)
        Include all details, steps, and verification procedures
        Maintain bullet points, numbered lists, or structured items

        Step 3: JSON Construction

        Create one JSON object per entity
        Combine all tests for that entity into a single content field
        Start the content with "mission_name / entity /"
        Then list all tests sequentially with proper numbering

        VALIDATION CHECKLIST
        Before returning your response, verify:

         All extracted content is verbatim from the source
         Mission name, entity, and thematiques are accurate
         All tests for each entity are combined into a single content field
         Tests are numbered sequentially
         JSON structure is valid and complete
         No content has been summarized or interpreted
         All audit tests are captured

        ERROR HANDLING

        If mission_name, entity, or thematiques cannot be identified, use "Not specified" as the value
        If no work program items are found, return an empty array: []
        If text is unclear or ambiguous, extract it as-is without interpretation

        EXAMPLE OUTPUT STRUCTURE
        json[
          {{
            "mission_name": "Audit de Projet d'investissement - Usine OCP Benguerir",
            "entity": "OCP Benguerir",
            "thematiques": "Sécurité",
            "data_type": "work_program",
            "content": "Audit de Projet d'investissement - Usine OCP Benguerir / OCP Benguerir / Test 1: Vérification de contrats de maintenance\\n- Vérifier l'existence du document 'Contrats de maintenance'\\n- S'assurer de la complétude de 'Contrats de maintenance'\\n- Contrôler la conformité de 'Contrats de maintenance' avec les procédures internes\\n- Vérifier la date de mise à jour de 'Contrats de maintenance'\\n- Évaluer l'utilisation de 'Contrats de maintenance' dans les processus décisionnels\\n\\nTest 2: Vérification de preuves de conformité\\n- Vérifier l'existence du document 'Preuves de conformité'\\n- S'assurer de la complétude de 'Preuves de conformité'\\n- Contrôler la conformité de 'Preuves de conformité' avec les procédures internes\\n- Vérifier la date de mise à jour de 'Preuves de conformité'\\n- Évaluer l'utilisation de 'Preuves de conformité' dans les processus décisionnels"
          }}
        ]
        DOCUMENT TEXT TO ANALYZE:
        {text[:15000]}
        """