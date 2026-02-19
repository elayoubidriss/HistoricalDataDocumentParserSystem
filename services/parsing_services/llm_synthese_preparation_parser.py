from base_parser import BaseParser
from models.settings import SynthesePreparationParserSettings


class SynthesePreparationParser(BaseParser):
    """Parser for 'Synthese Preparation' document type using LLM extraction."""

    def __init__(self, settings: SynthesePreparationParserSettings):
        super().__init__(settings)

    def _create_extraction_prompt(self, text: str) -> str:
        """Create prompt for LLM extraction."""
        return f"""
        You are a specialized AI document parser designed to extract structured data from French audit synthesis preparation documents. Your task is to analyze the provided document text and extract specific information elements without modifying, interpreting, or summarizing the original content.
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

        résultats_mission_antérieurs: Results from previous missions
        Référentiels: Reference frameworks, standards, or guidelines
        analyse_preliminaire: Preliminary analysis
        processus_couverts_par_la_mission: Processes covered by the mission
        priorisation_risques: Risk prioritization

        OUTPUT FORMAT
        Return your findings as a JSON array containing one object per data type per entity. Each object must follow this exact structure:
        json{{
          "mission_name": "exact mission name from document",
          "entity": "exact entity name from document", 
          "thematiques": "exact themes/subject areas from document",
          "data_type": "résultats_mission_antérieurs" OR "Référentiels" OR "analyse_preliminaire" OR "processus_couverts_par_la_mission" OR "priorisation_risques",
          "content": "mission_name / entity / All content for this data type"
        }}

        DETAILED INSTRUCTIONS
        Step 1: Document Analysis

        Read the entire document text carefully
        Identify the mission name, entity, and thematiques
        Locate all sections containing the specified data types

        Step 2: Content Extraction
        For each data type, extract ALL relevant content:

        Group all content for the same data type and entity together
        Preserve the original wording and structure
        Include all details, bullet points, and structured items
        Maintain the original formatting

        Step 3: JSON Construction

        Create one JSON object per data type per entity
        Combine all content for that data type and entity into a single content field
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
            "data_type": "résultats_mission_antérieurs",
            "content": "Audit de Projet d'investissement - Usine OCP Benguerir / OCP Benguerir / Lors de la mission précédente, les principaux résultats étaient: 1. Non-conformité mineure dans la documentation des procédures 2. Retard dans la mise en œuvre des actions correctives 3. Excellente collaboration de l'équipe sur place"
          }},
          {{
            "mission_name": "Audit de Projet d'investissement - Usine OCP Benguerir",
            "entity": "OCP Benguerir",
            "thematiques": "Sécurité",
            "data_type": "Référentiels",
            "content": "Audit de Projet d'investissement - Usine OCP Benguerir / OCP Benguerir / Référentiels applicables: 1. Norme ISO 9001:2015 2. Procédures internes OCP version 3.2 3. Réglementation sectorielle marocaine"
          }},
          {{
            "mission_name": "Audit de Projet d'investissement - Usine OCP Benguerir",
            "entity": "OCP Benguerir",
            "thematiques": "Sécurité",
            "data_type": "analyse_preliminaire",
            "content": "Audit de Projet d'investissement - Usine OCP Benguerir / OCP Benguerir / L'analyse préliminaire a identifié les zones à risque suivantes: 1. Processus de validation des fournisseurs 2. Contrôle qualité des matières premières 3. Documentation des procédures opérationnelles"
          }}
        ]
        DOCUMENT TEXT TO ANALYZE:
        {text[:15000]}
        """