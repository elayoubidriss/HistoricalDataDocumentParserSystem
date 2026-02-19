import os
import sys
from pathlib import Path
import pandas as pd
from typing import Literal

# Configuration du cache Hugging Face dans votre répertoire utilisateur
os.environ["HF_HOME"] = r"C:\Users\driss.elayoubi\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"  # Désactive les liens symboliques

# Créer le répertoire de cache si nécessaire
cache_dir = os.environ["HF_HOME"]
if not os.path.exists(cache_dir):
    try:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Création du répertoire de cache: {cache_dir}")
    except Exception as e:
        print(f"Erreur création cache: {e}")
        # Fallback vers un sous-dossier temporaire
        temp_cache = r"C:\Users\driss.elayoubi\AppData\Local\Temp\hf_cache"
        os.makedirs(temp_cache, exist_ok=True)
        os.environ["HF_HOME"] = temp_cache

# Ajouter ceci avant vos autres imports
sys.path.insert(0, r"C:\Users\driss.elayoubi\Desktop\historical_data_document_parser_system")

from models.settings import (
    LettreDeMissionParserSettings,
    WorkProgramParserSettings,
    SynthesePreparationParserSettings,
    SupportKickoffParserSettings,
    PipelineSettings, RapportFinalParserSettings, RestitutionFinalParserSettings
)
from services.chunking_services.chunking_service import ChunkerService
from services.llm_services.llm_service import LLMService
from services.vector_db_services.vector_db_service import VectorDBService
from services.embeddings_services.embeddings_service import EmbeddingsService
from services.parsing_services.batch_processor import BatchProcessor


def get_parser_settings(document_type):
    """Return appropriate parser settings based on document type"""
    parser_mapping = {
        "lettre_de_mission": LettreDeMissionParserSettings(extract_summaries=False),
        "work_program": WorkProgramParserSettings(extract_summaries=False),
        "synthese_de_preparation": SynthesePreparationParserSettings(extract_summaries=False),
        "kick_off": SupportKickoffParserSettings(extract_summaries=False),
        "rapport_final": RapportFinalParserSettings(extract_summaries=False),
        "restitution_final": RestitutionFinalParserSettings(extract_summaries=False),
    }
    return parser_mapping.get(document_type, LettreDeMissionParserSettings(extract_summaries=False))


def main():
    # Load configuration from environment variables
    pipeline_settings = PipelineSettings()

    # Initialize services using the nested settings
    LLMService.init(settings=pipeline_settings.llm)
    ChunkerService.init(settings=pipeline_settings.chunker)
    EmbeddingsService.init(settings=pipeline_settings.embeddings)

    # Configuration
    input_root_dir = "./inputs"
    output_file = "./outputs/historical_data_ingestion_output.xlsx"

    # Collect all files from subdirectories
    all_files = []
    for root, dirs, files in os.walk(input_root_dir):
        for file in files:
            if file.endswith(('.pdf', '.docx', '.pptx')):
                document_type = os.path.basename(root)
                file_path = os.path.join(root, file)
                parser_settings = get_parser_settings(document_type)
                all_files.append((file_path, parser_settings))

    # Process files and export to Excel
    processor = BatchProcessor()
    parsed_data = processor.process_files(all_files)

    # Group data by document type for separate Excel sheets
    data_by_type = {}
    for file_path, data in zip([f[0] for f in all_files], parsed_data):
        document_type = os.path.basename(os.path.dirname(file_path))
        if document_type not in data_by_type:
            data_by_type[document_type] = []
        data_by_type[document_type].extend(data)

    # Write to Excel with multiple sheets
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for doc_type, data in data_by_type.items():
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=doc_type[:31], index=False)  # Sheet name max 31 chars

    print("Processing completed successfully!")
    print(f"Output saved to {output_file}")


if __name__ == "__main__":
    main()