import abc
import os
import logging
import json
from typing import List, Dict
from models.settings import BaseParserSettings
from services.chunking_services.chunking_service import ChunkerService
from services.llm_services.llm_service import LLMService

# Configure logging
logger = logging.getLogger(__name__)


class BaseParser(abc.ABC):
    """Abstract base class for document parsers."""

    def __init__(self, settings: BaseParserSettings):
        self.settings = settings

    def parse(self, file_path: str) -> List[Dict[str, str]]:
        """
        Parse a document and extract structured data.

        Args:
            file_path: Path to the document file

        Returns:
            List of dictionaries representing extracted data rows
        """
        try:
            # Process PDF with DoclingChunker
            dir_path = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            _, _, _, texts, _, _ = ChunkerService.get().process_text(
                dir_path, file_name,
                summarize_texts=False,
                extract_tables=False,
                extract_images=False
            )

            # Reconstruct document text in reading order
            full_text = "\n".join([text[1] for text in texts])
            logger.debug(f"Reconstructed text:\n{full_text[:2000]}...")

            # Use LLM for extraction
            return self._extract_with_llm(full_text)

        except Exception as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
            raise

    def _extract_with_llm(self, text: str) -> List[Dict[str, str]]:
        """Extract structured data using LLM."""
        prompt = self._create_extraction_prompt(text)
        response = LLMService.get().generate(prompt)

        try:
            # Parse LLM response as JSON
            extracted_data = json.loads(response)
            return self._format_extracted_data(extracted_data)
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            return []

    def _format_extracted_data(self, extracted_data: List[Dict]) -> List[Dict[str, str]]:
        """Format extracted data to match expected output structure.

        Applies column_mapping from settings to rename internal field names
        to the configured output column names.
        """
        mapping = self.settings.column_mapping
        formatted_data = []

        # Internal keys with their default values
        defaults = {
            "mission_name": "Not specified",
            "entity": "Not specified",
            "thematiques": "Not specified",
            "data_type": "",
            "content": "",
        }

        for item in extracted_data:
            formatted_item = {}
            for internal_key, default_val in defaults.items():
                output_col = mapping.get(internal_key, internal_key)
                formatted_item[output_col] = item.get(internal_key, default_val)
            formatted_data.append(formatted_item)

        return formatted_data

    def get_output_columns(self) -> List[str]:
        """Return ordered list of output column names based on column_mapping."""
        internal_order = ["mission_name", "entity", "thematiques", "data_type", "content"]
        mapping = self.settings.column_mapping
        return [mapping.get(key, key) for key in internal_order]

    @abc.abstractmethod
    def _create_extraction_prompt(self, text: str) -> str:
        """Create prompt for LLM extraction. Must be implemented by subclasses."""
        pass