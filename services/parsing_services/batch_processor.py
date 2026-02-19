from typing import Dict, List, Tuple
import openpyxl
from openpyxl.workbook import Workbook
from models.settings import BaseParserSettings
from parser_service import ParserService


# Default internal column order
_INTERNAL_COLUMN_ORDER = ["mission_name", "entity", "thematiques", "data_type", "content"]


class BatchProcessor:
    """Processes multiple documents and exports to Excel."""

    def __init__(self):
        self._column_mappings: Dict[str, Dict[str, str]] = {}

    def process_files(self, files: List[Tuple[str, BaseParserSettings]]) -> Dict[str, List[Dict]]:
        """
        Process a list of files with their parser settings.

        Returns:
            Dictionary with parser types as keys and lists of parsed rows as values
        """
        results = {}

        for file_path, parser_settings in files:
            parser = ParserService.get_parser(parser_settings)
            parsed_data = parser.parse(file_path)

            parser_type = parser_settings.type
            if parser_type not in results:
                results[parser_type] = []
                self._column_mappings[parser_type] = parser_settings.column_mapping
            results[parser_type].extend(parsed_data)

        return results

    def to_excel(self, data: Dict[str, List[Dict]], output_path: str):
        """Export parsed data to Excel with separate sheets per parser type.

        Uses column_mapping from parser settings to determine column headers
        for each document type sheet.
        """
        wb = Workbook()
        wb.remove(wb.active)  # Remove default sheet

        for parser_type, rows in data.items():
            # Use sheet name without '_parser' suffix
            sheet_name = parser_type.replace('_parser', '')
            ws = wb.create_sheet(title=sheet_name[:31])  # Excel sheet name limit

            # Get column mapping for this parser type (fall back to identity mapping)
            mapping = self._column_mappings.get(parser_type, {
                k: k for k in _INTERNAL_COLUMN_ORDER
            })
            columns = [mapping.get(k, k) for k in _INTERNAL_COLUMN_ORDER]

            # Write header
            ws.append(columns)

            # Write data rows
            for row in rows:
                ws.append([row.get(col, "") for col in columns])

        wb.save(output_path)