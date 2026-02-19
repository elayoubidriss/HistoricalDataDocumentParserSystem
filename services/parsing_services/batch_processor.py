from typing import Dict, List, Tuple
import openpyxl
from openpyxl.workbook import Workbook
from models.settings import BaseParserSettings
from parser_service import ParserService


class BatchProcessor:
    """Processes multiple documents and exports to Excel."""

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
            results[parser_type].extend(parsed_data)

        return results

    def to_excel(self, data: Dict[str, List[Dict]], output_path: str):
        """Export parsed data to Excel with separate sheets per parser type."""
        wb = Workbook()
        wb.remove(wb.active)  # Remove default sheet

        # Standard Excel columns for all document types
        COLUMNS = ["mission_name", "entity", "thematiques", "data_type", "content"]

        for parser_type, rows in data.items():
            # Use sheet name without '_parser' suffix
            sheet_name = parser_type.replace('_parser', '')
            ws = wb.create_sheet(title=sheet_name[:31])  # Excel sheet name limit

            # Write header
            ws.append(COLUMNS)

            # Write data rows
            for row in rows:
                ws.append([row.get(col, "") for col in COLUMNS])

        wb.save(output_path)