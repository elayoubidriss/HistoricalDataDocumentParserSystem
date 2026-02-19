# Historical Data Document Parser System

A Python-based system for parsing audit documents and extracting structured information into Excel format.

## Overview

This system parses 4 types of audit documents:
- **Lettre de Mission** (PDF) - Mission letters
- **Work Program** (PDF) - Detailed audit procedures
- **Synthèse de Préparation** (PDF) - Preparation summaries
- **Support de Kickoff** (PPT) - Kickoff presentation materials

## Architecture

### Core Components

1. **Base Parser (`base_parser.py`)**
   - Abstract base class defining the parser interface
   - Handles common functionality like Excel export
   - Uses the existing DoclingChunker service

2. **Specialized Parsers**
   - `LettresDeMissionParser` - Extracts objectives, team, calendar, etc.
   - `WorkProgramParser` - Extracts test procedures by theme
   - `SynthesePreparationParser` - Extracts risks, control points, methodology
   - `SupportKickoffParser` - Extracts agenda, participants, deliverables

3. **Parser Service (`parser_service.py`)**
   - Factory pattern for creating parser instances
   - Manages parser lifecycle
   - Integrates with ChunkerService

4. **Batch Processor (`batch_processor.py`)**
   - Processes multiple documents
   - Consolidates results into a single Excel file

## Data Structure

Each parser outputs records with the following structure:
```python
{
    'mission_name': str,      # Name of the audit mission
    'entity': str,            # Entity being audited (optional)
    'thematiques': str,       # Theme/category (optional)
    'data_type': str,         # Type of information
    'content': str,           # Actual content
    'source_file': str        # Source filename
}
```

## Installation

1. Ensure all dependencies are installed:
```bash
pip install pandas openpyxl docling langchain
```

2. Add the parser classes to your project structure:
```
services/
├── chunking_services/
│   ├── base_chunker.py
│   ├── chunking_service.py
│   └── docling_chunker.py
└── parsing_services/
    ├── base_parser.py
    ├── parser_service.py
    ├── lettre_de_mission_parser.py
    ├── work_program_parser.py
    ├── synthese_preparation_parser.py
    ├── support_kickoff_parser.py
    └── batch_processor.py
```

## Usage

### Basic Usage

```python
from services.chunking_services.chunking_service import ChunkerService
from services.parsing_services.parser_service import ParserService
from models.settings import DoclingChunkerSettings

# Initialize services
chunker_settings = DoclingChunkerSettings(
    type="docling_chunker",
    max_tokens=512,
    overlap_tokens=50,
    tokenization_model="gpt2",
    crop_extension=0.1,
    use_custom_chunker=True
)

ChunkerService.init(settings=chunker_settings)
ParserService.init(chunker_service=ChunkerService)

# Parse a document
parser = ParserService.get("lettre_de_mission")
results = parser.parse("./documents", "mission_letter.pdf")

# Export to Excel
parser.to_excel(results, "output.xlsx")
```

### Batch Processing

```python
from services.parsing_services.batch_processor import BatchProcessor

processor = BatchProcessor(output_path="all_documents.xlsx")

documents = [
    {'path': './docs', 'filename': 'letter.pdf', 'type': 'lettre_de_mission'},
    {'path': './docs', 'filename': 'program.pdf', 'type': 'work_program'},
    # ... more documents
]

processor.process_documents(documents)
processor.export_all()
```

## Customization

### Adding a New Parser

1. Create a new parser class inheriting from `BaseParser`:

```python
from services.parsing_services.base_parser import BaseParser

class CustomDocumentParser(BaseParser):
    def get_sheet_name(self) -> str:
        return "custom_sheet"
    
    def extract_fields(self, texts, tables, images) -> List[Dict[str, Any]]:
        # Custom extraction logic
        records = []
        # ... extract data ...
        return records
```

2. Register it in `ParserFactory`:

```python
class ParserFactory:
    _mapping: Dict[str, Type[BaseParser]] = {
        "lettre_de_mission": LettresDeMissionParser,
        "custom_document": CustomDocumentParser,  # Add here
        # ...
    }
```

### Modifying Extraction Logic

Each parser's `extract_fields` method can be customized to extract different information. The parsers use regex patterns to identify and extract specific content from the documents.

## Output Format

The system generates Excel files with:
- One sheet per document type
- Columns: `mission_name`, `entity`, `thematiques`, `data_type`, `content`, `source_file`
- Each row represents a specific piece of extracted information

## Error Handling

- The system logs all operations using Python's logging module
- Failed document parsing is logged but doesn't stop batch processing
- Missing or empty fields are handled gracefully

## Dependencies

- `pandas` - Data manipulation and Excel export
- `openpyxl` - Excel file handling
- `docling` - Document processing (via DoclingChunker)
- `langchain` - LLM integration for summaries
- Python standard library: `re`, `logging`, `pathlib`, etc.