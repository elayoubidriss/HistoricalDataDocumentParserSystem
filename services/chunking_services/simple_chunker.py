from typing import List

from services.chunking_services.base_chunker import BaseChunker
from models.settings import BaseChunkerSettings

class SimpleChunker(BaseChunker):
    """A simple chunker that splits text into fixed-size chunks."""

    def __init__(self, settings: BaseChunkerSettings):
        self.settings = settings
        print(f"Initializing SimpleChunker with settings: {self.settings.json()}")

    def chunk(self, max_tokens, overlap_tokens,text) -> List[str]:
        return [text[i: i + max_tokens] for i in range(0, len(text), max_tokens)]