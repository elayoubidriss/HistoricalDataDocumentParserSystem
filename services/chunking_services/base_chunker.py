import abc
from typing import List


class BaseChunker(abc.ABC):
    """Abstract base class for a text chunking component."""

    @abc.abstractmethod
    def chunk(self, max_tokens, overlap_tokens, text) -> List[str]:
        """
        Split text into chunks of maximum length `chunk_size`.
        """
        pass

    def process_text(self, dir_path, file_name, summarize_texts, extract_tables, extract_images):
        pass