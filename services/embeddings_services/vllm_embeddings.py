from typing import List
from openai import OpenAI

from services.embeddings_services.base_embeddings import BaseEmbeddings
from models.settings import VLLMEmbeddingsSettings

class VLLMEmbeddings(BaseEmbeddings):
    """A dummy embeddings model that converts text into a list of floats."""

    def __init__(self, settings: VLLMEmbeddingsSettings):
        self.settings = settings
        self.client = OpenAI(
                    base_url=settings.base_url,
                    api_key = settings.api_key
                )
        self.model = settings.model

        print(f"Initializing embedding through VLLM with settings: {self.settings.json()}")

    def embed(self, text: str) -> List[float]:
        res = self.client.embeddings.create(model = self.model,input=text)
        return res.data[0].embedding
    
    def embed_query(self, text: str) -> List[float]:
        res = self.client.embeddings.create(model = self.model,input=text)
        return res.data[0].embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = self.client.embeddings.create(model = self.model,input=texts)
        return [x.embedding for x in res.data]
    
    def embed_parallel(self, texts: List[str]) -> List[List[float]]:
        res = self.client.embeddings.create(model = self.model,input=texts)
        return [x.embedding for x in res.data]
    

    