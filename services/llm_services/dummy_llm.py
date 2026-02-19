import asyncio
import time
from typing import Generator,List

from services.llm_services.base_llm import BaseLLM
from models.settings import BaseLLMSettings

class DummyLLM(BaseLLM):
    """A dummy LLM that simulates both full and streaming responses."""

    def __init__(self, settings: BaseLLMSettings):
        self.settings = settings
        print(f"Initializing DummyLLM with settings: {self.settings.json()}")

    def generate(self, prompt: str,temperature:float = 0.7, max_tokens:int = 512) -> str:
        return f"Full answer based on prompt:\n{prompt}"

    def generate_stream(self, prompt: str,system_prompt: str = "",temperature:float = 0.7, max_tokens:int = 512) -> Generator[str,None,None]:
        full_answer = f"Streaming answer based on prompt:\n{prompt}"
        for token in full_answer.split():
            time.sleep(0.1)
            yield token + " "

    async def generate_async(self, prompt: str, system_prompt: str = "",temperature:float = 0.7, max_tokens:int = 512) -> str:
        full_answer = f"Streaming answer based on prompt:\n{prompt}"
        async for token in full_answer.split():
            asyncio.sleep(0.1)
            yield token + " "


    def generate_batch(self,prompts: List[str],temperature: float=0.7,max_tokens: int = 512) -> List[str]: 
        raise NotImplementedError()
    
    
    async def async_generate_batch(self,prompts: List[str],temperature: float=0.7,max_tokens: int = 512) -> List[str]: 
        raise NotImplementedError()