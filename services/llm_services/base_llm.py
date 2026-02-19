import abc
from typing import Generator,List

class BaseLLM(abc.ABC):
    """Abstract base class for a Language Model (LLM)."""

    @abc.abstractmethod
    def generate(self, prompt: str,temperature:float = 0.7, max_tokens:int = 512) -> str:
        """Generate a complete answer from a prompt."""
        raise NotImplementedError()

    @abc.abstractmethod
    def generate_stream(self, prompt: str,temperature:float = 0.7, max_tokens:int = 512) -> Generator[str,None,None]:
        """Asynchronously generate a streaming answer from a prompt."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    async def generate_async(self, prompt: str, system_prompt: str = "",temperature:float = 0.7, max_tokens:int = 512) -> str:
        raise NotImplementedError()


    @abc.abstractmethod
    def generate_batch(self,prompts: List[str],temperature: float=0.7,max_tokens: int = 512) -> List[str]: 
        raise NotImplementedError()
    
    
    async def async_generate_batch(self,prompts: List[str],temperature: float=0.7,max_tokens: int = 512) -> List[str]: 
        raise NotImplementedError()
    