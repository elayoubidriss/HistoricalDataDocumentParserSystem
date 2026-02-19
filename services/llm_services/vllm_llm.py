from openai import OpenAI, AsyncOpenAI
from typing import Generator, List

from services.llm_services.base_llm import BaseLLM
from models.settings import VLLMLLMSettings
from langchain_openai import ChatOpenAI
from langchain_community.llms import VLLMOpenAI

class VLLMLLM(BaseLLM):
    """A dummy LLM that simulates both full and streaming responses."""

    def __init__(self, settings: VLLMLLMSettings):
        self.settings = settings
        self.client = OpenAI(
                    base_url=settings.base_url,
                    api_key = settings.api_key
                )
        self.async_client = AsyncOpenAI(
                    base_url=settings.base_url,
                    api_key = settings.api_key
                )
        self.model = settings.model
        
        print(f"Initializing ACSLLM with settings: {self.settings.json()}")

    def generate(self, prompt: str, system_prompt: str = "",temperature:float = 0.7, max_tokens:int = 512) -> str:
        if system_prompt:
            messages = [{"role":"system","content":system_prompt},{"role": "user", "content": prompt}]
        else:    
            messages = [{"role": "user", "content": prompt}]
        completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                    )
        return completion.choices[0].message.content
    
    def get_langchain_chat(self, temperature:float = 0.2) -> ChatOpenAI:
        return ChatOpenAI(temperature=temperature, model=self.model, base_url=self.settings.base_url, api_key=self.settings.api_key )
        #return VLLMOpenAI(model_name=self.model, openai_api_base=self.settings.base_url, openai_api_key=self.settings.api_key)
        
    
    async def generate_async(self, prompt: str, system_prompt: str = "",temperature:float = 0.7, max_tokens:int = 512) -> str:
        if system_prompt:
            messages = [{"role":"system","content":system_prompt},{"role": "user", "content": prompt}]
        else:    
            messages = [{"role": "user", "content": prompt}]
        completion = await self.async_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                    )
        return completion.choices[0].message.content

    def generate_stream(self, prompt: str,system_prompt: str = "",temperature:float = 0.7, max_tokens:int = 512) -> Generator[str,None,None]:
        if system_prompt:
            messages = [{"role":"system","content":system_prompt},{"role": "user", "content": prompt}]
        else:    
            messages = [{"role": "user", "content": prompt}]
        response =  self.client.chat.completions.create(
                        model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True
                    )
        for chunk in response:
            yield chunk.choices[0].delta.content

    def generate_batch(self,prompts: List[str],temperature: float=0.7,max_tokens: int = 512) -> List[str]: 
        batch = self.client.completions.create( 
            model=self.model, 
            prompt=prompts, 
            max_tokens=max_tokens, 
            temperature=temperature
            ) 
        return [x.text for x in batch.choices]
    
    async def generate_batch_async(self,prompts: List[str],temperature: float=0.7,max_tokens: int = 512) -> List[str]: 
        batch = await self.async_client.completions.create( 
            model=self.model, 
            prompt=prompts, 
            max_tokens=max_tokens, 
            temperature=temperature
            ) 
        return [x.text for x in batch.choices]
        


