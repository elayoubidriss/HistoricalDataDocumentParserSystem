
from typing import Type, Dict

from services.llm_services.base_llm import BaseLLM
from services.llm_services.vllm_llm import VLLMLLM
from services.llm_services.dummy_llm import DummyLLM
from models.settings import BaseLLMSettings

class LLMFactory:
    _mapping: Dict[str, Type[BaseLLM]] = {
        "dummy_llm": DummyLLM,
        "vllm_llm": VLLMLLM
    }

    @classmethod
    def create(cls, name: str, settings: BaseLLMSettings) -> BaseLLM:
        if name not in cls._mapping:
            raise ValueError(f"LLM '{name}' is not recognized. Options: {list(cls._mapping.keys())}")
        return cls._mapping[name](settings=settings)

class LLMService:
    _instance: BaseLLM = None

    @classmethod
    def init(cls, settings: BaseLLMSettings) -> None:
        # Create the LLM based on the type in the settings.
        cls._instance = LLMFactory.create(settings.type, settings=settings)
        print("LLMService initialized.")

    @classmethod
    def get(cls) -> BaseLLM:
        if cls._instance is None:
            raise RuntimeError("LLMService is not initialized.")
        return cls._instance