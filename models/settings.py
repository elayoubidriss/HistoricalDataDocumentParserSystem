from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import Literal, Union


class BaseLLMSettings(BaseModel):
    type: str


class DummyLLMSettings(BaseLLMSettings):
    type: Literal['dummy_llm'] = 'dummy_llm'


class VLLMLLMSettings(BaseLLMSettings):
    type: Literal['vllm_llm'] = 'vllm_llm'
    base_url: str
    model: str
    api_key: str = 'xxx'


###########################################################################

class BaseVectorDBSettings(BaseModel):
    type: str


class DummyVectorDBSettings(BaseVectorDBSettings):
    type: Literal['dummy_vector_db'] = 'dummy_vector_db'


class PGVectorDBSettings(BaseVectorDBSettings):
    type: Literal['pgvector_db'] = 'pgvector_db'
    drivername: str = "postgresql+psycopg"
    name: str
    username: str
    password: str
    host: str = "localhost"
    port: int
    embedding_dim: int


class ChromaVectorDBSettings(BaseVectorDBSettings):
    type: Literal['chroma_db'] = 'chroma_db'
    host: str = "localhost"
    port: int


###########################################################################

class BaseChunkerSettings(BaseModel):
    """Base settings for all chunkers with common fields"""
    type: str
    max_tokens: int = Field(default=4000)
    overlap_tokens: int = Field(default=500)
    tokenization_model: str = Field(default='BAAI/bge-m3')
    crop_extension: float = Field(default=0.4)
    use_custom_chunker: bool = Field(default=False)


class DoclingChunkerSettings(BaseChunkerSettings):
    """Settings specific to DoclingChunker"""
    type: Literal['docling_chunker'] = 'docling_chunker'


class UnstructuredChunkerSettings(BaseChunkerSettings):
    """Settings specific to UnstructuredChunker"""
    type: Literal['unstructured_chunker'] = 'unstructured_chunker'


class SimpleChunkerSettings(BaseChunkerSettings):
    """Settings specific to SimpleChunker"""
    type: Literal['simple_chunker'] = 'simple_chunker'


###########################################################################

class BaseEmbeddingsSettings(BaseModel):
    type: str


class DummyEmbeddingsSettings(BaseEmbeddingsSettings):
    type: Literal['dummy_embeddings'] = 'dummy_embeddings'


class VLLMEmbeddingsSettings(BaseEmbeddingsSettings):
    type: Literal['vllm_embeddings'] = 'vllm_embeddings'
    base_url: str
    model: str
    api_key: str = "xxx"


##############################################################################################################################

class BaseParserSettings(BaseModel):
    type: str


class PipelineSettings(BaseSettings):
    llm: DummyLLMSettings | VLLMLLMSettings = Field(..., discriminator="type")
    vector_db: PGVectorDBSettings | ChromaVectorDBSettings | DummyVectorDBSettings = Field(...,discriminator="type")
    chunker: DoclingChunkerSettings | UnstructuredChunkerSettings = Field(...,discriminator="type")
    embeddings: DummyEmbeddingsSettings | VLLMEmbeddingsSettings = Field(..., discriminator="type")

    class Config:
        env_nested_delimiter = '__'


class LettreDeMissionParserSettings(BaseParserSettings):
    type: Literal['lettre_de_mission_parser'] = 'lettre_de_mission_parser'
    extract_summaries: bool = True


class WorkProgramParserSettings(BaseParserSettings):
    type: Literal['work_program_parser'] = 'work_program_parser'
    extract_summaries: bool = True


class SynthesePreparationParserSettings(BaseParserSettings):
    type: Literal['synthese_de_preparation_parser'] = 'synthese_de_preparation_parser'
    extract_summaries: bool = True


class SupportKickoffParserSettings(BaseParserSettings):
    type: Literal['kick_off_parser'] = 'kick_off_parser'
    extract_summaries: bool = True


class RapportFinalParserSettings:
    type: Literal['rapport_final_parser'] = 'rapport_final_parser'
    extract_summaries: bool = True


class RestitutionFinalParserSettings:
    type: Literal['restitution_final_parser'] = 'restitution_final_parser'
    extract_summaries: bool = True