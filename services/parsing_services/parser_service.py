from typing import Dict, Type
from models.settings import (
    BaseParserSettings,
    LettreDeMissionParserSettings,
    WorkProgramParserSettings,
    SynthesePreparationParserSettings,
    SupportKickoffParserSettings
)
from base_parser import BaseParser
from llm_lettre_de_mission_parser import LettreDeMissionParser
from llm_work_program_parser import WorkProgramParser
from llm_synthese_preparation_parser import SynthesePreparationParser
from llm_support_kickoff_parser import SupportKickoffParser
from llm_rapport_final_parser import RapportFinalParser
from llm_restitution_final_parser import RestitutionFinalParser


class ParserFactory:
    """Factory for creating parser instances with settings."""

    _mapping: Dict[str, Type[BaseParser]] = {
        "lettre_de_mission_parser": LettreDeMissionParser,
        "work_program_parser": WorkProgramParser,
        "synthese_de_preparation_parser": SynthesePreparationParser,
        "kick_off_parser": SupportKickoffParser,
        "rapport_final_parser": RapportFinalParser,
        "restitution_final_parser": RestitutionFinalParser
    }

    @classmethod
    def create(cls, settings: BaseParserSettings) -> BaseParser:
        """Create a parser instance with the given settings."""
        parser_type = settings.type
        if parser_type not in cls._mapping:
            raise ValueError(f"Parser '{parser_type}' not recognized. Valid options: {list(cls._mapping.keys())}")
        return cls._mapping[parser_type](settings)


class ParserService:
    """Service for accessing document parsers with settings."""

    _parser_cache: Dict[str, BaseParser] = {}

    @classmethod
    def get_parser(cls, settings: BaseParserSettings) -> BaseParser:
        """Get a parser instance configured with the given settings."""
        parser_type = settings.type
        if parser_type not in cls._parser_cache:
            cls._parser_cache[parser_type] = ParserFactory.create(settings)
        return cls._parser_cache[parser_type]