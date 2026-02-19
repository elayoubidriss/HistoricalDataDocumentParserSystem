from datetime import datetime
from enum import StrEnum

from utils.entity import Entity

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class BaseEvent(BaseModel):
    correlation_id: Optional[str] = None
    timestamp: Optional[str] = None


class DocumentInputType(StrEnum):
    AnalyzeDocument = "AnalyzeDocument"
    GenerateSummaries = "GenerateSummaries"
    ExtractInformation = "ExtractInformation"

class DocumentInput(BaseEvent):
    type: DocumentInputType
    document_id: str | None = None
    project_id: str | None = None
    force_reload_document: bool = False


class DocumentStatus(StrEnum):
    Expected = "Expected"
    Stored = "Stored"
    ExtractingContent = "ExtractingContent"
    ContentExtracted = "ContentExtracted"
    Uploaded = "Uploaded"


class DocumentType(StrEnum):
    Unknown = "Unknown"
    CompanyFinancialPublication = "CompanyFinancialPublication"
    Article = "Article"
    


class Document(Entity):
    id: str
    organization: str
    path: str
    sender: int | None
    name: str | None
    description: str | None
    status: DocumentStatus
    created: datetime
    document_type: DocumentType | None
    category: DocumentType | None
    sub_category: str | None
    source_url: str | None
    sharepoint_id:int 

    def get_entity_name(self) -> str:
        return "Document" #MZ : return "self.name" ?

    def get_entity_id(self) -> str:
        return self.id

    def get_entity_organization(self) -> str:
        return self.organization
    
    def get_entity_sharepoint_id(self) -> int:
        return self.sharepoint_id

