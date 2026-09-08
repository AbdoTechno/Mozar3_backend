from typing import List, Optional
from pydantic import BaseModel, Field


class FarmerContext(BaseModel):
    """Personalized context about the Egyptian farmer and their crop/land."""

    farmer_name: Optional[str] = Field(
        default=None,
        description="Farmer's name, e.g., عبدالرحمن",
        examples=["عبدالرحمن"],
    )
    crop: Optional[str] = Field(
        default=None,
        description="Cultivated crop, e.g., قمح, بصل, ذرة, طماطم",
        examples=["قمح"],
    )
    soil_type: Optional[str] = Field(
        default=None,
        description="Type of soil, e.g., طينية, رملية, طميية, جيرية",
        examples=["طينية"],
    )
    irrigation_system: Optional[str] = Field(
        default=None,
        description="Irrigation method, e.g., ري بالغمر, ري بالتنقيط, ري بالرش",
        examples=["ري بالتنقيط"],
    )
    governorate: Optional[str] = Field(
        default=None,
        description="Egyptian governorate, e.g., كفر الشيخ, البحيرة, الدقهلية",
        examples=["كفر الشيخ"],
    )
    crop_age_days: Optional[int] = Field(
        default=None,
        description="Age of the crop in days from sowing or transplanting",
        examples=[40],
    )
    user_role: Optional[str] = Field(
        default="مزارع",
        description="Role of the user, e.g., 'مزارع', 'مهندس زراعي', 'تاجر', 'طالب / باحث'",
        examples=["مزارع"],
    )


class ChatMessage(BaseModel):
    """A single turn in conversational chat history."""

    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class ChatRequest(BaseModel):
    """Input payload for farmer query."""

    query: str = Field(
        ...,
        min_length=2,
        description="Question asked by the farmer in Arabic",
        examples=["ما هي كمية السماد المطلوبة لري المحاياة في القمح؟"],
    )
    farmer_context: Optional[FarmerContext] = Field(
        default=None,
        description="Optional farmer profile and agricultural parameters",
    )
    history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Previous messages in the conversation for continuity",
    )


class SourceItem(BaseModel):
    """Reference source chunk retrieved from official agricultural guides."""

    document: str = Field(..., description="Name of the official document/bulletin")
    page: Optional[int] = Field(default=None, description="Page number if available")
    snippet: str = Field(..., description="Relevant text snippet used in synthesis")


class ChatResponse(BaseModel):
    """Output response returned by RAG engine."""

    answer: str = Field(..., description="Generated answer in tailored Egyptian agricultural Arabic")
    sources: List[SourceItem] = Field(default_factory=list, description="List of source citations")
    relevant_chunks_count: int = Field(default=0, description="Total relevant vector chunks retrieved")


class IngestRequest(BaseModel):
    """Options for document re-indexing."""

    force_reindex: bool = Field(
        default=False,
        description="If True, clear existing vector store and re-index everything from scratch",
    )


class IngestResponse(BaseModel):
    """Result of ingestion operation."""

    status: str
    indexed_files: List[str] = Field(default_factory=list)
    skipped_files: List[str] = Field(default_factory=list)
    total_chunks_added: int
    message: str


class StatusResponse(BaseModel):
    """System and vector database health and indexing statistics."""

    status: str
    gemini_configured: bool
    total_documents_in_data: int
    indexed_documents_count: int
    total_chunks: int
    embedding_model: str
    chat_model: str
    indexed_files: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Basic health check status."""

    status: str
    version: str
    gemini_ready: bool
