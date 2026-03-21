from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class SourceItem(BaseModel):
    source_file: str | None = None
    chunk_index: int | str | None = None
    preview: str


class ChatResponse(BaseModel):
    query: str
    top_k: int
    answer: str
    sources: list[SourceItem]