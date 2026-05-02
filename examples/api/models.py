from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    user_id: str = Field(description="User identifier")


class StartSessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: str


class ChatRequest(BaseModel):
    session_id: str = Field(description="Session ID from /session/start")
    user_id: str = Field(description="User identifier")
    message: str = Field(description="User message")


class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    message: str


class EndSessionRequest(BaseModel):
    session_id: str
    user_id: str


class EndSessionResponse(BaseModel):
    session_id: str
    user_id: str
    status: str
    kg_extraction: str
    ended_at: str


class MemoryContextResponse(BaseModel):
    user_id: str
    context: str
    has_data: bool


class HealthResponse(BaseModel):
    status: str
    redis: bool
    neo4j: bool
