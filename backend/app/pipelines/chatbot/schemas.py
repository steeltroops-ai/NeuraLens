"""
MediLens Medical Chatbot - Schemas
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Chat message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Individual chat message"""
    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None
    
    class Config:
        use_enum_values = True


class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., min_length=1, max_length=4000, description="User message")
    conversation_history: Optional[List[ChatMessage]] = Field(
        default=[],
        max_length=50,
        description="Previous messages in the conversation"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for conversation continuity"
    )
    context: Optional[str] = Field(
        default=None,
        description="Additional context (e.g., current page, patient data)"
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    message: str = Field(..., description="AI assistant response")
    session_id: str = Field(..., description="Session identifier")
    tokens_used: Optional[int] = Field(default=None, description="Tokens consumed")
    processing_time: float = Field(..., description="Response time in seconds")
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score of the response"
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description="Medical sources referenced"
    )
    disclaimer: str = Field(
        default="This is AI-generated medical information for educational purposes only. Always consult a healthcare professional for medical advice.",
        description="Medical disclaimer"
    )


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history"""
    session_id: str
    messages: List[ChatMessage]
    created_at: datetime
    last_updated: datetime


class ChatbotInfoResponse(BaseModel):
    """Response model for chatbot info endpoint"""
    name: str = "MediLens Medical Assistant"
    version: str = "1.0.0"
    capabilities: List[str] = [
        "Medical information queries",
        "Symptom analysis guidance",
        "Medication information",
        "Health education",
        "Diagnostic report explanations",
        "Treatment options overview"
    ]
    specializations: List[str] = [
        "Retinal/Ophthalmology",
        "Cardiology",
        "Radiology",
        "Speech Pathology",
        "General Medicine"
    ]
    limitations: List[str] = [
        "Cannot provide medical diagnoses",
        "Cannot prescribe medications",
        "Not a replacement for professional medical advice"
    ]
    status: str = "operational"


class QuickReply(BaseModel):
    """Quick reply suggestion"""
    text: str
    icon: Optional[str] = None


class SuggestedQuestionsResponse(BaseModel):
    """Response model for suggested questions"""
    questions: List[QuickReply]
    context: Optional[str] = None
