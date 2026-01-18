"""
MediLens Medical Chatbot - API Router
Endpoints for the medical chatbot functionality
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import time

from .schemas import (
    ChatRequest,
    ChatResponse,
    ChatbotInfoResponse,
    SuggestedQuestionsResponse,
    ConversationHistoryResponse,
    ChatMessage,
    MessageRole
)
from .service import chatbot_service

router = APIRouter()


@router.get("/info", response_model=ChatbotInfoResponse)
async def get_chatbot_info():
    """
    Get chatbot information and capabilities
    
    Returns information about the medical chatbot including
    its capabilities, specializations, and limitations.
    """
    return ChatbotInfoResponse()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the medical chatbot
    
    Process a user message and return an AI-generated response
    specialized in medical information.
    
    Args:
        request: ChatRequest containing the message and optional context
        
    Returns:
        ChatResponse with the AI response and metadata
    """
    try:
        result = await chatbot_service.chat(
            message=request.message,
            conversation_history=request.conversation_history,
            session_id=request.session_id,
            context=request.context
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        print(f"[Chatbot] Error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CHAT_ERROR",
                "message": "Failed to process chat message",
                "details": str(e)
            }
        )


@router.get("/suggestions", response_model=SuggestedQuestionsResponse)
async def get_suggestions(
    context: Optional[str] = Query(None, description="Current page or diagnostic context")
):
    """
    Get suggested questions based on context
    
    Returns a list of suggested questions that the user
    might want to ask, tailored to their current context.
    
    Args:
        context: Optional context string (e.g., "retinal", "cardiology")
        
    Returns:
        SuggestedQuestionsResponse with relevant questions
    """
    questions = chatbot_service.get_suggested_questions(context)
    
    return SuggestedQuestionsResponse(
        questions=questions,
        context=context
    )


@router.get("/history/{session_id}")
async def get_conversation_history(session_id: str):
    """
    Get conversation history for a session
    
    Retrieves all messages from a conversation session.
    
    Args:
        session_id: The session identifier
        
    Returns:
        Conversation history or 404 if not found
    """
    history = chatbot_service.get_conversation_history(session_id)
    
    if history is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "SESSION_NOT_FOUND",
                "message": f"No conversation found for session {session_id}"
            }
        )
    
    return {
        "session_id": session_id,
        "messages": history,
        "count": len(history)
    }


@router.delete("/history/{session_id}")
async def clear_conversation(session_id: str):
    """
    Clear conversation history for a session
    
    Removes all messages from a conversation session.
    
    Args:
        session_id: The session identifier
        
    Returns:
        Success message or 404 if session not found
    """
    success = chatbot_service.clear_conversation(session_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "SESSION_NOT_FOUND",
                "message": f"No conversation found for session {session_id}"
            }
        )
    
    return {
        "success": True,
        "message": f"Conversation {session_id} cleared successfully"
    }


@router.get("/health")
async def health_check():
    """
    Health check for the chatbot service
    
    Returns the operational status of the chatbot.
    """
    return {
        "status": "operational",
        "service": "MediLens Medical Chatbot",
        "version": "1.0.0"
    }
