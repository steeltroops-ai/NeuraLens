# Chatbot Pipeline - Architecture Design Document

## Document Metadata
| Field | Value |
|-------|-------|
| Pipeline | Medical Chatbot Assistant |
| Version | 2.0.0 |
| Last Updated | 2026-01-17 |
| LLM Provider | Cerebras Cloud (Llama 3.3 70B) |
| Purpose | Conversational medical guidance and result interpretation |

---

## 1. Architecture Overview

```
+------------------------------------------------------------------+
|                    FRONTEND (Next.js 15)                          |
+------------------------------------------------------------------+
|  [Chat Interface]  [Message History]  [Quick Actions]             |
|         |               |                    |                    |
|         v               v                    v                    |
|  +------------------------------------------+                     |
|  |         Chat State Management            |                     |
|  |  - messages: [{role, content, timestamp}]|                     |
|  |  - context: current pipeline results     |                     |
|  |  - session_id: UUID                      |                     |
|  +------------------------------------------+                     |
|                   |                                               |
+------------------------------------------------------------------+
                    |  WebSocket / HTTP POST
                    v
+------------------------------------------------------------------+
|                    BACKEND (FastAPI)                              |
+------------------------------------------------------------------+
|  +------------------------------------------+                     |
|  |       CONVERSATION MANAGER               |                     |
|  |  - Message history (last 10)             |                     |
|  |  - Context injection                     |                     |
|  |  - Session tracking                      |                     |
|  +------------------------------------------+                     |
|                   |                                               |
|                   v                                               |
|  +------------------------------------------+                     |
|  |       CEREBRAS LLM                       |                     |
|  |  - llama-3.3-70b                         |                     |
|  |  - Medical system prompt                 |                     |
|  |  - Streaming response                    |                     |
|  +------------------------------------------+                     |
|                   |                                               |
|                   v                                               |
|  +------------------------------------------+                     |
|  |       RESPONSE HANDLER                   |                     |
|  |  - SSE streaming                         |                     |
|  |  - Safety filtering                      |                     |
|  |  - Action detection                      |                     |
|  +------------------------------------------+                     |
+------------------------------------------------------------------+
```

---

## 2. System Prompt

```python
CHATBOT_SYSTEM_PROMPT = """You are MediLens AI Assistant, a helpful and knowledgeable 
medical AI chatbot integrated into the MediLens diagnostic platform.

YOUR CAPABILITIES:
1. Explain medical test results in plain language
2. Answer questions about biomarkers and their meanings
3. Provide general health information
4. Guide users through the platform's features
5. Clarify medical terminology

IMPORTANT GUIDELINES:
- Always be helpful, empathetic, and professional
- NEVER provide definitive diagnoses
- Always recommend consulting healthcare professionals for medical decisions
- Explain that you are an AI assistant, not a doctor
- If asked about medication dosages or treatments, decline and recommend a physician
- Keep responses concise but informative
- Use bullet points for complex information
- If the user shares test results, reference them in your explanations

AVAILABLE PIPELINES (you can help explain):
- Speech Analysis: Voice biomarkers for neurological screening
- Retinal Imaging: Eye scan for diabetic retinopathy and other conditions
- Cardiology/ECG: Heart rhythm and HRV analysis
- Radiology: Chest X-ray interpretation
- Cognitive Assessment: Memory, attention, and executive function tests
- Motor Assessment: Tremor and movement analysis
- NRI Score: Combined neurological risk index

When users ask about their results, provide context about normal ranges and 
what deviations might indicate, while always emphasizing the need for 
professional medical consultation."""
```

---

## 3. Conversation Management

```python
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime
import uuid

@dataclass
class Message:
    role: str          # "user" | "assistant" | "system"
    content: str
    timestamp: datetime
    
@dataclass  
class ConversationContext:
    session_id: str
    pipeline_results: Optional[dict] = None
    current_pipeline: Optional[str] = None
    patient_info: Optional[dict] = None

class ConversationManager:
    """Manage chat conversation state and context"""
    
    MAX_HISTORY = 10  # Keep last 10 messages for context
    
    def __init__(self):
        self.sessions = {}  # session_id -> list of messages
    
    def get_or_create_session(self, session_id: str = None) -> str:
        if not session_id:
            session_id = str(uuid.uuid4())
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return session_id
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add message to conversation history"""
        msg = Message(role=role, content=content, timestamp=datetime.now())
        self.sessions[session_id].append(msg)
        
        # Trim to max history
        if len(self.sessions[session_id]) > self.MAX_HISTORY:
            self.sessions[session_id] = self.sessions[session_id][-self.MAX_HISTORY:]
    
    def get_messages_for_llm(
        self, 
        session_id: str, 
        context: ConversationContext = None
    ) -> list:
        """Format messages for LLM API call"""
        messages = [{"role": "system", "content": CHATBOT_SYSTEM_PROMPT}]
        
        # Inject context if available
        if context and context.pipeline_results:
            context_msg = self._format_context(context)
            messages.append({"role": "system", "content": context_msg})
        
        # Add conversation history
        for msg in self.sessions.get(session_id, []):
            messages.append({"role": msg.role, "content": msg.content})
        
        return messages
    
    def _format_context(self, context: ConversationContext) -> str:
        """Format pipeline results as context for the chatbot"""
        text = f"CURRENT CONTEXT:\nThe user is viewing {context.current_pipeline} results.\n"
        
        if context.pipeline_results:
            risk = context.pipeline_results.get("risk_assessment", {})
            text += f"Risk Score: {risk.get('overall_score', 'N/A')}/100\n"
            text += f"Category: {risk.get('category', 'N/A')}\n"
        
        return text
```

---

## 4. Chat Service Implementation

```python
from cerebras.cloud.sdk import Cerebras
import os

class ChatService:
    """LLM-powered chatbot service"""
    
    def __init__(self):
        self.client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        self.model = "llama-3.3-70b"
        self.conversation_manager = ConversationManager()
    
    async def chat(
        self,
        user_message: str,
        session_id: str = None,
        context: ConversationContext = None
    ) -> str:
        """Non-streaming chat response"""
        session_id = self.conversation_manager.get_or_create_session(session_id)
        self.conversation_manager.add_message(session_id, "user", user_message)
        
        messages = self.conversation_manager.get_messages_for_llm(session_id, context)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.5,
            max_tokens=600
        )
        
        assistant_msg = response.choices[0].message.content
        self.conversation_manager.add_message(session_id, "assistant", assistant_msg)
        
        return assistant_msg
    
    async def chat_stream(
        self,
        user_message: str,
        session_id: str = None,
        context: ConversationContext = None
    ):
        """Streaming chat response"""
        session_id = self.conversation_manager.get_or_create_session(session_id)
        self.conversation_manager.add_message(session_id, "user", user_message)
        
        messages = self.conversation_manager.get_messages_for_llm(session_id, context)
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.5,
            max_tokens=600,
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                full_response += text
                yield text
        
        self.conversation_manager.add_message(session_id, "assistant", full_response)
```

---

## 5. API Endpoints

```python
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/chat", tags=["chatbot"])

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    pipeline_context: Optional[str] = None
    pipeline_results: Optional[dict] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

@router.post("/")
async def chat(request: ChatRequest) -> ChatResponse:
    """Send message and get response"""
    service = ChatService()
    
    context = None
    if request.pipeline_results:
        context = ConversationContext(
            session_id=request.session_id or "",
            pipeline_results=request.pipeline_results,
            current_pipeline=request.pipeline_context
        )
    
    response = await service.chat(
        user_message=request.message,
        session_id=request.session_id,
        context=context
    )
    
    return ChatResponse(
        response=response,
        session_id=request.session_id or "new"
    )

@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """SSE streaming chat response"""
    service = ChatService()
    
    async def generator():
        async for chunk in service.chat_stream(
            user_message=request.message,
            session_id=request.session_id
        ):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generator(), media_type="text/event-stream")
```

---

## 6. Frontend Chat Component

```typescript
// ChatInterface.tsx
import { useState, useRef, useEffect } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export function ChatInterface({ pipelineContext, pipelineResults }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;
    
    const userMessage: Message = {
      role: 'user',
      content: input,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    
    // Add placeholder for assistant response
    const placeholderMsg: Message = {
      role: 'assistant',
      content: '',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, placeholderMsg]);
    
    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: input,
          session_id: sessionId,
          pipeline_context: pipelineContext,
          pipeline_results: pipelineResults
        })
      });
      
      const reader = response.body?.getReader();
      let assistantContent = '';
      
      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = new TextDecoder().decode(value);
        const parsed = parseSSE(chunk);
        if (parsed) {
          assistantContent += parsed;
          setMessages(prev => {
            const updated = [...prev];
            updated[updated.length - 1].content = assistantContent;
            return updated;
          });
        }
      }
    } catch (error) {
      console.error('Chat error:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            {msg.content}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="input-area">
        <input
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyPress={e => e.key === 'Enter' && sendMessage()}
          placeholder="Ask about your results..."
          disabled={isLoading}
        />
        <button onClick={sendMessage} disabled={isLoading}>
          Send
        </button>
      </div>
    </div>
  );
}
```

---

## 7. Quick Actions / Suggested Messages

```python
QUICK_ACTIONS = {
    "general": [
        "What does my risk score mean?",
        "What should I do next?",
        "Explain my results in simple terms",
        "Should I see a doctor?"
    ],
    "speech": [
        "What is jitter in voice analysis?",
        "Why is voice tremor important?",
        "What conditions can voice analysis detect?"
    ],
    "retinal": [
        "What are the stages of diabetic retinopathy?",
        "What does cup-to-disc ratio mean?",
        "How often should I get my eyes checked?"
    ],
    "cardiology": [
        "What is HRV and why does it matter?",
        "Is my heart rate normal?",
        "What does irregular rhythm mean?"
    ]
}
```

---

## 8. Technology Stack

```txt
# Backend
cerebras-cloud-sdk>=0.1.0
fastapi>=0.104.0
sse-starlette>=1.0.0

# Environment
CEREBRAS_API_KEY=your_key
```

---

## 9. File Structure

```
app/pipelines/chatbot/
├── __init__.py
├── ARCHITECTURE.md       # This document
├── router.py             # API endpoints
├── service.py            # ChatService
├── conversation.py       # ConversationManager
├── prompts.py            # System prompts
└── models.py             # Pydantic schemas
```

---

## 10. Safety Considerations

1. **No Diagnosis**: Always emphasize AI cannot diagnose
2. **Professional Referral**: Recommend doctor consultation
3. **Medication Restrictions**: Decline dosage/treatment advice
4. **Scope Boundaries**: Stay within explanation of results
5. **Disclaimer**: Clear AI assistant disclosure
