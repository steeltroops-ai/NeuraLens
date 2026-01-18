"""
MediLens Medical Chatbot - Service Layer
Uses Cerebras Cloud Llama 3.3 70B for medical chatbot responses
"""

import os
import uuid
import time
from typing import List, Dict, Optional, Any
from datetime import datetime

from .schemas import ChatMessage, MessageRole, QuickReply

# Cerebras API configuration (same as explain router)
CEREBRAS_API_KEY = os.environ.get(
    "CEREBRAS_API_KEY", 
    "csk-d2ry3r6e4rf5nf9h93kj8wed2f642enwjddh644k2xm8hmwt"
)

# Try to import Cerebras SDK
try:
    from cerebras.cloud.sdk import Cerebras
    cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)
    CEREBRAS_AVAILABLE = True
    print("[Chatbot] Cerebras SDK initialized successfully")
except ImportError:
    CEREBRAS_AVAILABLE = False
    cerebras_client = None
    print("[Chatbot] Cerebras SDK not available, using fallback responses")


class MedicalChatbotService:
    """
    Medical Chatbot Service
    Uses Cerebras Llama 3.3 70B for intelligent medical responses
    """
    
    # Medical system prompt for specialized responses
    SYSTEM_PROMPT = """You are MediLens Medical Assistant, an AI-powered healthcare companion integrated into the MediLens diagnostic platform. Your role is to provide helpful, accurate, and compassionate medical information.

CORE CAPABILITIES:
- Explain medical terminology and diagnostic results
- Provide general health education
- Discuss symptoms and when to seek medical attention
- Explain medication information and interactions
- Offer preventive health guidance
- Support understanding of medical reports and imaging results

SPECIALIZATIONS (aligned with MediLens pipelines):
1. **Ophthalmology/Retinal Health**: Diabetic retinopathy, AMD, glaucoma, retinal conditions
2. **Cardiology**: ECG interpretation, heart conditions, cardiovascular health
3. **Radiology**: X-ray and imaging explanations, chest conditions
4. **Speech Pathology**: Voice disorders, speech analysis, neurological markers

GUIDELINES:
- Always be empathetic and supportive
- Use clear, accessible language while maintaining medical accuracy
- Acknowledge limitations and recommend professional consultation
- Never provide specific diagnoses or prescriptions
- Flag urgent symptoms that require immediate medical attention
- Keep responses concise but informative

RESPONSE FORMAT:
- Be concise (2-4 paragraphs max)
- Use bullet points for lists
- Highlight critical information with **bold**
- End with a brief disclaimer about consulting healthcare professionals"""

    def __init__(self):
        """Initialize the chatbot service"""
        self.model = "llama-3.3-70b"
        self.max_tokens = 1024
        self.temperature = 0.3
        
        # Conversation storage (in production, use Redis or database)
        self._conversations: Dict[str, List[Dict]] = {}
        
        # Medical knowledge snippets for context enrichment
        self._medical_context = {
            "retinal": "Retinal imaging helps detect diabetic retinopathy, macular degeneration, glaucoma, and other eye conditions through fundus photography and OCT scans.",
            "cardiology": "Cardiac assessments include ECG analysis, heart rate variability, and rhythm detection to identify arrhythmias and cardiovascular conditions.",
            "radiology": "Chest X-ray analysis can detect pneumonia, tuberculosis, cardiomegaly, and other thoracic abnormalities.",
            "speech": "Speech analysis examines vocal biomarkers for neurological conditions, respiratory health, and cognitive decline indicators."
        }
    
    def _get_session_id(self, session_id: Optional[str] = None) -> str:
        """Generate or validate session ID"""
        if session_id and session_id in self._conversations:
            return session_id
        return str(uuid.uuid4())
    
    def _build_messages(
        self,
        user_message: str,
        conversation_history: List[ChatMessage],
        context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build message list for LLM API"""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        
        # Add medical context if provided
        if context:
            context_prompt = self._enrich_context(context)
            if context_prompt:
                messages.append({
                    "role": "system",
                    "content": f"Current context: {context_prompt}"
                })
        
        # Add conversation history
        for msg in conversation_history[-10:]:  # Keep last 10 messages for context
            messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def _enrich_context(self, context: str) -> Optional[str]:
        """Enrich context with medical knowledge"""
        context_lower = context.lower()
        enriched_parts = []
        
        for key, value in self._medical_context.items():
            if key in context_lower:
                enriched_parts.append(value)
        
        if enriched_parts:
            return " ".join(enriched_parts) + f" User is currently viewing: {context}"
        
        return context
    
    async def chat(
        self,
        message: str,
        conversation_history: List[ChatMessage] = None,
        session_id: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a chat message and generate response using Cerebras Llama 3.3 70B
        """
        start_time = time.time()
        conversation_history = conversation_history or []
        session_id = self._get_session_id(session_id)
        
        # Build messages for LLM
        messages = self._build_messages(message, conversation_history, context)
        
        # Check if Cerebras is available
        if not CEREBRAS_AVAILABLE:
            print("[Chatbot] Using fallback - Cerebras not available")
            response_text = await self._fallback_response(message, context)
            processing_time = time.time() - start_time
            
            return {
                "message": response_text,
                "session_id": session_id,
                "tokens_used": None,
                "processing_time": round(processing_time, 3),
                "confidence": 0.75,
                "sources": None,
                "disclaimer": "This is AI-generated medical information for educational purposes only. Always consult a healthcare professional for medical advice."
            }
        
        try:
            # Call Cerebras API with Llama 3.3 70B
            print(f"[Chatbot] Calling Cerebras API with message: {message[:50]}...")
            
            response = cerebras_client.chat.completions.create(
                messages=messages,
                model=self.model,
                stream=False,
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=1
            )
            
            response_text = response.choices[0].message.content
            tokens_used = getattr(response.usage, 'total_tokens', None) if hasattr(response, 'usage') else None
            
            print(f"[Chatbot] Cerebras response received ({len(response_text)} chars)")
            
        except Exception as e:
            print(f"[Chatbot] Cerebras API Error: {e}")
            response_text = await self._fallback_response(message, context)
            tokens_used = None
        
        processing_time = time.time() - start_time
        
        # Store conversation
        if session_id not in self._conversations:
            self._conversations[session_id] = []
        
        self._conversations[session_id].extend([
            {"role": "user", "content": message, "timestamp": datetime.now().isoformat()},
            {"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()}
        ])
        
        return {
            "message": response_text,
            "session_id": session_id,
            "tokens_used": tokens_used,
            "processing_time": round(processing_time, 3),
            "confidence": 0.90 if tokens_used else 0.75,
            "sources": self._extract_sources(response_text),
            "disclaimer": "This is AI-generated medical information for educational purposes only. Always consult a healthcare professional for medical advice."
        }
    
    async def _fallback_response(self, message: str, context: Optional[str] = None) -> str:
        """Generate fallback response when LLM is unavailable"""
        message_lower = message.lower()
        
        # Context-aware responses
        if context:
            context_lower = context.lower()
            if "retinal" in context_lower:
                return self._retinal_fallback(message_lower)
            elif "cardio" in context_lower:
                return self._cardiology_fallback(message_lower)
            elif "radio" in context_lower:
                return self._radiology_fallback(message_lower)
            elif "speech" in context_lower:
                return self._speech_fallback(message_lower)
        
        # General medical responses
        if any(word in message_lower for word in ["symptom", "feel", "pain", "hurt"]):
            return """I understand you're experiencing some symptoms. While I can provide general information, it's important to consult with a healthcare provider for proper evaluation.

**General guidance:**
- Note when symptoms started and any patterns
- Track severity on a scale of 1-10
- Document any triggers or relieving factors
- Monitor for any changes over time

If symptoms are severe, worsening, or concerning, please seek immediate medical attention.

*This is for educational purposes only. Please consult a healthcare professional.*"""
        
        if any(word in message_lower for word in ["result", "report", "diagnosis", "test"]):
            return """I can help you understand medical results and reports. MediLens provides detailed analysis for:

- **Retinal Scans**: Detection of diabetic retinopathy, AMD, glaucoma
- **Cardiac Assessments**: ECG analysis and heart rhythm evaluation
- **Chest X-rays**: Identification of pulmonary and cardiac conditions
- **Speech Analysis**: Vocal biomarker assessment for neurological indicators

Would you like me to explain specific findings from your assessment?

*This is for educational purposes only. Please consult a healthcare professional.*"""
        
        if any(word in message_lower for word in ["hello", "hi", "hey", "help"]):
            return """Hello! I'm the MediLens Medical Assistant powered by Llama 3.3 70B.

I can help you with:
- Understanding your diagnostic results
- Explaining medical terminology
- Providing health education
- Answering questions about conditions

**How can I assist you today?**

*This is for educational purposes only. Please consult a healthcare professional.*"""
        
        # Default response
        return """Thank you for your question. I'm the MediLens Medical Assistant.

I can help you:
- Understand diagnostic results from MediLens assessments
- Learn about medical conditions and terminology
- Get general health information
- Know when to seek professional care

Could you please provide more details about what you'd like to know?

*This is for educational purposes only. Please consult a healthcare professional.*"""
    
    def _retinal_fallback(self, message: str) -> str:
        """Fallback responses for retinal/ophthalmology context"""
        if "diabetic" in message or "retinopathy" in message:
            return """**Diabetic Retinopathy Overview:**

Diabetic retinopathy is a diabetes complication affecting the blood vessels in the retina. MediLens RetinaScan AI analyzes fundus images to detect:

- **Non-Proliferative DR (NPDR)**: Early stage with microaneurysms
- **Proliferative DR (PDR)**: Advanced stage with new blood vessel growth
- **Diabetic Macular Edema**: Fluid buildup in the macula

**Key Recommendations:**
- Regular eye exams (at least annually)
- Blood sugar control is crucial
- Blood pressure management

*This is for educational purposes only. Please consult a healthcare professional.*"""
        
        return """I can help you understand your retinal assessment. The MediLens RetinaScan AI analyzes:

- Optic disc health and cup-to-disc ratio
- Macular integrity
- Blood vessel patterns
- Signs of diabetic retinopathy or glaucoma

What specific aspect would you like me to explain?

*This is for educational purposes only. Please consult a healthcare professional.*"""
    
    def _cardiology_fallback(self, message: str) -> str:
        """Fallback responses for cardiology context"""
        return """**Cardiac Assessment Overview:**

The MediLens CardioPredict AI analyzes cardiac data to evaluate:

- **Heart Rhythm**: Regular vs irregular patterns
- **Heart Rate Variability**: Autonomic nervous system indicator
- **ECG Patterns**: Detection of arrhythmias

**Normal Ranges:**
- Resting heart rate: 60-100 bpm
- Blood pressure: typically <120/80 mmHg

*This is for educational purposes only. Please consult a healthcare professional.*"""
    
    def _radiology_fallback(self, message: str) -> str:
        """Fallback responses for radiology context"""
        return """**Chest X-Ray Analysis Overview:**

The MediLens ChestXplorer AI evaluates chest radiographs for:

- **Pulmonary Conditions**: Pneumonia, tuberculosis, nodules
- **Cardiac Size**: Cardiomegaly assessment
- **Pleural Space**: Effusions or abnormalities

**Understanding Results:**
- "Clear lungs" = no obvious pathology
- "Cardiomegaly" = enlarged heart shadow
- "Infiltrates" = possible infection or fluid

*This is for educational purposes only. Please consult a healthcare professional.*"""
    
    def _speech_fallback(self, message: str) -> str:
        """Fallback responses for speech analysis context"""
        return """**Speech Analysis Overview:**

The MediLens SpeechMD AI analyzes vocal biomarkers:

- **Voice Quality**: Hoarseness, breathiness, strain
- **Speech Patterns**: Fluency, articulation, rhythm
- **Neurological Indicators**: Potential early markers

**Applications:**
- Early detection of Parkinson's markers
- Cognitive health screening
- Post-stroke speech monitoring

*This is for educational purposes only. Please consult a healthcare professional.*"""
    
    def _extract_sources(self, response: str) -> Optional[List[str]]:
        """Extract medical sources mentioned in response"""
        sources = []
        source_keywords = [
            "WHO", "CDC", "NIH", "FDA", "AHA", "ACC",
            "guidelines", "study", "research", "journal"
        ]
        
        for keyword in source_keywords:
            if keyword.lower() in response.lower():
                sources.append(keyword)
        
        return sources if sources else None
    
    def get_suggested_questions(self, context: Optional[str] = None) -> List[QuickReply]:
        """Get context-aware suggested questions"""
        base_questions = [
            QuickReply(text="What can you help me with?", icon="help"),
            QuickReply(text="Explain my results", icon="chart"),
        ]
        
        if context:
            context_lower = context.lower()
            if "retinal" in context_lower:
                return [
                    QuickReply(text="What is diabetic retinopathy?", icon="eye"),
                    QuickReply(text="Explain my retinal scan", icon="scan"),
                    QuickReply(text="How often should I get eye exams?", icon="calendar"),
                    QuickReply(text="What affects eye health?", icon="heart"),
                ]
            elif "cardio" in context_lower:
                return [
                    QuickReply(text="What does my ECG show?", icon="activity"),
                    QuickReply(text="What is a normal heart rate?", icon="heart"),
                    QuickReply(text="How to improve heart health?", icon="trending-up"),
                    QuickReply(text="Explain arrhythmia", icon="zap"),
                ]
            elif "radio" in context_lower:
                return [
                    QuickReply(text="Explain my X-ray results", icon="image"),
                    QuickReply(text="What is cardiomegaly?", icon="heart"),
                    QuickReply(text="Signs of pneumonia?", icon="thermometer"),
                    QuickReply(text="X-ray safety information", icon="shield"),
                ]
            elif "speech" in context_lower:
                return [
                    QuickReply(text="What are speech biomarkers?", icon="mic"),
                    QuickReply(text="Explain voice analysis", icon="volume-2"),
                    QuickReply(text="Signs of speech disorders?", icon="alert"),
                    QuickReply(text="How to improve voice health?", icon="trending-up"),
                ]
        
        return base_questions + [
            QuickReply(text="Health tips", icon="heart"),
            QuickReply(text="When to see a doctor?", icon="user-plus"),
        ]
    
    def get_conversation_history(self, session_id: str) -> Optional[List[Dict]]:
        """Retrieve conversation history for a session"""
        return self._conversations.get(session_id)
    
    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history for a session"""
        if session_id in self._conversations:
            del self._conversations[session_id]
            return True
        return False


# Singleton instance
chatbot_service = MedicalChatbotService()
