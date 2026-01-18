"""
Voice Assistant Service - ElevenLabs TTS Integration
Designed for integration with ALL pipelines to speak LLM-generated explanations

Usage from any pipeline:
    from app.pipelines.voice import speak_text, speak_result
    
    # Simple text to speech
    audio_base64 = await speak_text("Your results are ready")
    
    # Speak pipeline results
    audio_base64 = await speak_result("speech", result_dict)
"""

import os
import base64
import time
import logging
from typing import Optional, AsyncGenerator
from dataclasses import dataclass
from io import BytesIO

from dotenv import load_dotenv

from .processor import preprocess_for_speech, get_medical_explanation
from .cache import audio_cache, usage_tracker

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Try importing ElevenLabs (new SDK)
ELEVENLABS_AVAILABLE = False
elevenlabs_client = None

try:
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_AVAILABLE = True
    logger.info("ElevenLabs SDK loaded successfully")
except ImportError:
    logger.warning("elevenlabs not installed. Run: pip install elevenlabs")

# Backup: Google TTS
GTTS_AVAILABLE = False
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    logger.warning("gtts not installed. Run: pip install gtts")


@dataclass
class VoiceResult:
    """Result of TTS generation"""
    audio_base64: str
    audio_bytes: bytes
    format: str
    duration_estimate_seconds: float
    provider: str
    characters_used: int
    cached: bool = False


# ElevenLabs Voice IDs (from their voice library)
ELEVENLABS_VOICES = {
    # Professional voices for medical content
    "rachel": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",
        "name": "Rachel",
        "description": "Professional female, clear articulation",
        "use_case": "Medical reports (default)"
    },
    "george": {
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",
        "name": "George",
        "description": "Warm male, British accent",
        "use_case": "Calm explanations"
    },
    "josh": {
        "voice_id": "TxGEqnHWrfWFTfGW9XjX",
        "name": "Josh",
        "description": "Professional male, American",
        "use_case": "Technical details"
    },
    "bella": {
        "voice_id": "EXAVITQu4vr4xnSDxMaL",
        "name": "Bella",
        "description": "Warm female, approachable",
        "use_case": "Patient-facing"
    },
    "adam": {
        "voice_id": "pNInz6obpgDQGcFmaJgB",
        "name": "Adam",
        "description": "Clear male, neutral accent",
        "use_case": "Technical explanations"
    },
}

# Default voice for all pipelines
DEFAULT_VOICE = "rachel"
DEFAULT_MODEL = "eleven_multilingual_v2"
DEFAULT_OUTPUT_FORMAT = "mp3_44100_128"


class VoiceService:
    """
    ElevenLabs Text-to-Speech Service for MediLens
    
    Integration with all pipelines:
    - Speech Analysis -> Speak biomarker explanations
    - Retinal Analysis -> Speak eye condition findings
    - Cardiology -> Speak ECG interpretation
    - Radiology -> Speak X-ray findings
    - Cognitive -> Speak assessment results
    - Motor -> Speak movement analysis
    - NRI Fusion -> Speak overall risk summary
    - AI Explanation -> Speak Cerebras LLM outputs
    """
    
    def __init__(self):
        self.elevenlabs_client = None
        self.provider = None
        
        
        # Initialize ElevenLabs client (REQUIRED - no fallback)
        # Force reload .env to ensure fresh key
        from dotenv import load_dotenv
        load_dotenv(override=True)
        
        api_key = os.getenv("ELEVENLABS_API_KEY")
        
        if not api_key:
            logger.error("ELEVENLABS_API_KEY not found in environment!")
            logger.error("Voice TTS requires a valid ElevenLabs API key")
            return
            
        # Clean key
        api_key = api_key.strip()
        logger.info(f"ElevenLabs Key Loaded: {api_key[:5]}...{api_key[-5:]}")
        
        if not ELEVENLABS_AVAILABLE:
            logger.error("ElevenLabs SDK not installed! Run: pip install elevenlabs")
            return
        
        try:
            self.elevenlabs_client = ElevenLabs(api_key=api_key)
            self.provider = "elevenlabs"
            # verify client
            logger.info("ElevenLabs client initialized successfully")
        except Exception as e:
            logger.error(f"ElevenLabs initialization failed: {e}")
            logger.error("Voice TTS will not be available")

    
    def get_voice_id(self, voice: str) -> str:
        """Get ElevenLabs voice ID from name"""
        voice = voice.lower()
        if voice in ELEVENLABS_VOICES:
            return ELEVENLABS_VOICES[voice]["voice_id"]
        # Assume it's already a voice ID
        return voice
    
    async def speak(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        model: str = DEFAULT_MODEL,
        use_cache: bool = True
    ) -> Optional[VoiceResult]:
        """
        Convert text to speech using ElevenLabs or gTTS fallback
        
        Args:
            text: Text to speak (max 5000 chars)
            voice: Voice name or ElevenLabs voice ID
            model: ElevenLabs model (eleven_multilingual_v2, eleven_turbo_v2_5)
            use_cache: Use audio caching
            
        Returns:
            VoiceResult with audio bytes and metadata
        """
        if not text:
            return None
        
        # Truncate if too long
        original_text = text
        if len(text) > 5000:
            text = text[:5000]
        
        # Preprocess for better pronunciation
        processed_text = preprocess_for_speech(text)
        voice_id = self.get_voice_id(voice)
        
        # Check cache first
        if use_cache:
            cached = audio_cache.get(processed_text, voice_id, 1.0)
            if cached:
                audio_bytes, cached_provider = cached
                return VoiceResult(
                    audio_base64=base64.b64encode(audio_bytes).decode(),
                    audio_bytes=audio_bytes,
                    format="mp3",
                    duration_estimate_seconds=len(text) / 15,
                    provider=cached_provider,
                    characters_used=len(original_text),
                    cached=True
                )
        
        # Try ElevenLabs first, fallback to gTTS if quota exceeded
        result = None
        try:
            if self.provider == "elevenlabs":
                result = await self._elevenlabs_speak(processed_text, voice_id, model)
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"ElevenLabs failed: {error_msg}")
            
            # Fallback to gTTS if ElevenLabs quota exceeded
            if "quota" in error_msg.lower() or "credits" in error_msg.lower():
                logger.info("Falling back to gTTS due to ElevenLabs quota limit")
                result = await self._gtts_speak(processed_text)
            else:
                raise
        
        # If no result from ElevenLabs (not configured), use gTTS
        if result is None and GTTS_AVAILABLE:
            result = await self._gtts_speak(processed_text)

        
        if result:
            # Track usage
            usage_tracker.track_usage(len(original_text), result.provider)
            
            # Cache result
            if use_cache:
                audio_cache.set(
                    processed_text, voice_id, 1.0,
                    result.audio_bytes, result.provider
                )
        
        return result
    
    async def _elevenlabs_speak(
        self,
        text: str,
        voice_id: str,
        model: str = DEFAULT_MODEL
    ) -> Optional[VoiceResult]:
        """
        Generate speech using ElevenLabs API
        
        Uses the official ElevenLabs SDK pattern:
        audio = client.text_to_speech.convert(...)
        """
        if not self.elevenlabs_client:
            raise Exception("ElevenLabs client not initialized. Check ELEVENLABS_API_KEY in .env")
        
        try:
            # Use ElevenLabs SDK (per quickstart)
            audio_generator = self.elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model,
                output_format=DEFAULT_OUTPUT_FORMAT,
            )
            
            # Collect audio chunks from generator
            audio_bytes = b""
            for chunk in audio_generator:
                audio_bytes += chunk
            
            if len(audio_bytes) < 100:
                raise Exception("ElevenLabs returned empty audio")
            
            logger.info(f"ElevenLabs TTS generated {len(audio_bytes)} bytes")
            
            return VoiceResult(
                audio_base64=base64.b64encode(audio_bytes).decode(),
                audio_bytes=audio_bytes,
                format="mp3",
                duration_estimate_seconds=len(text) / 15,
                provider="elevenlabs",
                characters_used=len(text)
            )
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"ElevenLabs TTS failed: {error_msg}")
            
            # Check for quota exceeded
            if "quota" in error_msg.lower() or "credits" in error_msg.lower():
                raise Exception("ElevenLabs quota exceeded - credits remaining insufficient")
            elif "Paid Subscription" in error_msg or "abuse" in error_msg.lower():
                raise Exception("ElevenLabs quota exceeded - free tier exhausted")
            elif "invalid" in error_msg.lower() or "unauthorized" in error_msg.lower():
                raise Exception("ElevenLabs API key is invalid. Please check ELEVENLABS_API_KEY in .env")
            else:
                raise Exception(f"ElevenLabs TTS error: {error_msg[:200]}")

    
    async def _gtts_speak(self, text: str) -> Optional[VoiceResult]:
        """Fallback: Generate speech using Google TTS (free)"""
        if not GTTS_AVAILABLE:
            return None
        
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            
            audio_buffer = BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            audio_bytes = audio_buffer.read()
            
            return VoiceResult(
                audio_base64=base64.b64encode(audio_bytes).decode(),
                audio_bytes=audio_bytes,
                format="mp3",
                duration_estimate_seconds=len(text) / 12,
                provider="gtts",
                characters_used=len(text)
            )
            
        except Exception as e:
            logger.error(f"gTTS failed: {e}")
            return None
    
    async def speak_streaming(
        self,
        text: str,
        voice_id: str = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks for faster playback start
        
        Yields audio chunks as they're generated
        """
        if voice_id is None:
            voice_id = self.get_voice_id(DEFAULT_VOICE)
        
        processed_text = preprocess_for_speech(text)
        
        if not self.elevenlabs_client:
            # gTTS doesn't stream - return all at once
            result = await self._gtts_speak(processed_text)
            if result:
                yield result.audio_bytes
            return
        
        try:
            audio_stream = self.elevenlabs_client.text_to_speech.convert_as_stream(
                text=processed_text,
                voice_id=voice_id,
                model_id=DEFAULT_MODEL,
            )
            
            for chunk in audio_stream:
                yield chunk
                
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            result = await self._gtts_speak(processed_text)
            if result:
                yield result.audio_bytes
    
    async def speak_explanation(
        self,
        explanation_text: str,
        voice: str = DEFAULT_VOICE
    ) -> Optional[VoiceResult]:
        """
        Speak LLM-generated explanation
        
        Called by AI Explanation Pipeline to vocalize Cerebras outputs
        
        Args:
            explanation_text: Text generated by Cerebras LLM
            voice: Voice to use
        """
        return await self.speak(explanation_text, voice=voice)
    
    async def speak_pipeline_result(
        self,
        pipeline: str,
        result: dict,
        voice: str = DEFAULT_VOICE
    ) -> Optional[VoiceResult]:
        """
        Generate and speak explanation for pipeline result
        
        Args:
            pipeline: Pipeline name (speech, retinal, cardiology, etc.)
            result: Pipeline result dictionary
            voice: Voice to use
        """
        explanation = self._generate_pipeline_explanation(pipeline, result)
        return await self.speak(explanation, voice=voice)
    
    def _generate_pipeline_explanation(self, pipeline: str, result: dict) -> str:
        """Generate human-readable explanation for pipeline result"""
        
        handlers = {
            "speech": self._explain_speech,
            "retinal": self._explain_retinal,
            "cardiology": self._explain_cardiology,
            "radiology": self._explain_radiology,
            "cognitive": self._explain_cognitive,
            "motor": self._explain_motor,
            "nri": self._explain_nri,
        }
        
        handler = handlers.get(pipeline)
        if handler:
            return handler(result)
        
        return f"Your {pipeline} analysis is complete. {result.get('recommendation', '')}"
    
    def _explain_speech(self, result: dict) -> str:
        score = result.get('risk_score', 0)
        if isinstance(score, float) and score <= 1:
            score = int(score * 100)
        
        if score < 25:
            return (
                f"Your voice analysis shows a low risk score of {score}. "
                "Your speech patterns appear healthy and within normal range."
            )
        elif score < 50:
            return (
                f"Your voice analysis shows a moderate risk score of {score}. "
                "Some minor variations in speech patterns were detected. "
                "We recommend a follow-up assessment in 6 months."
            )
        else:
            return (
                f"Your voice analysis detected an elevated risk score of {score}. "
                "Please consult with a healthcare provider for further evaluation."
            )
    
    def _explain_retinal(self, result: dict) -> str:
        score = result.get('risk_score', 0)
        if isinstance(score, float) and score <= 1:
            score = int(score * 100)
        
        conditions = result.get('conditions', [])
        
        if score < 25:
            return (
                f"Great news! Your retinal scan shows a low risk score of {score}. "
                "Your eye blood vessels appear healthy."
            )
        else:
            condition_text = ", ".join(conditions[:2]) if conditions else "some changes"
            return (
                f"Your retinal scan shows a risk score of {score}. "
                f"We detected {condition_text}. "
                "Please schedule a follow-up with an eye specialist."
            )
    
    def _explain_cardiology(self, result: dict) -> str:
        rhythm = result.get('rhythm', 'normal sinus rhythm')
        heart_rate = result.get('heart_rate', 72)
        
        return (
            f"Your ECG analysis is complete. "
            f"Your heart shows {rhythm} at {heart_rate} beats per minute. "
            f"{result.get('recommendation', '')}"
        )
    
    def _explain_radiology(self, result: dict) -> str:
        finding = result.get('primary_finding', 'No significant findings')
        confidence = result.get('confidence', 0)
        
        if confidence < 30:
            return "Your chest X-ray analysis shows no significant abnormalities."
        
        return (
            f"Your chest X-ray analysis detected {finding.lower()} "
            f"with {int(confidence)} percent confidence. "
            "Please consult with your healthcare provider."
        )
    
    def _explain_cognitive(self, result: dict) -> str:
        score = result.get('overall_score', result.get('risk_score', 0))
        
        return (
            f"Your cognitive assessment is complete with an overall score of {score}. "
            f"{result.get('recommendation', 'Continue regular mental exercises.')}"
        )
    
    def _explain_motor(self, result: dict) -> str:
        score = result.get('risk_score', 0)
        if isinstance(score, float) and score <= 1:
            score = int(score * 100)
        
        if score < 25:
            return f"Your motor assessment shows normal movement patterns with a risk score of {score}."
        
        return (
            f"Your motor assessment detected a risk score of {score}. "
            "Some movement variations were noted. "
            "Consider consulting a movement specialist."
        )
    
    def _explain_nri(self, result: dict) -> str:
        nri = result.get('nri_score', result.get('overall_score', 0))
        category = result.get('risk_category', 'unknown')
        
        return (
            f"Your comprehensive neurological risk assessment is complete. "
            f"Your overall score is {nri}, placing you in the {category} risk category. "
            f"{result.get('recommendation', '')}"
        )
    
    def get_available_voices(self) -> list:
        """Get list of available ElevenLabs voices"""
        return [
            {
                "id": key,
                "voice_id": info["voice_id"],
                "name": info["name"],
                "description": info["description"],
                "use_case": info["use_case"],
                "provider": self.provider or "none"
            }
            for key, info in ELEVENLABS_VOICES.items()
        ]


# Global service instance
voice_service = VoiceService()


# ============================================================
# CONVENIENCE FUNCTIONS FOR USE IN OTHER PIPELINES
# ============================================================

async def speak_text(
    text: str,
    voice: str = DEFAULT_VOICE
) -> Optional[str]:
    """
    Simple function to speak text and get base64 audio
    
    Usage in any pipeline:
        from app.pipelines.voice import speak_text
        audio_b64 = await speak_text("Your results are ready")
    
    Args:
        text: Text to speak
        voice: Voice name (rachel, josh, bella, adam, george)
        
    Returns:
        Base64 encoded MP3 audio or None
    """
    result = await voice_service.speak(text, voice=voice)
    return result.audio_base64 if result else None


async def speak_result(
    pipeline: str,
    result: dict,
    voice: str = DEFAULT_VOICE
) -> Optional[str]:
    """
    Speak pipeline result explanation
    
    Usage:
        from app.pipelines.voice import speak_result
        audio_b64 = await speak_result("speech", {"risk_score": 0.23})
    
    Args:
        pipeline: Pipeline name (speech, retinal, cardiology, etc.)
        result: Pipeline result dictionary
        voice: Voice to use
        
    Returns:
        Base64 encoded MP3 audio or None
    """
    result = await voice_service.speak_pipeline_result(pipeline, result, voice)
    return result.audio_base64 if result else None


async def speak_llm_explanation(
    explanation: str,
    voice: str = DEFAULT_VOICE
) -> Optional[str]:
    """
    Speak LLM-generated explanation (from Cerebras)
    
    Integration with AI Explanation Pipeline:
        from app.pipelines.voice import speak_llm_explanation
        audio_b64 = await speak_llm_explanation(llm_response)
    
    Args:
        explanation: Text generated by Cerebras LLM
        voice: Voice to use
        
    Returns:
        Base64 encoded MP3 audio or None
    """
    result = await voice_service.speak_explanation(explanation, voice=voice)
    return result.audio_base64 if result else None


async def get_audio_bytes(
    text: str,
    voice: str = DEFAULT_VOICE
) -> Optional[bytes]:
    """
    Get raw audio bytes for a text
    
    Args:
        text: Text to speak
        voice: Voice to use
        
    Returns:
        Raw MP3 bytes or None
    """
    result = await voice_service.speak(text, voice=voice)
    return result.audio_bytes if result else None
