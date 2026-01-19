"""
Voice Assistant Service - Amazon Polly TTS Integration
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
import logging
from typing import Optional, AsyncGenerator
from dataclasses import dataclass

from dotenv import load_dotenv

from .processor import preprocess_for_speech, get_medical_explanation
from .cache import audio_cache, usage_tracker

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Amazon Polly via boto3
POLLY_AVAILABLE = False
polly_client = None

try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    POLLY_AVAILABLE = True
    logger.info("boto3 SDK loaded successfully for Amazon Polly")
except ImportError:
    logger.warning("boto3 not installed. Run: pip install boto3")


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


# Amazon Polly Voice Options
POLLY_VOICES = {
    # Neural voices (high quality)
    "joanna": {
        "voice_id": "Joanna",
        "name": "Joanna",
        "description": "Professional female, American English",
        "engine": "neural",
        "use_case": "Medical reports (default)"
    },
    "matthew": {
        "voice_id": "Matthew",
        "name": "Matthew", 
        "description": "Professional male, American English",
        "engine": "neural",
        "use_case": "Technical explanations"
    },
    "amy": {
        "voice_id": "Amy",
        "name": "Amy",
        "description": "Professional female, British English",
        "engine": "neural",
        "use_case": "Calm explanations"
    },
    "brian": {
        "voice_id": "Brian",
        "name": "Brian",
        "description": "Professional male, British English",
        "engine": "neural",
        "use_case": "Formal reports"
    },
    "ruth": {
        "voice_id": "Ruth",
        "name": "Ruth",
        "description": "Warm female, American English",
        "engine": "neural",
        "use_case": "Patient-facing"
    },
}

# Default voice
DEFAULT_VOICE = "joanna"


class VoiceService:
    """
    Amazon Polly Text-to-Speech Service for MediLens
    
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
        self.polly_client = None
        self.provider = None
        
        # Initialize Amazon Polly client
        if not POLLY_AVAILABLE:
            logger.error("boto3 not installed! Run: pip install boto3")
            return
        
        # Load AWS credentials from environment
        load_dotenv(override=True)
        
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "ap-south-1")
        
        if not aws_access_key or not aws_secret_key:
            logger.error("AWS credentials not found in environment!")
            logger.error("Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")
            return
        
        try:
            self.polly_client = boto3.client(
                'polly',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region
            )
            self.provider = "polly"
            logger.info(f"Amazon Polly client initialized (region: {aws_region})")
        except Exception as e:
            logger.error(f"Amazon Polly initialization failed: {e}")
    
    def get_voice_id(self, voice: str) -> str:
        """Get Polly voice ID from name"""
        voice = voice.lower()
        if voice in POLLY_VOICES:
            return POLLY_VOICES[voice]["voice_id"]
        # Default to Joanna if not found
        return "Joanna"
    
    def get_voice_engine(self, voice: str) -> str:
        """Get engine type for voice (neural or standard)"""
        voice = voice.lower()
        if voice in POLLY_VOICES:
            return POLLY_VOICES[voice].get("engine", "neural")
        return "neural"
    
    async def speak(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        use_cache: bool = True
    ) -> Optional[VoiceResult]:
        """
        Convert text to speech using Amazon Polly
        
        Args:
            text: Text to speak (max 3000 chars per request)
            voice: Voice name (joanna, matthew, amy, brian, ruth)
            use_cache: Use audio caching
            
        Returns:
            VoiceResult with audio bytes and metadata
        """
        if not text:
            return None
        
        if not self.polly_client:
            logger.error("Polly client not initialized")
            return None
        
        # Truncate if too long (Polly limit is 3000 chars)
        original_text = text
        if len(text) > 3000:
            text = text[:3000]
        
        # Preprocess for better pronunciation
        processed_text = preprocess_for_speech(text)
        voice_id = self.get_voice_id(voice)
        engine = self.get_voice_engine(voice)
        
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
        
        # Generate with Amazon Polly
        try:
            response = self.polly_client.synthesize_speech(
                Engine=engine,
                LanguageCode='en-US',
                OutputFormat='mp3',
                SampleRate='24000',
                Text=processed_text,
                TextType='text',
                VoiceId=voice_id
            )
            
            if 'AudioStream' in response:
                audio_bytes = response['AudioStream'].read()
                
                if len(audio_bytes) < 100:
                    logger.error("Amazon Polly returned empty audio")
                    return None
                
                logger.info(f"Amazon Polly TTS generated {len(audio_bytes)} bytes")
                
                result = VoiceResult(
                    audio_base64=base64.b64encode(audio_bytes).decode(),
                    audio_bytes=audio_bytes,
                    format="mp3",
                    duration_estimate_seconds=len(text) / 15,
                    provider="polly",
                    characters_used=len(original_text)
                )
                
                # Track usage
                usage_tracker.track_usage(len(original_text), "polly")
                
                # Cache result
                if use_cache:
                    audio_cache.set(
                        processed_text, voice_id, 1.0,
                        result.audio_bytes, result.provider
                    )
                
                return result
            else:
                logger.error("Amazon Polly returned no audio stream")
                return None
                
        except Exception as e:
            logger.error(f"Amazon Polly TTS failed: {e}")
            return None
    
    async def speak_streaming(
        self,
        text: str,
        voice_id: str = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio (Polly doesn't support true streaming, returns all at once)
        
        Yields audio bytes
        """
        if voice_id is None:
            voice_id = self.get_voice_id(DEFAULT_VOICE)
        
        processed_text = preprocess_for_speech(text)
        
        result = await self.speak(processed_text)
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
        """Get list of available Polly voices"""
        return [
            {
                "id": key,
                "voice_id": info["voice_id"],
                "name": info["name"],
                "description": info["description"],
                "use_case": info["use_case"],
                "provider": "polly"
            }
            for key, info in POLLY_VOICES.items()
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
        voice: Voice name (joanna, matthew, amy, brian, ruth)
        
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
