"""
Voice Assistant Service - Using ElevenLabs API
For accessibility and sponsor prize!
"""

import os
import base64
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Try importing ElevenLabs
try:
    from elevenlabs import generate, set_api_key, Voice, VoiceSettings
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    print("WARNING: elevenlabs not installed. pip install elevenlabs")

# Backup: Google TTS
try:
    from gtts import gTTS
    import io
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


@dataclass
class VoiceResult:
    audio_base64: str
    format: str
    duration_estimate_seconds: float
    provider: str


class VoiceAssistant:
    """
    Voice output service for accessibility
    
    Primary: ElevenLabs (high quality, sponsor prize!)
    Backup: Google TTS (free, works offline)
    """
    
    # Professional medical voice IDs from ElevenLabs
    VOICES = {
        "professional": "21m00Tcm4TlvDq8ikWAM",  # Rachel
        "calm": "EXAVITQu4vr4xnSDxMaL",  # Bella
        "warm": "pNInz6obpgDQGcFmaJgB",  # Adam
    }
    
    def __init__(self):
        self.provider = None
        
        # Try ElevenLabs first
        if ELEVENLABS_AVAILABLE:
            api_key = os.getenv("ELEVENLABS_API_KEY")
            if api_key:
                set_api_key(api_key)
                self.provider = "elevenlabs"
                print("ElevenLabs initialized")
            else:
                print("ELEVENLABS_API_KEY not set")
        
        # Fall back to gTTS
        if self.provider is None and GTTS_AVAILABLE:
            self.provider = "gtts"
            print("Using Google TTS as fallback")
        
        if self.provider is None:
            print("No voice provider available!")
    
    def speak(self, text: str, voice_style: str = "professional") -> Optional[VoiceResult]:
        """
        Convert text to speech
        
        Args:
            text: Text to speak
            voice_style: "professional", "calm", or "warm"
        
        Returns:
            VoiceResult with base64 encoded audio
        """
        if self.provider == "elevenlabs":
            return self._speak_elevenlabs(text, voice_style)
        elif self.provider == "gtts":
            return self._speak_gtts(text)
        else:
            return None
    
    def _speak_elevenlabs(self, text: str, voice_style: str) -> VoiceResult:
        """Generate speech using ElevenLabs"""
        voice_id = self.VOICES.get(voice_style, self.VOICES["professional"])
        
        audio = generate(
            text=text,
            voice=voice_id,
            model="eleven_turbo_v2"  # Fast, cost-effective
        )
        
        # Convert generator to bytes
        audio_bytes = b"".join(audio)
        
        return VoiceResult(
            audio_base64=base64.b64encode(audio_bytes).decode(),
            format="mp3",
            duration_estimate_seconds=len(text) / 15,  # ~15 chars per second
            provider="elevenlabs"
        )
    
    def _speak_gtts(self, text: str) -> VoiceResult:
        """Generate speech using Google TTS (free backup)"""
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to bytes
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return VoiceResult(
            audio_base64=base64.b64encode(audio_buffer.read()).decode(),
            format="mp3",
            duration_estimate_seconds=len(text) / 12,
            provider="gtts"
        )
    
    def explain_result(self, pipeline: str, result: Dict[str, Any]) -> Optional[VoiceResult]:
        """
        Generate voice explanation for analysis result
        
        Args:
            pipeline: "retinal", "speech", "cardiology", "radiology", "cognitive", "nri"
            result: Analysis result dictionary
        """
        text = self._generate_explanation(pipeline, result)
        return self.speak(text)
    
    def _generate_explanation(self, pipeline: str, result: Dict) -> str:
        """Generate human-friendly explanation"""
        
        if pipeline == "retinal":
            return self._explain_retinal(result)
        elif pipeline == "cardiology":
            return self._explain_cardiology(result)
        elif pipeline == "radiology":
            return self._explain_radiology(result)
        elif pipeline == "speech":
            return self._explain_speech(result)
        elif pipeline == "cognitive":
            return self._explain_cognitive(result)
        elif pipeline == "nri":
            return self._explain_nri(result)
        else:
            return f"Your analysis is complete. {result.get('recommendation', '')}"
    
    def _explain_retinal(self, result: Dict) -> str:
        score = result.get('risk_score', 0)
        category = result.get('risk_category', 'unknown')
        
        if score < 25:
            return (
                f"Great news! Your retinal scan shows a low risk score of {score}. "
                "Your eye blood vessels appear healthy with normal patterns. "
                "I recommend continuing your regular eye exams."
            )
        elif score < 50:
            return (
                f"Your retinal scan shows a moderate risk score of {score}. "
                "I've detected some minor changes in your blood vessel patterns. "
                "Consider scheduling a follow-up appointment with an eye specialist."
            )
        else:
            return (
                f"Your retinal scan shows an elevated risk score of {score}. "
                "There are some findings that need attention. "
                "Please consult with a healthcare provider soon for further evaluation."
            )
    
    def _explain_cardiology(self, result: Dict) -> str:
        rhythm = result.get('rhythm', 'sinus rhythm')
        heart_rate = result.get('heart_rate', 72)
        risk = result.get('risk_level', 'normal')
        
        if risk == "normal":
            return (
                f"Your ECG analysis is complete. "
                f"Your heart shows a {rhythm.lower()} at {heart_rate} beats per minute. "
                "This is within the normal range. Keep up the healthy lifestyle!"
            )
        else:
            return (
                f"Your ECG analysis detected a {rhythm.lower()} "
                f"with a heart rate of {heart_rate} beats per minute. "
                f"{result.get('recommendation', 'Please consult with your doctor.')}"
            )
    
    def _explain_radiology(self, result: Dict) -> str:
        finding = result.get('primary_finding', 'No findings')
        confidence = result.get('confidence', 0)
        
        if "No Significant" in finding or confidence < 30:
            return (
                "Your chest X-ray analysis is complete. "
                "No significant abnormalities were detected. "
                "Your lungs and heart appear normal."
            )
        else:
            return (
                f"Your chest X-ray analysis detected {finding.lower()} "
                f"with {confidence}% confidence. "
                f"{result.get('recommendation', 'Please follow up with your doctor.')}"
            )
    
    def _explain_speech(self, result: Dict) -> str:
        score = result.get('risk_score', 0)
        
        if score < 25:
            return (
                f"Your voice analysis is complete with a risk score of {score}. "
                "Your speech patterns appear healthy with normal characteristics."
            )
        else:
            return (
                f"Your voice analysis detected a risk score of {score}. "
                "Some variations in your speech patterns were noted. "
                "Consider discussing these findings with your healthcare provider."
            )
    
    def _explain_cognitive(self, result: Dict) -> str:
        score = result.get('overall_score', 0)
        
        return (
            f"Your cognitive assessment is complete with an overall score of {score}. "
            f"{result.get('recommendation', 'Continue exercising your mind regularly.')}"
        )
    
    def _explain_nri(self, result: Dict) -> str:
        nri_score = result.get('nri_score', 0)
        category = result.get('risk_category', 'unknown')
        
        return (
            f"Your comprehensive neurological risk assessment is complete. "
            f"Your N.R.I. score is {nri_score}, placing you in the {category} risk category. "
            f"{result.get('recommendation', '')}"
        )
