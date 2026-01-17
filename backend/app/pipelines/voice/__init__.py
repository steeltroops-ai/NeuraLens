"""
Voice Pipeline - ElevenLabs Integration
Voice output for accessibility
"""

from .service import VoiceAssistant
from .router import router

__all__ = ["VoiceAssistant", "router"]
