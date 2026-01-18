# Voice Assistant Pipeline - Architecture Design Document

## Document Metadata
| Field | Value |
|-------|-------|
| Pipeline | Voice Assistant (Text-to-Speech) |
| Version | 2.0.0 |
| Last Updated | 2026-01-17 |
| Purpose | Natural voice output for medical results |
| Primary Provider | ElevenLabs API |
| Fallback Provider | gTTS (Google Text-to-Speech) |

---

## 1. Pipeline Architecture Overview

```
+------------------------------------------------------------------+
|                    FRONTEND (Next.js 15)                          |
+------------------------------------------------------------------+
|                                                                   |
|  [Voice Toggle]  [Voice Selection]  [Speed Control]  [Auto-Read]  |
|         |              |                 |               |        |
|         v              v                 v               v        |
|  +------------------------------------------+                     |
|  |           Voice Settings State           |                     |
|  |  - enabled: boolean                      |                     |
|  |  - voice_id: string                      |                     |
|  |  - speed: 0.5 - 2.0                      |                     |
|  |  - auto_read: boolean                    |                     |
|  +------------------------------------------+                     |
|                        |                                          |
|                        v                                          |
|  +------------------------------------------+                     |
|  |          Text to Speak Request           |                     |
|  |  - text: string (medical result)         |                     |
|  |  - voice_id: string                      |                     |
|  |  - speed: float                          |                     |
|  +------------------------------------------+                     |
|                        |                                          |
+------------------------------------------------------------------+
                         |                                    ^
                         | HTTPS POST /api/voice/speak        |
                         v                                    |
+------------------------------------------------------------------+
|                    BACKEND (FastAPI)                              |
+------------------------------------------------------------------+
|  +------------------------------------------+                     |
|  |           INPUT LAYER                    |                     |
|  |  router.py                               |                     |
|  |  - Text length validation (<=5000 chars) |                     |
|  |  - Voice ID validation                   |                     |
|  |  - Rate limiting                         |                     |
|  +------------------------------------------+                     |
|                        |                                          |
|                        v                                          |
|  +------------------------------------------+                     |
|  |         TEXT PROCESSING LAYER            |                     |
|  |  processor.py                            |                     |
|  |  - Medical term pronunciation hints      |                     |
|  |  - SSML injection (optional)             |                     |
|  |  - Text chunking for long content        |                     |
|  +------------------------------------------+                     |
|                        |                                          |
|                        v                                          |
|  +------------------------------------------+                     |
|  |         TTS PROVIDER SELECTION           |                     |
|  |                                          |                     |
|  |  Check: ELEVENLABS_API_KEY available?    |                     |
|  |    YES --> ElevenLabs TTS                |                     |
|  |    NO  --> gTTS Fallback                 |                     |
|  |                                          |                     |
|  +------------------------------------------+                     |
|         |                        |                                |
|         v                        v                                |
|  +---------------+        +---------------+                       |
|  | ElevenLabs   |        | gTTS          |                       |
|  | API          |        | (Free)        |                       |
|  | (Premium)    |        |               |                       |
|  +---------------+        +---------------+                       |
|         |                        |                                |
|         +----------+-------------+                                |
|                    |                                              |
|                    v                                              |
|  +------------------------------------------+                     |
|  |           OUTPUT LAYER                   |                     |
|  |  - Audio bytes (MP3)                     |                     |
|  |  - Streaming response (chunked)          |                     |
|  |  - Cache headers                         |                     |
|  +------------------------------------------+                     |
+------------------------------------------------------------------+
                         |
                         | Audio Stream (MP3)
                         v
+------------------------------------------------------------------+
|                    FRONTEND (Audio Player)                        |
+------------------------------------------------------------------+
|  +------------------------------------------+                     |
|  |           HTML5 Audio Element            |                     |
|  |  - Playback controls                     |                     |
|  |  - Progress indicator                    |                     |
|  |  - Volume control                        |                     |
|  +------------------------------------------+                     |
+------------------------------------------------------------------+
```

---

## 2. ElevenLabs Integration

### 2.1 Recommended Voices
```python
ELEVENLABS_VOICES = {
    "rachel": {
        "voice_id": "21m00Tcm4TlvDq8ikWAM",
        "name": "Rachel",
        "description": "Professional female, clear articulation",
        "use_case": "Medical reports (default)"
    },
    "josh": {
        "voice_id": "TxGEqnHWrfWFTfGW9XjX",
        "name": "Josh",
        "description": "Professional male, confident tone",
        "use_case": "Alternative male voice"
    },
    "bella": {
        "voice_id": "EXAVITQu4vr4xnSDxMaL",
        "name": "Bella",
        "description": "Warm female, approachable",
        "use_case": "Patient-facing explanations"
    },
    "adam": {
        "voice_id": "pNInz6obpgDQGcFmaJgB",
        "name": "Adam",
        "description": "Clear male, neutral accent",
        "use_case": "Technical explanations"
    }
}

VOICE_SETTINGS = {
    "stability": 0.65,        # Voice consistency
    "similarity_boost": 0.75, # Speaker similarity
    "style": 0.0,             # Style exaggeration
    "use_speaker_boost": True
}
```

### 2.2 ElevenLabs Implementation
```python
from elevenlabs import ElevenLabs
import os

class VoiceService:
    """Text-to-Speech service with ElevenLabs and gTTS fallback"""
    
    def __init__(self):
        self.elevenlabs_key = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_available = bool(self.elevenlabs_key)
        
        if self.elevenlabs_available:
            self.client = ElevenLabs(api_key=self.elevenlabs_key)
    
    async def speak(
        self,
        text: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel
        speed: float = 1.0,
        optimize_latency: int = 2
    ) -> bytes:
        """
        Convert text to speech audio
        
        Args:
            text: Text to convert (max 5000 chars)
            voice_id: ElevenLabs voice ID
            speed: Speaking speed multiplier (0.5-2.0)
            optimize_latency: 1-4 (1=quality, 4=speed)
        
        Returns:
            MP3 audio bytes
        """
        if len(text) > 5000:
            text = text[:5000]
        
        if self.elevenlabs_available:
            return await self._elevenlabs_speak(text, voice_id, speed)
        else:
            return self._gtts_speak(text, speed)
    
    async def _elevenlabs_speak(
        self,
        text: str,
        voice_id: str,
        speed: float
    ) -> bytes:
        """Generate speech using ElevenLabs API"""
        
        try:
            audio = self.client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id="eleven_turbo_v2_5",  # Fastest model
                voice_settings={
                    "stability": 0.65,
                    "similarity_boost": 0.75,
                    "style": 0.0,
                    "use_speaker_boost": True,
                    "speed": speed
                }
            )
            
            # Collect audio chunks
            audio_bytes = b""
            for chunk in audio:
                audio_bytes += chunk
            
            return audio_bytes
            
        except Exception as e:
            # Fallback to gTTS on error
            return self._gtts_speak(text, speed)
    
    def _gtts_speak(self, text: str, speed: float = 1.0) -> bytes:
        """Fallback: Generate speech using gTTS"""
        from gtts import gTTS
        from io import BytesIO
        
        # gTTS doesn't support speed directly, but we can set slow=True/False
        slow = speed < 0.8
        
        tts = gTTS(text=text, lang='en', slow=slow)
        
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        return audio_buffer.read()
    
    async def speak_streaming(
        self,
        text: str,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM"
    ):
        """
        Stream audio chunks for faster first-byte time
        
        Yields:
            Audio chunks (bytes)
        """
        if not self.elevenlabs_available:
            # gTTS doesn't support streaming, return all at once
            yield self._gtts_speak(text)
            return
        
        audio_stream = self.client.text_to_speech.convert_as_stream(
            voice_id=voice_id,
            text=text,
            model_id="eleven_turbo_v2_5"
        )
        
        for chunk in audio_stream:
            yield chunk
```

---

## 3. Text Processing

### 3.1 Medical Pronunciation Hints
```python
MEDICAL_PRONUNCIATIONS = {
    # Common medical terms with pronunciation hints
    "HbA1c": "H-B-A-1-C",
    "mmHg": "millimeters of mercury",
    "mg/dL": "milligrams per deciliter",
    "ECG": "E-C-G",
    "EKG": "E-K-G",
    "MRI": "M-R-I",
    "CT": "C-T scan",
    "BP": "blood pressure",
    "HR": "heart rate",
    "HRV": "heart rate variability",
    "RMSSD": "R-M-S-S-D",
    "SDNN": "S-D-N-N",
    "AFib": "atrial fibrillation",
    "PVC": "premature ventricular contraction",
    "DR": "diabetic retinopathy",
    "AMD": "age-related macular degeneration",
    "NRI": "Neurological Risk Index",
    "MCI": "mild cognitive impairment"
}

def preprocess_for_speech(text: str) -> str:
    """
    Preprocess medical text for better TTS pronunciation
    """
    for abbrev, pronunciation in MEDICAL_PRONUNCIATIONS.items():
        # Replace with pronunciation hint
        text = text.replace(abbrev, pronunciation)
    
    # Add pauses after punctuation
    text = text.replace(". ", "... ")
    text = text.replace(": ", ": ... ")
    
    # Format numbers for speech
    text = format_numbers_for_speech(text)
    
    return text

def format_numbers_for_speech(text: str) -> str:
    """Format numbers for clearer speech"""
    import re
    
    # Format percentages
    text = re.sub(r'(\d+)%', r'\1 percent', text)
    
    # Format decimal numbers
    text = re.sub(r'(\d+)\.(\d+)', r'\1 point \2', text)
    
    return text
```

---

## 4. API Specification

### 4.1 Endpoints
```python
from fastapi import APIRouter, Response
from pydantic import BaseModel, Field
from typing import Optional

router = APIRouter(prefix="/api/voice", tags=["voice"])

class SpeakRequest(BaseModel):
    text: str = Field(..., max_length=5000)
    voice_id: Optional[str] = "21m00Tcm4TlvDq8ikWAM"
    speed: Optional[float] = Field(1.0, ge=0.5, le=2.0)

class ExplainRequest(BaseModel):
    term: str
    context: Optional[str] = None

@router.post("/speak")
async def text_to_speech(request: SpeakRequest):
    """
    Convert text to speech audio
    
    Returns: MP3 audio file
    """
    voice_service = VoiceService()
    
    # Preprocess text
    processed_text = preprocess_for_speech(request.text)
    
    # Generate audio
    audio_bytes = await voice_service.speak(
        text=processed_text,
        voice_id=request.voice_id,
        speed=request.speed
    )
    
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": "inline; filename=speech.mp3"
        }
    )

@router.post("/speak/stream")
async def text_to_speech_stream(request: SpeakRequest):
    """
    Stream text to speech for faster playback start
    
    Returns: Chunked MP3 audio stream
    """
    from fastapi.responses import StreamingResponse
    
    voice_service = VoiceService()
    processed_text = preprocess_for_speech(request.text)
    
    return StreamingResponse(
        voice_service.speak_streaming(processed_text, request.voice_id),
        media_type="audio/mpeg"
    )

@router.post("/explain")
async def explain_term(request: ExplainRequest):
    """
    Generate audio explanation of a medical term
    """
    explanation = get_medical_explanation(request.term, request.context)
    
    voice_service = VoiceService()
    audio_bytes = await voice_service.speak(explanation)
    
    return Response(
        content=audio_bytes,
        media_type="audio/mpeg"
    )
```

---

## 5. Frontend Audio Player Component

```typescript
// VoicePlayer.tsx
import { useState, useRef, useEffect } from 'react';

interface VoicePlayerProps {
  text: string;
  autoPlay?: boolean;
  voiceId?: string;
  onComplete?: () => void;
}

export function VoicePlayer({ 
  text, 
  autoPlay = false,
  voiceId = '21m00Tcm4TlvDq8ikWAM',
  onComplete
}: VoicePlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const audioRef = useRef<HTMLAudioElement>(null);
  
  const speak = async () => {
    if (!text) return;
    
    setIsLoading(true);
    
    try {
      const response = await fetch('/api/voice/speak', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, voice_id: voiceId })
      });
      
      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      
      if (audioRef.current) {
        audioRef.current.src = audioUrl;
        audioRef.current.play();
        setIsPlaying(true);
      }
    } catch (error) {
      console.error('Voice playback failed:', error);
    } finally {
      setIsLoading(false);
    }
  };
  
  const togglePlayback = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };
  
  useEffect(() => {
    if (autoPlay && text) {
      speak();
    }
  }, [text, autoPlay]);
  
  return (
    <div className="voice-player">
      <audio 
        ref={audioRef}
        onEnded={() => {
          setIsPlaying(false);
          onComplete?.();
        }}
        onTimeUpdate={(e) => {
          const audio = e.currentTarget;
          setProgress((audio.currentTime / audio.duration) * 100);
        }}
      />
      
      <button 
        onClick={isPlaying ? togglePlayback : speak}
        disabled={isLoading}
        className="voice-button"
      >
        {isLoading ? (
          <LoadingSpinner />
        ) : isPlaying ? (
          <PauseIcon />
        ) : (
          <PlayIcon />
        )}
      </button>
      
      {isPlaying && (
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
    </div>
  );
}
```

---

## 6. Cost Management

### 6.1 Usage Tracking
```python
class UsageTracker:
    """Track TTS API usage for cost management"""
    
    # ElevenLabs pricing (approximate)
    ELEVENLABS_FREE_CHARS = 10000  # per month
    ELEVENLABS_COST_PER_1K = 0.30  # after free tier
    
    def __init__(self):
        self.monthly_chars = 0
        self.monthly_reset = datetime.now().replace(day=1)
    
    def track_usage(self, char_count: int, provider: str):
        """Track character usage"""
        if datetime.now().month != self.monthly_reset.month:
            self.monthly_chars = 0
            self.monthly_reset = datetime.now().replace(day=1)
        
        self.monthly_chars += char_count
        
        # Log if approaching limits
        if provider == "elevenlabs":
            if self.monthly_chars > self.ELEVENLABS_FREE_CHARS * 0.8:
                logger.warning(f"Approaching ElevenLabs free tier limit: {self.monthly_chars}")
    
    def should_use_fallback(self) -> bool:
        """Determine if should use gTTS to save costs"""
        return self.monthly_chars > self.ELEVENLABS_FREE_CHARS * 0.9
```

### 6.2 Caching Strategy
```python
import hashlib
from functools import lru_cache

class AudioCache:
    """Cache generated audio to reduce API calls"""
    
    def __init__(self, max_size_mb: int = 100):
        self.cache = {}
        self.max_size = max_size_mb * 1024 * 1024
        self.current_size = 0
    
    def _get_key(self, text: str, voice_id: str, speed: float) -> str:
        """Generate cache key from parameters"""
        content = f"{text}:{voice_id}:{speed}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, text: str, voice_id: str, speed: float) -> bytes | None:
        """Retrieve cached audio if available"""
        key = self._get_key(text, voice_id, speed)
        return self.cache.get(key)
    
    def set(self, text: str, voice_id: str, speed: float, audio: bytes):
        """Cache audio bytes"""
        key = self._get_key(text, voice_id, speed)
        
        # Check size limits
        if self.current_size + len(audio) > self.max_size:
            self._evict_oldest()
        
        self.cache[key] = audio
        self.current_size += len(audio)
```

---

## 7. Technology Stack

### Backend Dependencies
```txt
# Core
fastapi>=0.104.0
pydantic>=2.0.0

# TTS Providers
elevenlabs>=0.2.0
gtts>=2.4.0

# Audio Processing (optional)
pydub>=0.25.1  # For audio manipulation
```

### Environment Variables
```bash
# Required for premium TTS
ELEVENLABS_API_KEY=your_api_key_here

# Optional: OpenAI TTS fallback
OPENAI_API_KEY=your_openai_key
```

---

## 8. File Structure

```
app/pipelines/voice/
├── __init__.py
├── ARCHITECTURE.md         # This document
├── router.py               # FastAPI endpoints
├── service.py              # VoiceService class
├── processor.py            # Text preprocessing
├── cache.py                # Audio caching
└── models.py               # Pydantic schemas
```

---

## 9. Clinical Use Cases

| Use Case | Example Text |
|----------|--------------|
| Risk Summary | "Your overall neurological risk score is 28 out of 100, indicating low risk." |
| Biomarker Alert | "Your heart rate variability is below normal range. Please consult your physician." |
| Recommendation | "Based on your results, we recommend a follow-up examination in 6 months." |
| Term Explanation | "Diabetic retinopathy refers to damage to the blood vessels in the retina caused by diabetes." |
