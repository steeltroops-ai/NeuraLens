# MediLens Voice Assistant Pipeline PRD

## Document Info
| Field | Value |
|-------|-------|
| Version | 2.0.0 |
| Priority | P2 - Medium (Accessibility) |
| Est. Dev Time | 3 hours |
| Integration | ElevenLabs API |

---

## 1. Overview

### Purpose
Provide natural voice output for medical results and recommendations to:
- **Improve Accessibility** for visually impaired users
- **Enhance UX** with conversational feedback
- **Win Sponsor Prize** (ElevenLabs integration)

### Technology
**ElevenLabs** - Best-in-class neural text-to-speech with:
- Human-like voice quality
- Medical terminology support
- Multiple voice options
- Low latency streaming

---

## 2. Technology Stack

| Component | Technology | Fallback |
|-----------|-----------|----------|
| **Primary TTS** | ElevenLabs API | gTTS |
| **Backend** | elevenlabs Python SDK | gtts |
| **Frontend** | HTML5 Audio API | Same |

### Installation
```bash
pip install elevenlabs gtts
```

### API Key
```bash
# .env
ELEVENLABS_API_KEY=your_api_key_here
```

---

## 3. Voice Options

### ElevenLabs Recommended Voices

| Voice | ID | Style | Use For |
|-------|-----|-------|---------|
| **Rachel** | `21m00Tcm4TlvDq8ikWAM` | Professional F | Medical results |
| **Josh** | `TxGEqnHWrfWFTfGW9XjX` | Professional M | Formal reports |
| **Bella** | `EXAVITQu4vr4xnSDxMaL` | Warm F | Friendly feedback |
| **Adam** | `pNInz6obpgDQGcFmaJgB` | Clear M | Technical details |

### Voice Settings
```python
voice_settings = {
    "stability": 0.5,        # 0-1, higher = more consistent
    "similarity_boost": 0.75, # 0-1, higher = more like original
    "style": 0.5,            # 0-1, expressiveness
    "use_speaker_boost": True
}
```

---

## 4. Use Cases

| Scenario | Text Template | Trigger |
|----------|---------------|---------|
| **NRI Score** | "Your neurological risk score is {score}, classified as {category} risk." | After NRI calculation |
| **Speech Results** | "Voice analysis complete. Your risk score is {score}. {recommendation}" | After speech analysis |
| **Retinal Results** | "Retinal scan analyzed. {primary_finding}. {recommendation}" | After retinal analysis |
| **Biomarker** | "Your {biomarker} is {value}, which is {status} compared to normal range." | On hover/click |
| **Recommendation** | "{recommendation}" | On request |
| **Error** | "We encountered an issue. Please try again or contact support." | On error |

---

## 5. API Specification

### Endpoint 1: Text to Speech
```
POST /api/voice/speak
Content-Type: application/json
```

### Request
```json
{
  "text": "Your neurological risk score is 23.4, classified as low risk.",
  "voice": "rachel",
  "speed": 1.0,
  "format": "mp3"
}
```

### Response
```json
{
  "success": true,
  "audio_base64": "//uQxAAAAAANIAAAAAExBT...",
  "duration_seconds": 4.2,
  "format": "mp3",
  "voice_used": "rachel",
  "characters_used": 62,
  "fallback_used": false,
  "processing_time_ms": 850
}
```

### Endpoint 2: Explain Term
```
POST /api/voice/explain
Content-Type: application/json
```

### Request
```json
{
  "term": "jitter",
  "context": "speech_analysis"
}
```

### Response
```json
{
  "success": true,
  "term": "jitter",
  "explanation": "Jitter measures the variation in your voice pitch from one vocal cord vibration to the next. Higher jitter values may indicate vocal cord instability or neurological changes.",
  "audio_base64": "//uQxAAAAAANIAAAAAExBT...",
  "duration_seconds": 8.5
}
```

---

## 6. Implementation

### ElevenLabs Integration
```python
from elevenlabs import generate, set_api_key, Voice, VoiceSettings
import base64
from gtts import gTTS
import io

class VoiceService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        if api_key:
            set_api_key(api_key)
        
        self.voices = {
            "rachel": "21m00Tcm4TlvDq8ikWAM",
            "josh": "TxGEqnHWrfWFTfGW9XjX",
            "bella": "EXAVITQu4vr4xnSDxMaL",
            "adam": "pNInz6obpgDQGcFmaJgB"
        }
    
    async def speak(
        self, 
        text: str, 
        voice: str = "rachel",
        speed: float = 1.0
    ) -> dict:
        """Generate speech from text"""
        
        if self.api_key:
            try:
                return await self._elevenlabs_speak(text, voice)
            except Exception as e:
                print(f"ElevenLabs failed: {e}, using fallback")
        
        return await self._gtts_speak(text)
    
    async def _elevenlabs_speak(self, text: str, voice: str) -> dict:
        """Generate speech using ElevenLabs"""
        
        voice_id = self.voices.get(voice, self.voices["rachel"])
        
        audio = generate(
            text=text,
            voice=Voice(
                voice_id=voice_id,
                settings=VoiceSettings(
                    stability=0.5,
                    similarity_boost=0.75,
                    style=0.5,
                    use_speaker_boost=True
                )
            ),
            model="eleven_multilingual_v2"
        )
        
        audio_b64 = base64.b64encode(audio).decode()
        
        return {
            "success": True,
            "audio_base64": audio_b64,
            "format": "mp3",
            "voice_used": voice,
            "fallback_used": False,
            "characters_used": len(text)
        }
    
    async def _gtts_speak(self, text: str) -> dict:
        """Fallback to Google TTS"""
        
        tts = gTTS(text=text, lang='en')
        mp3_buffer = io.BytesIO()
        tts.write_to_fp(mp3_buffer)
        mp3_buffer.seek(0)
        
        audio_b64 = base64.b64encode(mp3_buffer.read()).decode()
        
        return {
            "success": True,
            "audio_base64": audio_b64,
            "format": "mp3",
            "voice_used": "gtts",
            "fallback_used": True,
            "characters_used": len(text)
        }
```

### Medical Term Dictionary
```python
MEDICAL_EXPLANATIONS = {
    "jitter": "Jitter measures the variation in your voice pitch from one vocal cord vibration to the next. Higher values may indicate vocal instability.",
    
    "shimmer": "Shimmer measures the variation in your voice loudness between vocal cord vibrations. It indicates voice stability.",
    
    "hnr": "Harmonics-to-noise ratio shows how clear your voice is. Higher values mean a clearer, more resonant voice.",
    
    "nri": "The Neurological Risk Index combines results from multiple tests into a single score indicating overall neurological health.",
    
    "cup_disc_ratio": "The cup-to-disc ratio measures the size of the optic cup compared to the optic disc in your eye. Higher ratios may indicate glaucoma risk.",
    
    "rmssd": "RMSSD is a measure of heart rate variability that shows how well your autonomic nervous system regulates your heart rhythm.",
    
    "diabetic_retinopathy": "Diabetic retinopathy is damage to the blood vessels in the retina caused by high blood sugar levels.",
}

def get_explanation(term: str) -> str:
    return MEDICAL_EXPLANATIONS.get(
        term.lower().replace(" ", "_"),
        f"{term} is a biomarker used in our medical analysis."
    )
```

---

## 7. Frontend Integration

### Audio Player Component
```javascript
const VoicePlayer = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [audio, setAudio] = useState(null);
  
  const speak = async (text) => {
    const response = await fetch('/api/voice/speak', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, voice: 'rachel' })
    });
    
    const { audio_base64 } = await response.json();
    
    const audioElement = new Audio(`data:audio/mp3;base64,${audio_base64}`);
    setAudio(audioElement);
    
    audioElement.onplay = () => setIsPlaying(true);
    audioElement.onended = () => setIsPlaying(false);
    audioElement.play();
  };
  
  return (
    <button onClick={() => speak("Your results are ready")}>
      {isPlaying ? <PauseIcon /> : <PlayIcon />}
      Read Results
    </button>
  );
};

// Auto-read results when they arrive
useEffect(() => {
  if (results && voiceEnabled) {
    speak(`Your risk score is ${results.risk_score}. ${results.recommendations[0]}`);
  }
}, [results, voiceEnabled]);
```

### Global Voice Toggle
```javascript
const VoiceSettings = () => {
  const [enabled, setEnabled] = useState(false);
  const [voice, setVoice] = useState('rachel');
  const [speed, setSpeed] = useState(1.0);
  
  return (
    <div className="voice-settings">
      <Toggle checked={enabled} onChange={setEnabled}>
        Enable Voice
      </Toggle>
      
      <Select value={voice} onChange={setVoice}>
        <option value="rachel">Rachel (Female)</option>
        <option value="josh">Josh (Male)</option>
      </Select>
      
      <Slider value={speed} min={0.5} max={2} step={0.1} 
              onChange={setSpeed}>
        Speed: {speed}x
      </Slider>
    </div>
  );
};
```

---

## 8. Implementation Checklist

### Backend
- [ ] ElevenLabs API integration
- [ ] gTTS fallback
- [ ] Voice selection
- [ ] Base64 encoding
- [ ] Medical term dictionary
- [ ] Error handling

### Frontend
- [ ] Voice toggle setting
- [ ] "Read Results" button
- [ ] Audio player component
- [ ] Voice selection dropdown
- [ ] Speed adjustment
- [ ] Auto-read option
- [ ] Loading state

---

## 9. Cost Management

| Service | Free Tier | Cost |
|---------|-----------|------|
| **ElevenLabs** | 10,000 chars/month | $0.30/1K after |
| **gTTS** | Unlimited | Free |

### Strategy
- Use ElevenLabs for important results
- Cache common phrases
- Use gTTS fallback for development/demo

---

## 10. Files

```
app/pipelines/voice/
├── __init__.py
├── router.py           # FastAPI endpoints
├── service.py          # ElevenLabs + gTTS integration
```
