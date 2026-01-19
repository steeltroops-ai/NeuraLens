# MediLens AI Results Explanation Pipeline PRD

## Document Info
| Field | Value |
|-------|-------|
| Version | 1.0.0 |
| Priority | P1 - High (UX Enhancement) |
| Est. Dev Time | 4 hours |
| Integration | Cerebras Cloud + ElevenLabs |

---

## 1. Overview

### Purpose
Use **Cerebras Cloud Llama 3.3 70B** to generate natural language explanations of medical results for each pipeline, then optionally speak them using **ElevenLabs** or **OpenAI TTS**.

### User Experience
After each analysis:
1. Results appear in main content area
2. **AI Explanation Panel** appears on right side (desktop) or below (mobile)
3. Llama 3.3 generates comprehensive, patient-friendly explanation
4. User can click "Listen" to hear the explanation spoken

### Clinical Basis
Medical results are often confusing for patients. Natural language explanations improve:
- Patient understanding (85%+ improvement)
- Compliance with recommendations
- Reduced anxiety from unclear results
- Better patient-provider communication

---

## 2. Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Cerebras Cloud Llama 3.3 70B | Natural language generation |
| **TTS Primary** | ElevenLabs | High-quality voice output |
| **TTS Fallback** | OpenAI TTS | Alternative voice |
| **Streaming** | SSE/WebSocket | Real-time text generation |

### API Configuration

```bash
# .env
CEREBRAS_API_KEY=csk-d2ry3r6e4rf5nf9h93kj8wed2f642enwjddh644k2xm8hmwt
ELEVENLABS_API_KEY=your_elevenlabs_key
OPENAI_API_KEY=your_openai_key
```

### Installation
```bash
pip install cerebras-cloud-sdk elevenlabs openai
```

---

## 3. Cerebras Cloud Integration

### Basic Setup
```python
import os
from cerebras.cloud.sdk import Cerebras

client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY")
)

def generate_explanation(
    pipeline: str,
    results: dict,
    patient_context: dict = None
) -> str:
    """
    Generate natural language explanation of medical results
    
    Args:
        pipeline: Name of the pipeline (speech, retinal, etc.)
        results: Analysis results dictionary
        patient_context: Optional patient info (age, history)
    
    Returns:
        Natural language explanation string
    """
    
    system_prompt = get_system_prompt(pipeline)
    user_prompt = format_results_prompt(pipeline, results, patient_context)
    
    stream = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.3-70b",
        stream=True,
        max_completion_tokens=2048,
        temperature=0.2,  # Low for consistent, factual output
        top_p=1
    )
    
    explanation = ""
    for chunk in stream:
        content = chunk.choices[0].delta.content or ""
        explanation += content
        yield content  # For streaming to frontend
    
    return explanation
```

---

## 4. Pipeline-Specific System Prompts

### Speech Analysis Prompt
```python
SPEECH_SYSTEM_PROMPT = """You are a medical AI assistant explaining voice biomarker analysis results.

Your role:
1. Explain what each biomarker means in simple terms
2. Interpret the risk score and what it indicates
3. Highlight any concerning findings
4. Explain recommendations in actionable terms
5. Be reassuring but honest about any concerns

Guidelines:
- Use 8th-grade reading level
- Avoid medical jargon unless explained
- Be empathetic and supportive
- Never diagnose - only screen/suggest
- Always recommend professional consultation for concerns

Format your response with:
1. Overview (2-3 sentences)
2. Key Findings (bullet points)
3. What This Means (explanation)
4. Recommendations (actionable steps)
"""
```

### Retinal Analysis Prompt
```python
RETINAL_SYSTEM_PROMPT = """You are a medical AI assistant explaining retinal imaging analysis results.

Your role:
1. Explain what the fundus image analysis found
2. Interpret diabetic retinopathy grading if applicable
3. Explain biomarkers like cup-to-disc ratio, AV ratio
4. Describe what the heatmap highlights mean
5. Provide clear recommendations

Guidelines:
- Explain eye anatomy only when relevant
- Use analogies (e.g., "blood vessels like tiny highways")
- Be clear about urgency levels
- Emphasize importance of regular eye exams

Format your response with:
1. Summary (2-3 sentences)
2. What We Found (bullet points)
3. Risk Assessment (clear explanation)
4. Next Steps (specific recommendations)
"""
```

### Cardiology/ECG Prompt
```python
CARDIOLOGY_SYSTEM_PROMPT = """You are a medical AI assistant explaining ECG/heart analysis results.

Your role:
1. Explain heart rate and rhythm findings
2. Interpret HRV (heart rate variability) metrics
3. Explain what intervals (PR, QRS, QT) mean
4. Discuss autonomic nervous system health
5. Provide lifestyle recommendations

Guidelines:
- Heart is relatable - use simple analogies
- Explain that HRV is about nervous system balance
- Be clear about normal vs concerning findings
- Mention lifestyle factors (stress, sleep, exercise)

Format your response with:
1. Heart Health Overview (2-3 sentences)
2. Key Metrics Explained (bullet points)
3. What Your HRV Tells Us (interpretation)
4. Recommendations (actionable advice)
"""
```

### Radiology/X-Ray Prompt
```python
RADIOLOGY_SYSTEM_PROMPT = """You are a medical AI assistant explaining chest X-ray analysis results.

Your role:
1. Explain what conditions were screened for
2. Interpret probability scores for findings
3. Explain what the heatmap highlights
4. Clarify the difference between screening and diagnosis
5. Recommend appropriate follow-up

Guidelines:
- Be clear this is AI screening, not radiologist diagnosis
- Explain findings in simple terms
- Don't cause unnecessary alarm for low probabilities
- Emphasize the importance of clinical correlation

Format your response with:
1. Screening Summary (2-3 sentences)
2. Findings (bullet points with probabilities)
3. What This Means (clear interpretation)
4. Recommended Actions (specific next steps)
"""
```

### Cognitive Pipeline Prompt
```python
COGNITIVE_SYSTEM_PROMPT = """You are a medical AI assistant explaining cognitive assessment results.

Your role:
1. Explain what each cognitive domain measures
2. Interpret scores in context of age norms
3. Highlight strengths and areas of concern
4. Explain what percentiles mean
5. Suggest cognitive maintenance strategies

Guidelines:
- Be encouraging - cognition can be improved
- Explain age-adjustment clearly
- Suggest lifestyle factors (exercise, sleep, social engagement)
- Normalize some variation in performance

Format your response with:
1. Cognitive Health Summary (2-3 sentences)
2. Domain Breakdown (bullet points)
3. Strengths & Areas to Watch (balanced view)
4. Brain Health Tips (actionable recommendations)
"""
```

### Motor Assessment Prompt
```python
MOTOR_SYSTEM_PROMPT = """You are a medical AI assistant explaining motor function assessment results.

Your role:
1. Explain what the motor tests measure
2. Interpret tremor analysis findings
3. Explain tapping speed and regularity
4. Discuss coordination and fatigue patterns
5. Recommend appropriate follow-up

Guidelines:
- Don't jump to Parkinson's conclusions
- Explain that mild tremor is often normal
- Discuss factors like caffeine, fatigue, stress
- Suggest when to seek specialist evaluation

Format your response with:
1. Motor Function Summary (2-3 sentences)
2. Test Results Explained (bullet points)
3. Interpretation (what findings suggest)
4. Recommendations (when to be concerned)
"""
```

### NRI Fusion Prompt
```python
NRI_SYSTEM_PROMPT = """You are a medical AI assistant explaining the Neurological Risk Index (NRI) combined assessment.

Your role:
1. Explain what NRI combines and why
2. Interpret the overall score and category
3. Explain individual modality contributions
4. Discuss confidence levels
5. Provide comprehensive recommendations

Guidelines:
- Explain multi-modal assessment value
- Clarify which areas need attention
- Be holistic - consider the whole picture
- Emphasize this is screening, not diagnosis

Format your response with:
1. Overall Assessment (2-3 sentences)
2. Contributing Factors (ranked by contribution)
3. Areas of Strength (positive findings)
4. Areas to Monitor (concerns if any)
5. Comprehensive Recommendations (prioritized list)
"""
```

---

## 5. Results Formatting

```python
def format_results_prompt(
    pipeline: str,
    results: dict,
    patient_context: dict = None
) -> str:
    """Format results into prompt for LLM"""
    
    context = ""
    if patient_context:
        context = f"""
Patient Context:
- Age: {patient_context.get('age', 'Not provided')}
- Sex: {patient_context.get('sex', 'Not provided')}
- Medical History: {', '.join(patient_context.get('history', ['None provided']))}
"""
    
    if pipeline == "speech":
        return f"""
{context}

Speech Analysis Results:
- Risk Score: {results['risk_score']}/100 ({results['category']} risk)
- Confidence: {results['confidence']*100:.0f}%

Biomarkers:
- Jitter: {results['biomarkers']['jitter']:.4f} (normal: 0.01-0.04)
- Shimmer: {results['biomarkers']['shimmer']:.4f} (normal: 0.02-0.06)
- HNR: {results['biomarkers']['hnr']:.1f} dB (normal: 15-25)
- Speech Rate: {results['biomarkers']['speech_rate']:.1f} syllables/sec (normal: 3.5-5.5)
- Pause Ratio: {results['biomarkers']['pause_ratio']:.2f} (normal: 0.10-0.25)
- Voice Tremor: {results['biomarkers']['voice_tremor']:.2f} (normal: <0.10)
- Fluency: {results['biomarkers']['fluency_score']:.2f} (normal: >0.75)

System Recommendations:
{chr(10).join('- ' + r for r in results.get('recommendations', []))}

Please explain these results to the patient in a clear, supportive way.
"""

    elif pipeline == "retinal":
        return f"""
{context}

Retinal Analysis Results:
- Risk Score: {results['risk_score']}/100 ({results['risk_category']} risk)
- Confidence: {results['confidence']*100:.0f}%
- Diabetic Retinopathy Grade: {results.get('dr_grade', 'N/A')}

Biomarkers:
- Vessel Tortuosity: {results['biomarkers']['vessel_tortuosity']:.2f}
- AV Ratio: {results['biomarkers']['av_ratio']:.2f}
- Cup-to-Disc Ratio: {results['biomarkers']['cup_disc_ratio']:.2f}
- Vessel Density: {results['biomarkers']['vessel_density']:.2f}

Findings:
{chr(10).join('- ' + f['type'] + ' (' + f['severity'] + ')' for f in results.get('findings', []))}

Please explain these retinal findings to the patient clearly.
"""

    # Similar for other pipelines...
    
    return f"Results: {results}\n\nPlease explain these {pipeline} results clearly."
```

---

## 6. API Endpoints

### Endpoint: Generate Explanation
```
POST /api/explain
Content-Type: application/json
```

### Request
```json
{
  "pipeline": "speech",
  "results": {
    "risk_score": 28.5,
    "category": "low",
    "confidence": 0.87,
    "biomarkers": {
      "jitter": 0.025,
      "shimmer": 0.042,
      "hnr": 18.3
    },
    "recommendations": ["Continue annual monitoring"]
  },
  "patient_context": {
    "age": 65,
    "sex": "female",
    "history": ["hypertension"]
  },
  "voice_output": true,
  "voice_provider": "elevenlabs"
}
```

### Response (Streaming)
```json
{
  "success": true,
  "session_id": "exp_123456",
  "explanation_stream": "[SSE stream of text chunks]",
  "audio_base64": "...(generated after text complete)...",
  "processing_time_ms": 3200
}
```

---

## 7. Backend Implementation

```python
# app/pipelines/explain/router.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
import json
import asyncio
from cerebras.cloud.sdk import Cerebras

router = APIRouter()

# Initialize Cerebras client
cerebras_client = Cerebras(
    api_key=os.environ.get("CEREBRAS_API_KEY", 
                           "csk-d2ry3r6e4rf5nf9h93kj8wed2f642enwjddh644k2xm8hmwt")
)

class ExplanationRequest(BaseModel):
    pipeline: str
    results: Dict[str, Any]
    patient_context: Optional[Dict[str, Any]] = None
    voice_output: bool = False
    voice_provider: str = "elevenlabs"

SYSTEM_PROMPTS = {
    "speech": SPEECH_SYSTEM_PROMPT,
    "retinal": RETINAL_SYSTEM_PROMPT,
    "cardiology": CARDIOLOGY_SYSTEM_PROMPT,
    "radiology": RADIOLOGY_SYSTEM_PROMPT,
    "cognitive": COGNITIVE_SYSTEM_PROMPT,
    "motor": MOTOR_SYSTEM_PROMPT,
    "nri": NRI_SYSTEM_PROMPT,
}

@router.post("/explain")
async def explain_results(request: ExplanationRequest):
    """Generate natural language explanation of results"""
    
    system_prompt = SYSTEM_PROMPTS.get(
        request.pipeline,
        "You are a medical AI assistant explaining health assessment results."
    )
    
    user_prompt = format_results_prompt(
        request.pipeline,
        request.results,
        request.patient_context
    )
    
    async def generate():
        stream = cerebras_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b",
            stream=True,
            max_completion_tokens=2048,
            temperature=0.2,
            top_p=1
        )
        
        full_text = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content or ""
            full_text += content
            yield f"data: {json.dumps({'text': content})}\n\n"
        
        # Generate voice if requested
        if request.voice_output:
            audio_data = await generate_voice(
                full_text, 
                request.voice_provider
            )
            yield f"data: {json.dumps({'audio_base64': audio_data})}\n\n"
        
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@router.post("/explain/sync")
async def explain_results_sync(request: ExplanationRequest):
    """Non-streaming version for simple integration"""
    
    system_prompt = SYSTEM_PROMPTS.get(request.pipeline, "...")
    user_prompt = format_results_prompt(...)
    
    response = cerebras_client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model="llama-3.3-70b",
        stream=False,
        max_completion_tokens=2048,
        temperature=0.2,
        top_p=1
    )
    
    explanation = response.choices[0].message.content
    
    result = {
        "success": True,
        "explanation": explanation,
        "pipeline": request.pipeline
    }
    
    if request.voice_output:
        result["audio_base64"] = await generate_voice(
            explanation,
            request.voice_provider
        )
    
    return result
```

### Voice Generation
```python
async def generate_voice(text: str, provider: str = "elevenlabs") -> str:
    """Generate voice audio from text"""
    import base64
    
    if provider == "elevenlabs":
        from elevenlabs import generate, set_api_key
        set_api_key(os.environ.get("ELEVENLABS_API_KEY"))
        
        audio = generate(
            text=text,
            voice="Rachel",
            model="eleven_multilingual_v2"
        )
        return base64.b64encode(audio).decode()
    
    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        return base64.b64encode(response.content).decode()
    
    else:
        # Fallback to gTTS
        from gtts import gTTS
        import io
        
        tts = gTTS(text=text, lang='en')
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode()
```

---

## 8. Frontend Integration

### Explanation Panel Component
```tsx
// components/ExplanationPanel.tsx

import { useState, useEffect, useRef } from 'react';
import { Volume2, VolumeX, Loader2, Sparkles } from 'lucide-react';

interface ExplanationPanelProps {
  pipeline: string;
  results: any;
  patientContext?: any;
}

export function ExplanationPanel({ 
  pipeline, 
  results, 
  patientContext 
}: ExplanationPanelProps) {
  const [explanation, setExplanation] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioData, setAudioData] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  useEffect(() => {
    if (results) {
      generateExplanation();
    }
  }, [results]);

  const generateExplanation = async () => {
    setIsLoading(true);
    setExplanation('');

    try {
      const response = await fetch('/api/explain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pipeline,
          results,
          patient_context: patientContext,
          voice_output: true,
          voice_provider: 'elevenlabs'
        })
      });

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader!.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            
            if (data.text) {
              setExplanation(prev => prev + data.text);
            }
            if (data.audio_base64) {
              setAudioData(data.audio_base64);
            }
            if (data.done) {
              setIsLoading(false);
            }
          }
        }
      }
    } catch (error) {
      console.error('Explanation failed:', error);
      setIsLoading(false);
    }
  };

  const playAudio = () => {
    if (audioData) {
      const audio = new Audio(`data:audio/mp3;base64,${audioData}`);
      audioRef.current = audio;
      audio.onplay = () => setIsPlaying(true);
      audio.onended = () => setIsPlaying(false);
      audio.play();
    }
  };

  const stopAudio = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
    }
  };

  return (
    <div className="bg-gradient-to-br from-zinc-900 to-zinc-950 border border-zinc-800 rounded-2xl p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-purple-400" />
          <h3 className="text-lg font-semibold text-white">
            AI Explanation
          </h3>
        </div>
        
        {audioData && (
          <button
            onClick={isPlaying ? stopAudio : playAudio}
            className="flex items-center gap-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white text-sm transition-colors"
          >
            {isPlaying ? (
              <>
                <VolumeX className="w-4 h-4" />
                Stop
              </>
            ) : (
              <>
                <Volume2 className="w-4 h-4" />
                Listen
              </>
            )}
          </button>
        )}
      </div>
      
      {/* Loading State */}
      {isLoading && explanation === '' && (
        <div className="flex items-center gap-3 text-zinc-400">
          <Loader2 className="w-5 h-5 animate-spin" />
          <span>Generating explanation...</span>
        </div>
      )}
      
      {/* Streaming Text */}
      <div className="prose prose-invert prose-sm max-w-none">
        <div className="text-zinc-300 leading-relaxed whitespace-pre-wrap">
          {explanation}
          {isLoading && (
            <span className="inline-block w-2 h-4 bg-purple-400 animate-pulse ml-1" />
          )}
        </div>
      </div>
      
      {/* Powered By */}
      <div className="mt-4 pt-4 border-t border-zinc-800 flex items-center gap-2 text-xs text-zinc-500">
        <span>Powered by</span>
        <span className="text-purple-400">Cerebras Llama 3.3 70B</span>
        <span>+</span>
        <span className="text-blue-400">ElevenLabs</span>
      </div>
    </div>
  );
}
```

### Integration in Pipeline Pages
```tsx
// app/dashboard/speech/page.tsx

import { ExplanationPanel } from '@/components/ExplanationPanel';

export default function SpeechAnalysisPage() {
  const [results, setResults] = useState(null);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Main Results - 2 columns */}
      <div className="lg:col-span-2">
        <SpeechAnalysisResults results={results} />
      </div>
      
      {/* AI Explanation Panel - 1 column */}
      <div className="lg:col-span-1">
        {results && (
          <ExplanationPanel
            pipeline="speech"
            results={results}
            patientContext={{ age: 65 }}
          />
        )}
      </div>
    </div>
  );
}
```

---

## 9. Example Output

### Speech Analysis Explanation
```
## Your Voice Analysis Results

Your voice analysis shows overall healthy patterns with a low risk score of 28.5 out of 100. This is reassuring news!

### What We Found:

- **Voice Stability (Jitter & Shimmer)**: Your voice shows normal stability. Jitter at 0.025 and shimmer at 0.042 are both within healthy ranges, indicating your vocal cords are functioning well.

- **Voice Clarity (HNR)**: At 18.3 dB, your voice has good clarity and resonance. This suggests healthy vocal cord vibration.

- **Speech Patterns**: Your speaking rate of 4.2 syllables per second and pause ratio of 0.18 are normal, showing no signs of word-finding difficulties.

- **Tremor**: Minimal voice tremor detected (0.08), which is well within normal limits.

### What This Means:

Your voice biomarkers don't show any concerning patterns that might suggest early neurological changes. The slight variations we detected are completely normal and can be influenced by factors like fatigue, hydration, or stress.

### Recommendations:

1. Continue annual voice screenings as part of routine health monitoring
2. Stay hydrated - it helps maintain voice quality
3. If you notice changes in your voice (hoarseness, trembling), mention it to your doctor
4. No immediate action needed based on these results

Remember: This is a screening tool, not a diagnosis. If you have concerns, please consult with a healthcare professional.
```

---

## 10. Implementation Checklist

### Backend
- [ ] Install cerebras-cloud-sdk
- [ ] Add CEREBRAS_API_KEY to .env
- [ ] Create /api/explain endpoint
- [ ] Implement streaming response
- [ ] Create pipeline-specific prompts
- [ ] Integrate ElevenLabs for voice
- [ ] Add OpenAI TTS fallback
- [ ] Add gTTS fallback
- [ ] Error handling

### Frontend
- [ ] Create ExplanationPanel component
- [ ] Implement SSE streaming
- [ ] Add audio playback
- [ ] Loading states with typing effect
- [ ] Integrate into all pipeline pages
- [ ] Mobile responsive design
- [ ] Voice toggle setting

### Pipeline Integration
- [ ] Speech Analysis page
- [ ] Retinal Analysis page
- [ ] Cardiology page
- [ ] Radiology page
- [ ] Cognitive Assessment page
- [ ] Motor Assessment page
- [ ] NRI Fusion page

---

## 11. Files Structure

```
app/pipelines/explain/
├── __init__.py
├── router.py           # FastAPI endpoints
├── prompts.py          # System prompts per pipeline
├── formatter.py        # Results formatting
└── voice.py            # TTS integration

frontend/src/components/
├── ExplanationPanel.tsx    # Main component
└── VoiceControls.tsx       # Audio playback
```

---

## 12. Dependencies

```txt
# Backend
cerebras-cloud-sdk>=0.1.0
elevenlabs>=0.2.0
openai>=1.0.0
gtts>=2.4.0

# Frontend
# Standard React/Next.js (no additional deps)
```

---

## 13. Cost Considerations

| Service | Free Tier | Cost After |
|---------|-----------|------------|
| **Cerebras Cloud** | Check limits | Pay per token |
| **ElevenLabs** | 10K chars/mo | $0.30/1K chars |
| **OpenAI TTS** | - | $15/1M chars |
| **gTTS** | Unlimited | Free |

### Optimization
- Cache common explanations
- Use gTTS for development
- Batch similar requests
- Limit max tokens per explanation
