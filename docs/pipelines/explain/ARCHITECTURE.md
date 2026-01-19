# AI Explanation Pipeline - Architecture Design Document

## Document Metadata
| Field | Value |
|-------|-------|
| Pipeline | AI Results Explanation (LLM-powered) |
| Version | 2.0.0 |
| Last Updated | 2026-01-17 |
| LLM Provider | Cerebras Cloud (Llama 3.3 70B) |
| Voice Integration | ElevenLabs / OpenAI TTS |

---

## 1. Architecture Overview

```
[Pipeline Results] --> [Prompt Construction] --> [Cerebras LLM]
                                                      |
                                                      v
                              [SSE Streaming] --> [Frontend Panel]
                                                      |
                                                      v (optional)
                              [Voice Service] --> [Audio Playback]
```

---

## 2. System Prompts (Per Pipeline)

```python
SYSTEM_PROMPTS = {
    "base": """You are a medical AI assistant explaining diagnostic results.
- Use clear, accessible language
- Explain medical terms when used
- Be factual and evidence-based
- Do NOT diagnose - only explain results
- Keep explanations concise (2-3 paragraphs)""",

    "speech": "Explain voice biomarkers: jitter, shimmer, HNR, speech rate, voice tremor. Context: early Parkinson's/Alzheimer's indicators.",
    
    "retinal": "Explain diabetic retinopathy grades 0-4, cup-disc ratio, vessel changes. Heatmap shows AI attention areas.",
    
    "cardiology": "Explain heart rate, HRV metrics (RMSSD, SDNN), rhythm classification, ECG intervals.",
    
    "radiology": "Explain chest X-ray findings: pneumonia, cardiomegaly, effusion, etc. AI supplements radiologist.",
    
    "cognitive": "Explain memory, attention, executive function scores. Age-adjusted percentiles.",
    
    "motor": "Explain tremor frequency/amplitude, tapping metrics, fatigue index. PD vs ET patterns.",
    
    "nri": "Explain composite NRI score (0-100), modality contributions, risk categories."
}
```

---

## 3. Cerebras Integration

```python
from cerebras.cloud.sdk import Cerebras
import os

class ExplanationService:
    def __init__(self):
        self.client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        self.model = "llama-3.3-70b"
    
    async def generate_stream(self, pipeline: str, results: dict):
        """Stream explanation with SSE"""
        system = SYSTEM_PROMPTS["base"] + "\n" + SYSTEM_PROMPTS.get(pipeline, "")
        user = f"Explain these {pipeline} results:\n{format_results(results)}"
        
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.3,
            max_tokens=500,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
```

---

## 4. API Endpoints

```python
@router.post("/stream")
async def explain_stream(request: ExplainRequest):
    """SSE streaming endpoint for real-time text generation"""
    service = ExplanationService()
    
    async def generator():
        async for chunk in service.generate_stream(request.pipeline, request.results):
            yield f"data: {json.dumps({'text': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(generator(), media_type="text/event-stream")

@router.post("/")
async def explain(request: ExplainRequest):
    """Non-streaming endpoint"""
    service = ExplanationService()
    explanation = await service.generate(request.pipeline, request.results)
    return {"explanation": explanation}
```

---

## 5. Frontend Component

```typescript
export function ExplanationPanel({ pipeline, results }) {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  
  const generate = async () => {
    setLoading(true);
    setText('');
    
    const response = await fetch('/api/explain/stream', {
      method: 'POST',
      body: JSON.stringify({ pipeline, results })
    });
    
    const reader = response.body?.getReader();
    while (reader) {
      const { done, value } = await reader.read();
      if (done) break;
      // Parse SSE and append text with typing effect
      setText(prev => prev + parseChunk(value));
    }
    setLoading(false);
  };
  
  return (
    <div className="explanation-panel">
      <button onClick={generate}>{loading ? 'Generating...' : 'Explain'}</button>
      <p>{text}</p>
      {text && <VoicePlayer text={text} />}
    </div>
  );
}
```

---

## 6. Technology Stack

```txt
cerebras-cloud-sdk>=0.1.0
elevenlabs>=0.2.0          # Voice output
sse-starlette>=1.0.0       # SSE streaming
```

**Environment Variables:**
- `CEREBRAS_API_KEY`
- `ELEVENLABS_API_KEY` (optional)

---

## 7. File Structure

```
app/pipelines/explain/
├── __init__.py
├── ARCHITECTURE.md
├── router.py          # API endpoints
├── service.py         # ExplanationService
├── prompts.py         # System prompts
└── formatter.py       # Results formatting
```
