"""
MediLens AI Explanation Router
Uses Cerebras Cloud Llama 3.3 70B to generate natural language explanations
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
import json
import base64
import asyncio

router = APIRouter()

# Cerebras API configuration
CEREBRAS_API_KEY = os.environ.get(
    "CEREBRAS_API_KEY", 
    "csk-d2ry3r6e4rf5nf9h93kj8wed2f642enwjddh644k2xm8hmwt"
)

# Try to import Cerebras SDK
try:
    from cerebras.cloud.sdk import Cerebras
    cerebras_client = Cerebras(api_key=CEREBRAS_API_KEY)
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False
    cerebras_client = None


class ExplanationRequest(BaseModel):
    pipeline: str
    results: Dict[str, Any]
    patient_context: Optional[Dict[str, Any]] = None
    voice_output: bool = False
    voice_provider: str = "elevenlabs"


class ExplanationResponse(BaseModel):
    success: bool
    explanation: str
    pipeline: str
    audio_base64: Optional[str] = None
    processing_time_ms: int


# System prompts for each pipeline
SYSTEM_PROMPTS = {
    "speech": """You are a medical AI assistant explaining voice biomarker analysis results.

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

Format with: Overview, Key Findings, What This Means, Recommendations.""",

    "retinal": """You are a medical AI assistant explaining retinal imaging analysis results.

Your role:
1. Explain what the fundus image analysis found
2. Interpret diabetic retinopathy grading if applicable
3. Explain biomarkers like cup-to-disc ratio, AV ratio
4. Describe what findings mean
5. Provide clear recommendations

Guidelines:
- Use simple analogies for eye anatomy
- Be clear about urgency levels
- Emphasize importance of regular eye exams

Format with: Summary, What We Found, Risk Assessment, Next Steps.""",

    "cardiology": """You are a medical AI assistant explaining ECG/heart analysis results.

Your role:
1. Explain heart rate and rhythm findings
2. Interpret HRV (heart rate variability) metrics
3. Discuss autonomic nervous system health
4. Provide lifestyle recommendations

Guidelines:
- Use simple analogies for heart function
- Explain that HRV reflects nervous system balance
- Be clear about normal vs concerning findings

Format with: Heart Health Overview, Key Metrics, HRV Interpretation, Recommendations.""",

    "radiology": """You are a medical AI assistant explaining chest X-ray analysis results.

Your role:
1. Explain what conditions were screened for
2. Interpret probability scores for findings
3. Clarify screening vs diagnosis difference
4. Recommend appropriate follow-up

Guidelines:
- Be clear this is AI screening, not radiologist diagnosis
- Don't cause unnecessary alarm for low probabilities
- Emphasize clinical correlation importance

Format with: Screening Summary, Findings, What This Means, Recommended Actions.""",

    "cognitive": """You are a medical AI assistant explaining cognitive assessment results.

Your role:
1. Explain what each cognitive domain measures
2. Interpret scores in context of age norms
3. Highlight strengths and areas of concern
4. Suggest cognitive maintenance strategies

Guidelines:
- Be encouraging - cognition can be improved
- Explain age-adjustment clearly
- Suggest lifestyle factors

Format with: Cognitive Health Summary, Domain Breakdown, Strengths & Areas to Watch, Brain Health Tips.""",

    "motor": """You are a medical AI assistant explaining motor function assessment results.

Your role:
1. Explain what the motor tests measure
2. Interpret tremor analysis findings
3. Discuss coordination and fatigue patterns
4. Recommend appropriate follow-up

Guidelines:
- Don't jump to conclusions
- Explain that mild tremor is often normal
- Discuss factors like caffeine, fatigue, stress

Format with: Motor Function Summary, Test Results, Interpretation, Recommendations.""",

    "nri": """You are a medical AI assistant explaining the Neurological Risk Index (NRI) combined assessment.

Your role:
1. Explain what NRI combines and why
2. Interpret the overall score and category
3. Explain individual modality contributions
4. Provide comprehensive recommendations

Guidelines:
- Explain multi-modal assessment value
- Clarify which areas need attention
- Emphasize this is screening, not diagnosis

Format with: Overall Assessment, Contributing Factors, Areas of Strength, Areas to Monitor, Recommendations.""",
}


def format_results_prompt(
    pipeline: str,
    results: Dict[str, Any],
    patient_context: Optional[Dict[str, Any]] = None
) -> str:
    """Format results into prompt for LLM using pipeline-specific builders."""
    
    # Try to use the new prompt builder
    try:
        from .prompt_builder import PromptBuilder
        builder = PromptBuilder(pipeline)
        return builder.build_user_prompt(results, patient_context)
    except ImportError:
        pass  # Fall back to legacy formatting
    
    # Legacy formatting (kept for compatibility)
    context = ""
    if patient_context:
        context = f"""
Patient Context:
- Age: {patient_context.get('age', 'Not provided')}
- Sex: {patient_context.get('sex', 'Not provided')}
"""
    
    # Format based on pipeline
    if pipeline == "speech":
        biomarkers = results.get('biomarkers', {})
        condition_risks = results.get('condition_risks', [])
        extended = results.get('extended_biomarkers', {})
        
        # Build biomarker section with full details
        bio_lines = []
        for key, bio in biomarkers.items():
            if isinstance(bio, dict):
                value = bio.get('value', 0)
                unit = bio.get('unit', '')
                normal = bio.get('normal_range', [0, 1])
                status = "NORMAL" if normal[0] <= value <= normal[1] else "ABNORMAL"
                bio_lines.append(f"- {key}: {value:.2f} {unit} (range: {normal[0]}-{normal[1]}) [{status}]")
            else:
                bio_lines.append(f"- {key}: {bio}")
        
        # Build condition risks
        cond_lines = []
        for cond in condition_risks:
            if isinstance(cond, dict) and cond.get('probability', 0) > 0.1:
                cond_lines.append(
                    f"- {cond['condition']}: {cond['probability']*100:.0f}% probability "
                    f"({cond.get('risk_level', 'unknown')} risk)"
                )
        
        return f"""{context}

Speech Analysis Results:
- Overall Risk Score: {results.get('risk_score', 0)*100:.0f}/100
- Confidence: {results.get('confidence', 0)*100:.0f}%
- Quality Score: {results.get('quality_score', 0)*100:.0f}%

Biomarkers Analyzed:
{chr(10).join(bio_lines)}

{f"Condition Risk Assessment:{chr(10)}{chr(10).join(cond_lines)}" if cond_lines else ""}

Clinical Notes: {results.get('clinical_notes', 'None')}
Recommendations: {results.get('recommendations', [])}

Explain these results clearly and supportively."""

    elif pipeline == "retinal":
        biomarkers = results.get('biomarkers', {})
        return f"""{context}

Retinal Analysis Results:
- Risk Score: {results.get('risk_score', 'N/A')}/100
- Risk Category: {results.get('risk_category', 'N/A')}

Biomarkers:
- Vessel Tortuosity: {biomarkers.get('vessel_tortuosity', 'N/A')}
- AV Ratio: {biomarkers.get('av_ratio', 'N/A')}
- Cup-to-Disc Ratio: {biomarkers.get('cup_disc_ratio', 'N/A')}
- RNFL Thickness: {biomarkers.get('rnfl_thickness', 'N/A')}

Findings: {results.get('findings', [])}
DR Grade: {results.get('dr_grade', 'None')}

Explain these retinal findings clearly."""

    elif pipeline == "cardiology":
        return f"""{context}

Cardiology/ECG Results:
- Heart Rate: {results.get('heart_rate', 'N/A')} bpm
- Rhythm: {results.get('rhythm', 'N/A')}
- RMSSD: {results.get('rmssd_ms', 'N/A')} ms
- SDNN: {results.get('sdnn_ms', 'N/A')} ms

Explain these heart analysis results."""

    elif pipeline == "nri":
        return f"""{context}

NRI Fusion Results:
- NRI Score: {results.get('nri_score', 'N/A')}/100
- Risk Category: {results.get('risk_category', 'N/A')}
- Modalities: {results.get('modalities', [])}
- Contributions: {results.get('modality_contributions', [])}

Explain this multi-modal assessment."""

    else:
        return f"{context}\n\n{pipeline.title()} Results:\n{json.dumps(results, indent=2)}\n\nExplain these results."


def get_system_prompt(pipeline: str) -> str:
    """Get system prompt for a pipeline, using new prompt builder if available."""
    try:
        from .prompt_builder import PromptBuilder
        builder = PromptBuilder(pipeline)
        return builder.build_system_prompt()
    except ImportError:
        return SYSTEM_PROMPTS.get(
            pipeline,
            "You are a medical AI assistant explaining health assessment results clearly and supportively."
        )



async def generate_voice(text: str, provider: str = "polly") -> Optional[str]:
    """Generate voice audio from text using the unified Voice Service"""
    # Clean text for TTS (remove markdown)
    clean_text = text.replace('**', '').replace('##', '').replace('###', '').replace('*', '')
    
    # Use the unified service wrapper which handles Polly and caching
    try:
        from app.pipelines.voice.service import speak_llm_explanation
        return await speak_llm_explanation(clean_text)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Unified voice service failed: {e}")
        return None




@router.post("")
async def explain_results_streaming(request: ExplanationRequest):
    """Generate streaming natural language explanation"""
    
    if not CEREBRAS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Cerebras SDK not available. Install with: pip install cerebras-cloud-sdk"
        )
    
    # Use enhanced prompt builder
    system_prompt = get_system_prompt(request.pipeline)
    
    user_prompt = format_results_prompt(
        request.pipeline,
        request.results,
        request.patient_context
    )

    
    async def generate():
        try:
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
                audio_data = await generate_voice(full_text, request.voice_provider)
                if audio_data:
                    yield f"data: {json.dumps({'audio_base64': audio_data})}\n\n"
            
            yield f"data: {json.dumps({'done': True, 'total_length': len(full_text)})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@router.post("/sync", response_model=ExplanationResponse)
async def explain_results_sync(request: ExplanationRequest):
    """Non-streaming explanation endpoint"""
    import time
    start = time.time()
    
    if not CEREBRAS_AVAILABLE:
        # Fallback to mock explanation
        explanation = generate_mock_explanation(request.pipeline, request.results)
    else:
        # Use enhanced prompt builder
        system_prompt = get_system_prompt(request.pipeline)
        user_prompt = format_results_prompt(
            request.pipeline,
            request.results,
            request.patient_context
        )

        
        try:
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
        except Exception as e:
            explanation = generate_mock_explanation(request.pipeline, request.results)
    
    audio_data = None
    if request.voice_output:
        audio_data = await generate_voice(explanation, request.voice_provider)
    
    return ExplanationResponse(
        success=True,
        explanation=explanation,
        pipeline=request.pipeline,
        audio_base64=audio_data,
        processing_time_ms=int((time.time() - start) * 1000)
    )


def generate_mock_explanation(pipeline: str, results: Dict[str, Any]) -> str:
    """Generate a mock explanation when LLM is not available"""
    
    risk_score = results.get('risk_score', 25)
    
    if risk_score < 30:
        risk_text = "low risk"
        recommendation = "Continue with routine monitoring."
    elif risk_score < 60:
        risk_text = "moderate risk"
        recommendation = "Consider scheduling a follow-up assessment."
    else:
        risk_text = "elevated risk"
        recommendation = "We recommend consulting with a healthcare professional."
    
    return f"""## Your {pipeline.title()} Analysis Results

Your analysis shows a **{risk_text}** score of {risk_score}/100.

### What This Means

Based on the biomarkers we analyzed, your results are within {"acceptable" if risk_score < 50 else "concerning"} ranges. 

### Key Findings

The analysis examined multiple biomarkers to assess your {pipeline} health:
- Overall risk score: {risk_score}/100
- Confidence level: {results.get('confidence', 0.85)*100:.0f}%

### Recommendations

{recommendation}

*Note: This is an AI-powered screening tool and should not replace professional medical advice.*
"""


@router.get("/health")
async def health():
    return {
        "status": "ok",
        "module": "explain",
        "cerebras_available": CEREBRAS_AVAILABLE,
        "model": "llama-3.3-70b" if CEREBRAS_AVAILABLE else "mock"
    }

