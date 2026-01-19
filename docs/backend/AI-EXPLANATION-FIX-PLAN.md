# AI Explanation Pipeline - Complete Fix Plan

## Full-Stack End-to-End Implementation Guide

**Document Version:** 1.0
**Created:** 2026-01-19
**Status:** Implementation Ready

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Architecture Overview](#2-architecture-overview)
3. [Pipeline-Specific Explanation Rules](#3-pipeline-specific-explanation-rules)
4. [Implementation Tasks](#4-implementation-tasks)
5. [API Route Fixes](#5-api-route-fixes)
6. [Frontend Fixes](#6-frontend-fixes)
7. [Testing Plan](#7-testing-plan)

---

## 1. Problem Analysis

### Current Issues

| Issue | Location | Impact |
|-------|----------|--------|
| Generic prompts don't use pipeline-specific rules | `explain/router.py` | Explanations lack clinical depth |
| Speech rules exist but not integrated | `speech/explanation_rules.py` | Not used by explain pipeline |
| No retinal explanation rules | `retinal/` | Generic explanations only |
| No chatbot context awareness | `chatbot/` | Chatbot doesn't follow medical rules |
| Biomarker formatting incomplete | `format_results_prompt()` | Missing new biomarkers |
| No condition-specific explanations | All pipelines | Missing clinical correlations |
| Frontend doesn't display structured results | `ExplanationPanel.tsx` | Raw markdown only |

### Data Flow Analysis

```
[Frontend UI] 
    |
    | POST /api/explain
    v
[Next.js API Route] --> frontend/src/app/api/explain/route.ts
    |
    | POST to backend
    v
[FastAPI Router] --> backend/app/pipelines/explain/router.py
    |
    | Calls format_results_prompt()
    | Uses SYSTEM_PROMPTS[pipeline]
    v
[Cerebras LLM] --> llama-3.3-70b
    |
    | Generated explanation
    v
[Response] --> Back to frontend
```

### What's Missing

1. **Pipeline-specific rule loading**: Each pipeline should have its own `explanation_rules.py`
2. **Biomarker context injection**: Full biomarker details with clinical meanings
3. **Condition risk integration**: Explain detected conditions
4. **Structured output format**: Consistent format for frontend rendering

---

## 2. Architecture Overview

### Target Architecture

```
backend/app/pipelines/
├── explain/
│   ├── router.py              # Main API router
│   ├── rule_loader.py         # NEW: Loads pipeline rules
│   └── prompt_builder.py      # NEW: Builds context-aware prompts
│
├── speech/
│   └── explanation_rules.py   # EXISTS: Speech-specific rules
│
├── retinal/
│   └── explanation_rules.py   # NEW: Retinal-specific rules
│
├── cardiology/
│   └── explanation_rules.py   # NEW: Cardiology rules
│
├── cognitive/
│   └── explanation_rules.py   # NEW: Cognitive rules
│
├── motor/
│   └── explanation_rules.py   # NEW: Motor rules
│
├── chatbot/
│   └── explanation_rules.py   # NEW: Chatbot interaction rules
│
└── nri/
    └── explanation_rules.py   # NEW: Multi-modal NRI rules
```

---

## 3. Pipeline-Specific Explanation Rules

### 3.1 Speech Pipeline Rules

**File:** `speech/explanation_rules.py` (EXISTS - needs integration)

```yaml
SPEECH_RULES:
  tone: "Professional but accessible medical communication"
  structure:
    1. Overall Summary (1-2 sentences)
    2. Key Findings (top 3 biomarkers)
    3. What This Means (clinical interpretation)
    4. Recommendations (next steps)
    5. Disclaimer (always include)
  
  biomarkers:
    jitter:
      normal: "Voice pitch stability is excellent"
      abnormal: "Voice pitch variation elevated - may indicate vocal strain"
    shimmer:
      normal: "Voice amplitude stability is healthy"
      abnormal: "Amplitude variation elevated - potential vocal fold issues"
    hnr:
      normal: "Voice clarity excellent"
      abnormal: "Increased breathiness/hoarseness detected"
    cpps:
      normal: "Voice quality score is excellent"
      abnormal: "Voice quality below optimal - possible dysphonia"
    voice_tremor:
      normal: "No significant tremor detected"
      abnormal: "Voice tremor detected - neurological evaluation may help"
  
  conditions:
    parkinsons: "Patterns sometimes seen in early Parkinson's disease"
    cognitive_decline: "Speech patterns associated with cognitive changes"
    depression: "Voice characteristics associated with mood changes"
    dysarthria: "Motor speech changes affecting articulation"
  
  mandatory_disclaimer: |
    This is for screening purposes only and is NOT a medical diagnosis.
    Always consult a qualified healthcare provider.
```

### 3.2 Retinal Pipeline Rules

**File:** `retinal/explanation_rules.py` (NEW - to be created)

```yaml
RETINAL_RULES:
  tone: "Clear, educational about eye health"
  structure:
    1. Eye Health Summary
    2. What We Analyzed (fundus image)
    3. Key Findings (biomarkers)
    4. Risk Assessment
    5. Recommended Actions
    6. Disclaimer
  
  biomarkers:
    cup_disc_ratio:
      normal_range: [0.3, 0.5]
      normal: "Optic disc appears healthy"
      abnormal: "Cup-to-disc ratio elevated - glaucoma screening recommended"
      explanation: "The optic disc is where nerve fibers exit the eye"
    
    vessel_tortuosity:
      normal_range: [0.1, 0.3]
      normal: "Blood vessels show normal patterns"
      abnormal: "Increased vessel tortuosity - may indicate vascular changes"
      explanation: "Blood vessels in the retina can show early signs of disease"
    
    av_ratio:
      normal_range: [0.6, 0.8]
      normal: "Artery-to-vein ratio normal"
      abnormal: "AV ratio abnormal - hypertensive changes possible"
    
    rnfl_thickness:
      normal: "Nerve fiber layer appears intact"
      abnormal: "RNFL thinning detected - neurological evaluation advised"
    
    hemorrhages:
      normal: "No hemorrhages detected"
      abnormal: "Retinal hemorrhages found - diabetic retinopathy screening needed"
  
  conditions:
    diabetic_retinopathy:
      grading: ["None", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]
      explanation_template: |
        Diabetic retinopathy is graded as {grade}. This means:
        {grade_explanation}
    
    glaucoma_risk:
      low: "Optic nerve appears healthy - continue annual eye exams"
      moderate: "Some signs warrant monitoring - consider specialist referral"
      high: "Features suggestive of glaucoma - urgent ophthalmology referral"
    
    macular_degeneration:
      explanation: "Signs that may warrant AMD screening detected"
  
  urgency_levels:
    routine: "Schedule regular follow-up (annual)"
    soon: "Schedule appointment within 1-3 months"
    urgent: "Seek ophthalmology evaluation within 1-2 weeks"
    emergent: "Seek immediate eye care if vision changes occur"
  
  mandatory_disclaimer: |
    This AI screening is not a substitute for a comprehensive eye exam.
    Regular eye exams by an ophthalmologist are essential for complete assessment.
```

### 3.3 Cardiology Pipeline Rules

**File:** `cardiology/explanation_rules.py` (NEW)

```yaml
CARDIOLOGY_RULES:
  tone: "Reassuring yet informative about heart health"
  structure:
    1. Heart Health Summary
    2. Heart Rate & Rhythm
    3. HRV Analysis
    4. What This Means
    5. Lifestyle Recommendations
    6. Disclaimer
  
  biomarkers:
    heart_rate:
      bradycardia: "< 60 bpm - often normal in fit individuals"
      normal: "60-100 bpm - healthy resting heart rate"
      tachycardia: "> 100 bpm - may warrant evaluation if persistent"
    
    rmssd:
      explanation: "Reflects your body's ability to adapt to stress"
      low: "Lower parasympathetic activity - consider stress management"
      normal: "Good nervous system balance"
      high: "Strong parasympathetic tone - excellent recovery capacity"
    
    sdnn:
      explanation: "Overall heart rate variability - a marker of heart health"
      low: "Reduced variability - focus on sleep and stress reduction"
      normal: "Healthy variability pattern"
      high: "Excellent variability - indicates strong heart health"
  
  lifestyle_recommendations:
    stress: "Practice relaxation techniques like deep breathing"
    sleep: "Aim for 7-9 hours of quality sleep"
    exercise: "Regular moderate exercise improves HRV"
    hydration: "Stay well hydrated throughout the day"
```

### 3.4 Cognitive Pipeline Rules

**File:** `cognitive/explanation_rules.py` (NEW)

```yaml
COGNITIVE_RULES:
  tone: "Encouraging and empowering"
  structure:
    1. Cognitive Health Summary
    2. Domain Breakdown
    3. Strengths
    4. Areas to Monitor
    5. Brain Health Tips
    6. Disclaimer
  
  domains:
    memory:
      good: "Your memory performance is strong"
      needs_attention: "Memory scores suggest room for improvement"
      tips: ["Memory games", "Association techniques", "Good sleep"]
    
    attention:
      good: "Focus and attention are well-maintained"
      needs_attention: "Attention performance could benefit from practice"
      tips: ["Mindfulness", "Reduce distractions", "Regular breaks"]
    
    processing_speed:
      good: "Quick and accurate processing"
      needs_attention: "Processing speed shows some variation"
      tips: ["Brain training games", "Physical exercise", "Adequate rest"]
    
    executive_function:
      good: "Strong planning and decision-making abilities"
      needs_attention: "Executive function may benefit from training"
      tips: ["Strategic games", "Planning exercises", "Novel challenges"]
  
  age_context: |
    Cognitive abilities naturally change with age. Your scores are compared 
    to others in your age group. Many cognitive skills can be maintained 
    or improved with regular mental and physical exercise.
```

### 3.5 Chatbot Pipeline Rules

**File:** `chatbot/explanation_rules.py` (NEW)

```yaml
CHATBOT_RULES:
  role: "MediLens AI Health Assistant"
  
  core_principles:
    - Never diagnose or prescribe
    - Always recommend professional consultation
    - Be empathetic and supportive
    - Use plain language
    - Provide educational context
    - Respect privacy
  
  response_guidelines:
    greetings:
      - "Hello! I'm your MediLens health assistant."
      - "How can I help you understand your health today?"
    
    result_questions:
      - Explain what biomarkers mean
      - Provide context for risk scores
      - Suggest appropriate next steps
      - Never minimize concerning findings
    
    medical_boundary:
      trigger_phrases: ["prescribe", "diagnose", "treat", "medication"]
      response: |
        I can provide health information and explain your screening results,
        but I cannot diagnose conditions or recommend treatments. Please 
        consult with a qualified healthcare provider for medical advice.
    
    emergency_phrases: ["chest pain", "can't breathe", "stroke", "emergency"]
    emergency_response: |
      If you're experiencing a medical emergency, please call emergency 
      services immediately (911 in the US). This AI cannot provide 
      emergency medical guidance.
  
  context_awareness:
    - Reference user's recent analysis results
    - Remember conversation context
    - Provide consistent explanations
    - Track which results have been discussed
```

### 3.6 NRI (Multi-modal) Rules

**File:** `nri/explanation_rules.py` (NEW)

```yaml
NRI_RULES:
  tone: "Comprehensive and integrative"
  structure:
    1. Overall NRI Assessment
    2. Contributing Factors
    3. Modality Breakdown
    4. Areas of Strength
    5. Areas to Monitor
    6. Integrated Recommendations
    7. Disclaimer
  
  modality_weights:
    speech: 0.25
    retinal: 0.25
    cardiology: 0.20
    cognitive: 0.15
    motor: 0.15
  
  fusion_explanation: |
    The Neurological Risk Index (NRI) combines multiple health indicators
    to provide a comprehensive assessment. Each modality contributes to
    the overall picture of your neurological health.
  
  interpretation:
    low: "Multi-modal assessment shows low neurological risk"
    moderate: "Some indicators warrant attention - individual modality details below"
    high: "Multiple indicators suggest clinical evaluation is advisable"
    critical: "Significant findings across modalities - prompt consultation recommended"
```

---

## 4. Implementation Tasks

### Phase 1: Backend Rule Files (Priority: HIGH)

| Task | File | Status |
|------|------|--------|
| Create retinal explanation rules | `retinal/explanation_rules.py` | TODO |
| Create cardiology explanation rules | `cardiology/explanation_rules.py` | TODO |
| Create cognitive explanation rules | `cognitive/explanation_rules.py` | TODO |
| Create motor explanation rules | `motor/explanation_rules.py` | TODO |
| Create chatbot explanation rules | `chatbot/explanation_rules.py` | TODO |
| Create NRI explanation rules | `nri/explanation_rules.py` | TODO |
| Update speech rules integration | `speech/explanation_rules.py` | EXISTS |

### Phase 2: Rule Loader & Prompt Builder (Priority: HIGH)

| Task | File | Description |
|------|------|-------------|
| Create rule loader | `explain/rule_loader.py` | Dynamically loads pipeline rules |
| Create prompt builder | `explain/prompt_builder.py` | Builds context-aware prompts |
| Update router | `explain/router.py` | Use new rule loader |

### Phase 3: Frontend Updates (Priority: MEDIUM)

| Task | File | Description |
|------|------|-------------|
| Update ExplanationPanel | `ExplanationPanel.tsx` | Display structured results |
| Add biomarker highlighting | `ExplanationPanel.tsx` | Highlight key findings |
| Add voice icon states | `ExplanationPanel.tsx` | Better voice UI |

### Phase 4: API Route Fixes (Priority: HIGH)

| Task | File | Description |
|------|------|-------------|
| Fix result formatting | `explain/router.py` | Include all biomarkers |
| Add condition risks | `explain/router.py` | Pass condition data |
| Add clinical context | `explain/router.py` | Include clinical notes |

---

## 5. API Route Fixes

### 5.1 Updated `format_results_prompt()` for Speech

```python
def format_results_prompt_speech(
    results: Dict[str, Any],
    patient_context: Optional[Dict[str, Any]] = None
) -> str:
    """Format speech results with full biomarker context."""
    from app.pipelines.speech.explanation_rules import (
        BIOMARKER_EXPLANATIONS,
        CONDITION_EXPLANATIONS,
        RiskLevel
    )
    
    context = format_patient_context(patient_context)
    biomarkers = results.get('biomarkers', {})
    extended = results.get('extended_biomarkers', {})
    condition_risks = results.get('condition_risks', [])
    
    # Build biomarker section with clinical context
    biomarker_text = []
    for key, config in BIOMARKER_EXPLANATIONS.items():
        if key in biomarkers:
            bio = biomarkers[key]
            value = bio.get('value', 0) if isinstance(bio, dict) else bio
            normal_min, normal_max = config.normal_range
            status = "normal" if normal_min <= value <= normal_max else "abnormal"
            biomarker_text.append(
                f"- {config.friendly_name}: {value:.2f} {config.unit} "
                f"(normal: {normal_min}-{normal_max}) [{status}]"
            )
    
    # Build condition risks section
    condition_text = []
    for cond in condition_risks:
        if cond.get('probability', 0) > 0.1:
            condition_text.append(
                f"- {cond['condition']}: {cond['probability']*100:.0f}% probability "
                f"({cond['risk_level']} risk)"
            )
    
    return f"""{context}

Speech Analysis Results:
- Overall Risk Score: {results.get('risk_score', 0)*100:.0f}/100
- Confidence: {results.get('confidence', 0)*100:.0f}%
- Quality Score: {results.get('quality_score', 0)*100:.0f}%

Biomarkers Analyzed:
{chr(10).join(biomarker_text)}

{f"Condition Risk Assessment:{chr(10)}{chr(10).join(condition_text)}" if condition_text else ""}

Clinical Notes: {results.get('clinical_notes', 'None')}

Recommendations: {results.get('recommendations', [])}

Explain these results following the speech explanation rules:
1. Start with a brief summary
2. Explain key findings in simple terms
3. Provide clinical interpretation
4. Give actionable recommendations
5. Include the mandatory disclaimer
"""
```

### 5.2 Updated System Prompts

```python
SYSTEM_PROMPTS = {
    "speech": """You are a medical AI assistant explaining voice biomarker analysis results.

RULES YOU MUST FOLLOW:
1. NEVER diagnose - only screen and suggest
2. Use plain language at 8th-grade reading level
3. Be empathetic and supportive
4. Explain each abnormal biomarker's clinical significance
5. Reference condition probabilities if elevated (>15%)
6. Always emphasize this is a screening tool
7. End with recommendations and disclaimer

BIOMARKER INTERPRETATIONS:
- Jitter: Pitch stability. Elevated = vocal strain or neurological issue
- Shimmer: Amplitude stability. Elevated = vocal fold problems
- HNR: Voice clarity. Low = breathiness or hoarseness
- CPPS: Voice quality gold standard. Low = dysphonia
- Speech Rate: Speaking speed. Slow = motor/cognitive issues
- Voice Tremor: Rhythmic oscillations. Elevated = neurological concern
- Pause Ratio: Silence proportion. High = word-finding difficulty

FORMAT:
## Your Voice Analysis Summary
[1-2 sentence overview]

### Key Findings
[Top 3 notable biomarkers with explanations]

### What This Means
[Clinical interpretation in plain language]

### Recommendations
[Actionable next steps]

### Important Note
[Mandatory screening disclaimer]""",

    "retinal": """You are a medical AI assistant explaining retinal imaging analysis.

RULES YOU MUST FOLLOW:
1. Explain eye anatomy in simple terms
2. Clarify this is AI screening, not diagnosis
3. Emphasize importance of regular eye exams
4. Be clear about urgency levels
5. Never minimize concerning findings

BIOMARKER INTERPRETATIONS:
- Cup-to-Disc Ratio: The optic disc center vs outer ring. High = glaucoma risk
- AV Ratio: Artery vs vein width. Abnormal = hypertensive changes
- Vessel Tortuosity: Blood vessel straightness. High = vascular issues
- RNFL Thickness: Nerve fiber layer. Thin = neurological concern
- Hemorrhages: Bleeding spots. Present = diabetic retinopathy indicator

URGENCY LEVELS:
- Routine: Annual follow-up
- Soon: 1-3 months
- Urgent: 1-2 weeks
- Emergent: Immediate if vision changes

FORMAT:
## Your Eye Health Summary
[Overview of retinal analysis]

### What We Analyzed
[Explain fundus imaging briefly]

### Key Findings
[Biomarkers with clinical context]

### Risk Assessment
[Overall assessment with urgency]

### Recommended Actions
[Specific next steps]

### Important Note
[Eye exam recommendation + disclaimer]""",

    # ... (other pipelines similarly detailed)
}
```

---

## 6. Frontend Fixes

### 6.1 Enhanced ExplanationPanel

```typescript
// Add structured result display
interface StructuredExplanation {
  summary: string;
  keyFindings: Array<{
    biomarker: string;
    value: string;
    status: 'normal' | 'warning' | 'abnormal';
    explanation: string;
  }>;
  interpretation: string;
  recommendations: string[];
  disclaimer: string;
}

// Parse markdown into structured sections
function parseExplanation(markdown: string): StructuredExplanation {
  const sections = markdown.split('###');
  // ... parsing logic
}

// Render with visual indicators
function ExplanationSection({ finding }: { finding: KeyFinding }) {
  return (
    <div className={cn(
      "p-3 rounded-lg border",
      finding.status === 'abnormal' && "border-red-200 bg-red-50",
      finding.status === 'warning' && "border-amber-200 bg-amber-50",
      finding.status === 'normal' && "border-green-200 bg-green-50"
    )}>
      <div className="font-medium">{finding.biomarker}</div>
      <div className="text-sm">{finding.value}</div>
      <div className="text-xs text-gray-600">{finding.explanation}</div>
    </div>
  );
}
```

### 6.2 Pipeline-Specific Styling

```typescript
const PIPELINE_THEMES: Record<string, PipelineTheme> = {
  speech: {
    icon: Mic,
    color: 'blue',
    gradient: 'from-blue-500 to-indigo-600'
  },
  retinal: {
    icon: Eye,
    color: 'emerald',
    gradient: 'from-emerald-500 to-teal-600'
  },
  cardiology: {
    icon: Heart,
    color: 'red',
    gradient: 'from-red-500 to-rose-600'
  },
  cognitive: {
    icon: Brain,
    color: 'purple',
    gradient: 'from-purple-500 to-violet-600'
  },
  motor: {
    icon: Hand,
    color: 'orange',
    gradient: 'from-orange-500 to-amber-600'
  },
  nri: {
    icon: Activity,
    color: 'cyan',
    gradient: 'from-cyan-500 to-blue-600'
  }
};
```

---

## 7. Testing Plan

### 7.1 Backend Tests

```python
# test_explanation_rules.py

def test_speech_explanation_generation():
    """Test speech explanation with all biomarkers."""
    from app.pipelines.speech.explanation_rules import generate_speech_explanation
    
    mock_results = {
        "risk_score": 0.35,
        "confidence": 0.87,
        "biomarkers": {
            "jitter": {"value": 1.5, "normal_range": [0, 1.04]},
            "shimmer": {"value": 2.0, "normal_range": [0, 3.81]},
            # ... more biomarkers
        },
        "condition_risks": [
            {"condition": "parkinsons", "probability": 0.25}
        ]
    }
    
    explanation = generate_speech_explanation(mock_results)
    
    assert "Voice" in explanation
    assert "jitter" in explanation.lower() or "pitch" in explanation.lower()
    assert "disclaimer" in explanation.lower() or "screening" in explanation.lower()

def test_all_pipelines_have_rules():
    """Verify all pipelines have explanation rules."""
    pipelines = ['speech', 'retinal', 'cardiology', 'cognitive', 'motor', 'nri']
    
    for pipeline in pipelines:
        module = importlib.import_module(f'app.pipelines.{pipeline}.explanation_rules')
        assert hasattr(module, 'generate_explanation') or hasattr(module, 'EXPLANATION_RULES')
```

### 7.2 Frontend Tests

```typescript
// ExplanationPanel.test.tsx

describe('ExplanationPanel', () => {
  it('renders explanation for speech pipeline', async () => {
    const mockResults = {
      risk_score: 0.35,
      biomarkers: { jitter: { value: 1.2 } }
    };
    
    render(<ExplanationPanel pipeline="speech" results={mockResults} />);
    
    await waitFor(() => {
      expect(screen.getByText(/AI Explanation/)).toBeInTheDocument();
    });
  });
  
  it('shows pipeline-specific theming', () => {
    // Test each pipeline has correct icon/colors
  });
});
```

### 7.3 E2E Tests

```typescript
// e2e/explanation.spec.ts

test('speech analysis generates explanation', async ({ page }) => {
  // Upload audio
  await page.goto('/dashboard/speech');
  await page.setInputFiles('input[type="file"]', 'test.wav');
  
  // Wait for analysis
  await page.waitForSelector('[data-testid="analysis-complete"]');
  
  // Check explanation generated
  await expect(page.locator('.explanation-panel')).toContainText('Voice');
  await expect(page.locator('.explanation-panel')).toContainText('Recommendations');
});
```

---

## 8. File Creation Checklist

### Backend Files Created (DONE)

- [x] `backend/app/pipelines/speech/explanation_rules.py` - EXISTS
- [x] `backend/app/pipelines/retinal/explanation_rules.py` - CREATED
- [x] `backend/app/pipelines/cardiology/explanation_rules.py` - CREATED
- [x] `backend/app/pipelines/cognitive/explanation_rules.py` - CREATED
- [x] `backend/app/pipelines/motor/explanation_rules.py` - CREATED
- [x] `backend/app/pipelines/chatbot/explanation_rules.py` - CREATED
- [x] `backend/app/pipelines/nri/explanation_rules.py` - CREATED
- [x] `backend/app/pipelines/explain/rule_loader.py` - CREATED
- [x] `backend/app/pipelines/explain/prompt_builder.py` - CREATED

### Backend Files Updated (DONE)

- [x] `backend/app/pipelines/explain/router.py` - Updated to use prompt builder
- [x] `backend/app/schemas/assessment.py` - Added extended biomarkers & condition risks

### Frontend Files To Update (TODO)

- [ ] `frontend/src/components/explanation/ExplanationPanel.tsx` - Display structured results
- [ ] `frontend/src/app/api/explain/route.ts` - Already working, minor enhancements possible
- [ ] `frontend/src/types/explanation.ts` - Create if needed for typed explanations


---

## 9. Quick Start Implementation

### Step 1: Create Retinal Rules (Example)

```bash
# Run this to create the retinal explanation rules
touch backend/app/pipelines/retinal/explanation_rules.py
```

### Step 2: Update Router

Modify `explain/router.py` to load rules dynamically:

```python
from importlib import import_module

def get_pipeline_rules(pipeline: str) -> dict:
    try:
        module = import_module(f'app.pipelines.{pipeline}.explanation_rules')
        return getattr(module, 'EXPLANATION_RULES', {})
    except ImportError:
        return {}
```

### Step 3: Test Integration

```bash
cd backend
python -c "from app.pipelines.explain.router import get_pipeline_rules; print(get_pipeline_rules('speech'))"
```

---

*Document Status: Ready for Implementation*
*Estimated Effort: 8-12 hours*
*Priority: HIGH*
