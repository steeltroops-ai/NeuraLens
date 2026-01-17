# NRI Fusion Pipeline - Product Requirements Document

## Agent Assignment: NRI-AGENT-03
## Branch: `feature/nri-fusion-fix`
## Priority: P0 (Core USP - Unified Risk Score)

---

## Overview

The NRI (Neurological Risk Index) Fusion Pipeline is the **most unique selling point** of NeuraLens. It combines results from all assessment modalities (Speech, Retinal, Motor, Cognitive) into a single unified risk score. This is what differentiates us from point solutions.

**Why This Matters for Judges**:
- Shows sophisticated multi-modal AI
- Demonstrates Bayesian fusion techniques
- Provides clear, actionable output (0-100 score)
- More clinically relevant than individual tests

---

## Current Architecture

### Backend Files

```
backend/app/pipelines/nri/
  |-- __init__.py     (31 bytes)
  |-- analyzer.py     (13,015 bytes) - Fusion algorithm
  |-- router.py       (9,609 bytes)  - FastAPI routes
```

### Frontend Files

```
frontend/src/app/dashboard/nri-fusion/
  |-- page.tsx            - Main NRI page
  |-- _components/        - Fusion-specific components

frontend/src/lib/ml/
  |-- nri-fusion.ts       (16,720 bytes) - Frontend fusion logic
  |-- risk-assessment.ts  (17,444 bytes) - Risk calculation
```

---

## Requirements

### Functional Requirements

| ID | Requirement | Priority | Status |
|----|-------------|----------|--------|
| NRI-F01 | Accept modality scores from 1-4 pipelines | P0 | Needs improvement |
| NRI-F02 | Calculate weighted fusion score | P0 | Existing |
| NRI-F03 | Handle missing modalities gracefully | P0 | Needs implementation |
| NRI-F04 | Apply evidence-based modality weights | P0 | Needs calibration |
| NRI-F05 | Calculate confidence intervals | P1 | Needs implementation |
| NRI-F06 | Provide risk category (Low/Moderate/High/Very High) | P0 | Existing |
| NRI-F07 | Show modality contribution breakdown | P1 | Needs improvement |
| NRI-F08 | Generate clinical recommendations | P1 | Existing |
| NRI-F09 | Support Bayesian fusion method | P2 | Partial |

### Non-Functional Requirements

| ID | Requirement | Priority | Target |
|----|-------------|----------|--------|
| NRI-NF01 | Fusion calculation < 100ms | P0 | 50ms |
| NRI-NF02 | Intuitive gauge visualization | P1 | Circular gauge |
| NRI-NF03 | Export combined report | P2 | PDF with all modalities |

---

## Agent Task Breakdown

### Step 1: Fix Modality Weight Calibration (2 hours)

**File**: `backend/app/pipelines/nri/analyzer.py`

**Tasks**:
1. Define evidence-based weights for each modality
2. Implement dynamic weight adjustment based on data quality
3. Support configurable weight profiles
4. Document weight rationale in comments

**Weight Configuration**:
```python
# Evidence-based modality weights
# References: 
# - Speech: JAMA Neurology 2020, voice biomarkers in Parkinson's
# - Retinal: Alzheimer's & Dementia 2021, retinal imaging review
# - Motor: Movement Disorders 2019, finger tapping meta-analysis
# - Cognitive: Neurology 2022, cognitive screening tools comparison

MODALITY_WEIGHTS = {
    "speech": {
        "base_weight": 0.25,
        "reliability": 0.92,
        "evidence_level": "high"
    },
    "retinal": {
        "base_weight": 0.25,
        "reliability": 0.89,
        "evidence_level": "moderate-high"
    },
    "motor": {
        "base_weight": 0.30,
        "reliability": 0.88,
        "evidence_level": "high"
    },
    "cognitive": {
        "base_weight": 0.20,
        "reliability": 0.91,
        "evidence_level": "high"
    }
}

def get_adjusted_weights(available_modalities: list[str]) -> dict[str, float]:
    """Adjust weights when some modalities are missing"""
    total_weight = sum(
        MODALITY_WEIGHTS[m]["base_weight"] 
        for m in available_modalities
    )
    
    return {
        m: MODALITY_WEIGHTS[m]["base_weight"] / total_weight
        for m in available_modalities
    }
```

### Step 2: Handle Missing Modalities (1.5 hours)

**File**: `backend/app/pipelines/nri/analyzer.py`

**Tasks**:
1. Accept partial modality data (1-4 modalities)
2. Recalculate weights for available modalities
3. Apply confidence penalty for missing data
4. Return which modalities were used in calculation

**Pattern**:
```python
def calculate_nri_score(modality_results: dict) -> dict:
    """Calculate NRI with handling for missing modalities"""
    
    available = [m for m in modality_results if modality_results[m] is not None]
    
    if len(available) == 0:
        raise ValueError("At least one modality result required")
    
    # Adjust weights for available modalities
    weights = get_adjusted_weights(available)
    
    # Calculate weighted score
    weighted_sum = sum(
        modality_results[m]["risk_score"] * weights[m]
        for m in available
    )
    
    # Confidence penalty for missing modalities
    # Full confidence at 4 modalities, -5% per missing modality
    missing_count = 4 - len(available)
    confidence_penalty = missing_count * 0.05
    
    # Combine modality confidences
    avg_confidence = sum(
        modality_results[m]["confidence"]
        for m in available
    ) / len(available)
    
    final_confidence = max(0.5, avg_confidence - confidence_penalty)
    
    return {
        "nri_score": round(weighted_sum, 1),
        "confidence": round(final_confidence, 2),
        "modalities_used": available,
        "modalities_missing": [m for m in ["speech", "retinal", "motor", "cognitive"] if m not in available],
        "weights_applied": weights
    }
```

### Step 3: Add Confidence Intervals (1 hour)

**File**: `backend/app/pipelines/nri/analyzer.py`

**Tasks**:
1. Calculate 95% confidence interval for NRI score
2. Account for individual modality uncertainties
3. Display as range (e.g., "28.5 [24.2 - 32.8]")

**Pattern**:
```python
def calculate_confidence_interval(nri_score: float, modality_results: dict) -> tuple:
    """Calculate 95% CI based on modality uncertainties"""
    
    # Collect uncertainties
    uncertainties = [
        modality_results[m].get("uncertainty", 0.1)
        for m in modality_results
        if modality_results[m] is not None
    ]
    
    # Propagate uncertainty (simplified)
    combined_uncertainty = (sum(u**2 for u in uncertainties) ** 0.5) / len(uncertainties)
    
    # 95% CI = +/- 1.96 * uncertainty
    margin = 1.96 * combined_uncertainty * 100  # Scale to 0-100
    
    lower = max(0, nri_score - margin)
    upper = min(100, nri_score + margin)
    
    return (round(lower, 1), round(upper, 1))
```

### Step 4: Fix Risk Categorization (1 hour)

**File**: `backend/app/pipelines/nri/analyzer.py`

**Tasks**:
1. Define clear threshold boundaries
2. Add category descriptions
3. Include clinical recommendations per category

**Risk Categories**:
```python
RISK_CATEGORIES = {
    "low": {
        "range": (0, 25),
        "label": "Low Risk",
        "color": "#10B981",  # Green
        "description": "Minimal neurological risk indicators detected.",
        "recommendations": [
            "Continue regular health monitoring",
            "Maintain healthy lifestyle habits",
            "Annual follow-up assessment recommended"
        ]
    },
    "moderate": {
        "range": (25, 50),
        "label": "Moderate Risk", 
        "color": "#F59E0B",  # Amber
        "description": "Some risk factors present. Monitoring recommended.",
        "recommendations": [
            "Consult with primary care physician",
            "Consider specialist referral for detailed evaluation",
            "Repeat assessment in 6 months"
        ]
    },
    "high": {
        "range": (50, 75),
        "label": "High Risk",
        "color": "#EF4444",  # Red
        "description": "Elevated risk indicators. Clinical evaluation recommended.",
        "recommendations": [
            "Schedule appointment with neurologist",
            "Consider comprehensive neurological workup",
            "Discuss with healthcare provider promptly"
        ]
    },
    "very_high": {
        "range": (75, 100),
        "label": "Very High Risk",
        "color": "#7F1D1D",  # Dark red
        "description": "Significant risk indicators. Urgent evaluation needed.",
        "recommendations": [
            "Seek urgent neurological consultation",
            "Consider immediate clinical evaluation",
            "Discuss results with healthcare provider today"
        ]
    }
}
```

### Step 5: Fix Frontend Visualization (2 hours)

**File**: `frontend/src/app/dashboard/nri-fusion/page.tsx`

**Tasks**:
1. Add circular gauge for NRI score
2. Show modality contribution stacked bar
3. Display confidence interval
4. Animate score reveal

**Gauge Component**:
```tsx
interface NRIGaugeProps {
  score: number;
  category: string;
  confidence: number;
  confidenceInterval: [number, number];
}

function NRIGauge({ score, category, confidence, confidenceInterval }: NRIGaugeProps) {
  return (
    <div className="relative w-64 h-64">
      {/* Circular gauge SVG */}
      <svg viewBox="0 0 100 100" className="w-full h-full">
        {/* Background arc */}
        <circle
          cx="50" cy="50" r="45"
          fill="none" stroke="#e5e7eb" strokeWidth="8"
          strokeDasharray="212" strokeDashoffset="53"
          transform="rotate(135 50 50)"
        />
        {/* Filled arc based on score */}
        <circle
          cx="50" cy="50" r="45"
          fill="none" stroke={getCategoryColor(category)} strokeWidth="8"
          strokeDasharray={`${score * 2.12} 212`}
          strokeDashoffset="53"
          transform="rotate(135 50 50)"
          className="transition-all duration-1000"
        />
      </svg>
      {/* Center score */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-bold">{score.toFixed(1)}</span>
        <span className="text-sm text-gray-500">
          [{confidenceInterval[0]} - {confidenceInterval[1]}]
        </span>
        <span className="text-lg font-medium" style={{ color: getCategoryColor(category) }}>
          {category.replace('_', ' ').toUpperCase()}
        </span>
      </div>
    </div>
  );
}
```

---

## API Contract

### POST /api/v1/nri/calculate

**Request**:
```json
{
  "session_id": "uuid",
  "modality_results": {
    "speech": {
      "risk_score": 25.0,
      "confidence": 0.92,
      "biomarkers": { "jitter": 0.8, "shimmer": 4.2 }
    },
    "retinal": {
      "risk_score": 32.5,
      "confidence": 0.89,
      "biomarkers": { "cup_disc_ratio": 0.42 }
    },
    "motor": null,
    "cognitive": {
      "risk_score": 22.0,
      "confidence": 0.91,
      "biomarkers": { "memory_score": 0.85 }
    }
  }
}
```

**Success Response** (200):
```json
{
  "success": true,
  "data": {
    "nri_score": 26.8,
    "risk_category": "moderate",
    "confidence": 0.87,
    "confidence_interval": [22.5, 31.2],
    "modalities_used": ["speech", "retinal", "cognitive"],
    "modalities_missing": ["motor"],
    "modality_contributions": {
      "speech": 8.3,
      "retinal": 10.8,
      "cognitive": 7.7
    },
    "weights_applied": {
      "speech": 0.33,
      "retinal": 0.33,
      "cognitive": 0.34
    },
    "interpretation": "Moderate neurological risk. Some indicators present across speech and retinal modalities.",
    "recommendations": [
      "Consider follow-up with neurologist",
      "Complete motor assessment for comprehensive evaluation",
      "Repeat assessment in 6 months"
    ],
    "processing_time_ms": 45
  }
}
```

---

## Fusion Algorithm Explanation

### Weighted Linear Fusion (Current)

```
NRI = Σ (weight_i × score_i) for all available modalities
```

### Bayesian Fusion (Enhanced - P2)

```
P(disease | all_modalities) = 
    P(d | speech) × P(d | retinal) × P(d | motor) × P(d | cognitive) × P(d)
    / P(all_evidence)
```

For hackathon, **weighted linear fusion is sufficient**. Bayesian can be mentioned as future work.

---

## Test Cases

### Backend Unit Tests

```python
# tests/test_nri_fusion.py

def test_nri_all_modalities():
    """Should calculate NRI with all 4 modalities"""
    pass

def test_nri_partial_modalities():
    """Should calculate NRI with 2-3 modalities"""
    pass

def test_nri_single_modality():
    """Should calculate NRI with 1 modality only"""
    pass

def test_nri_weight_adjustment():
    """Should adjust weights for missing modalities"""
    pass

def test_nri_confidence_penalty():
    """Should penalize confidence for missing modalities"""
    pass

def test_nri_risk_categorization():
    """Should assign correct risk category"""
    pass

def test_nri_confidence_interval():
    """Should calculate valid confidence interval"""
    pass
```

---

## Verification Checklist

When this pipeline is complete, verify:

- [ ] Can calculate NRI with all 4 modalities
- [ ] Can calculate NRI with 1-3 modalities (partial)
- [ ] Missing modality handling works
- [ ] Weight adjustment is correct
- [ ] Risk category thresholds work
- [ ] Confidence interval displays
- [ ] Frontend gauge animates smoothly
- [ ] Modality contribution breakdown shows
- [ ] Recommendations are relevant
- [ ] Processing time < 100ms

---

## Demo Script

For the hackathon video, demonstrate:

1. "Now let's see the power of multi-modal fusion"
2. Show completed assessments: Speech, Retinal, Cognitive
3. "Our NRI algorithm combines these into a unified risk score"
4. Show the circular gauge animating to final score
5. "Notice how each modality contributes to the overall risk"
6. Point to contribution breakdown chart
7. "Even with motor assessment pending, we can provide an initial risk assessment"
8. Show recommendations section
9. "This unified view is what makes NeuraLens unique"

---

## Estimated Time

| Task | Hours |
|------|-------|
| Weight calibration | 2.0 |
| Missing modality handling | 1.5 |
| Confidence intervals | 1.0 |
| Risk categorization | 1.0 |
| Frontend visualization | 2.0 |
| Testing | 1.0 |
| **Total** | **8.5 hours** |
