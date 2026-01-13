# NRI Fusion Fix Plan

## Overview

The NRI (Neurological Risk Index) Fusion pipeline combines all assessment modalities into a unified risk score. This document outlines all fixes needed for the multi-modal fusion algorithm.

## Current State

### Backend Components
- `backend/app/api/v1/endpoints/nri.py` - API endpoints
- `backend/app/ml/realtime/realtime_nri.py` - Fusion algorithm
- `backend/app/schemas/assessment.py` - Pydantic schemas

### Frontend Components
- `frontend/src/components/dashboard/NRIFusionDashboard.tsx` - Dashboard view
- `frontend/src/components/assessment/steps/RiskAssessmentStep.tsx` - Results step

## Issues to Fix

### Backend Issues

#### B-NRI-001: Modality Weighting
**Priority**: P0
**Description**: Modality weights not calibrated based on clinical evidence.
**Files**: `backend/app/ml/realtime/realtime_nri.py`
**Fix**:
- Implement evidence-based weighting
- Add dynamic weight adjustment based on data quality
- Support configurable weight profiles
- Document weight rationale

#### B-NRI-002: Missing Modality Handling
**Priority**: P0
**Description**: Fusion fails when some modalities are missing.
**Files**: `backend/app/ml/realtime/realtime_nri.py`
**Fix**:
- Handle partial modality data gracefully
- Adjust weights for available modalities
- Provide confidence penalty for missing data
- Return which modalities were used

#### B-NRI-003: Bayesian Fusion
**Priority**: P1
**Description**: Bayesian fusion method not implemented correctly.
**Files**: `backend/app/ml/realtime/realtime_nri.py`
**Fix**:
- Implement proper Bayesian inference
- Add prior probability estimation
- Calculate posterior probabilities
- Provide uncertainty quantification

#### B-NRI-004: Risk Categorization
**Priority**: P0
**Description**: Risk category thresholds need calibration.
**Files**: `backend/app/ml/realtime/realtime_nri.py`
**Fix**:
- Define clear threshold boundaries
- Add intermediate categories
- Provide category descriptions
- Include clinical recommendations per category

#### B-NRI-005: Confidence Calculation
**Priority**: P1
**Description**: Overall confidence calculation needs improvement.
**Files**: `backend/app/ml/realtime/realtime_nri.py`
**Fix**:
- Combine modality confidences properly
- Account for inter-modality agreement
- Penalize conflicting results
- Provide confidence intervals

#### B-NRI-006: Trend Analysis
**Priority**: P2
**Description**: No longitudinal trend analysis.
**Files**: `backend/app/ml/realtime/realtime_nri.py`
**Fix**:
- Store historical NRI scores
- Calculate trend direction
- Detect significant changes
- Provide trend visualization data

### Frontend Issues

#### F-NRI-001: Score Display
**Priority**: P0
**Description**: NRI score display not intuitive.
**Files**: `frontend/src/components/dashboard/NRIFusionDashboard.tsx`
**Fix**:
- Add circular gauge visualization
- Show score on 0-100 scale
- Color code by risk category
- Add animation on score reveal

#### F-NRI-002: Modality Contributions
**Priority**: P0
**Description**: Modality contributions not shown clearly.
**Files**: `frontend/src/components/dashboard/NRIFusionDashboard.tsx`
**Fix**:
- Add stacked bar chart
- Show each modality's contribution
- Display modality weights
- Highlight dominant factors

#### F-NRI-003: Risk Category Explanation
**Priority**: P1
**Description**: Risk category not explained to user.
**Files**: `frontend/src/components/assessment/steps/RiskAssessmentStep.tsx`
**Fix**:
- Add category description
- Show what the category means
- Provide next steps guidance
- Include clinical context

#### F-NRI-004: Confidence Display
**Priority**: P1
**Description**: Confidence score not displayed meaningfully.
**Files**: `frontend/src/components/dashboard/NRIFusionDashboard.tsx`
**Fix**:
- Show confidence as percentage
- Add confidence interval visualization
- Explain what confidence means
- Highlight low confidence warnings

#### F-NRI-005: Comparison View
**Priority**: P2
**Description**: No comparison to previous assessments.
**Files**: `frontend/src/components/dashboard/NRIFusionDashboard.tsx`
**Fix**:
- Add historical score chart
- Show trend direction
- Highlight significant changes
- Provide trend interpretation

#### F-NRI-006: Export/Share
**Priority**: P2
**Description**: Cannot export or share NRI results.
**Files**: `frontend/src/components/dashboard/NRIFusionDashboard.tsx`
**Fix**:
- Add PDF export option
- Generate clinical report
- Support sharing with healthcare provider
- Include all modality details

## Test Cases

### Unit Tests

```python
# Backend tests
def test_nri_fusion_all_modalities():
    """Test NRI fusion with all modalities present"""
    pass

def test_nri_fusion_partial_modalities():
    """Test NRI fusion with missing modalities"""
    pass

def test_nri_bayesian_fusion():
    """Test Bayesian fusion method"""
    pass

def test_nri_risk_categorization():
    """Test risk category assignment"""
    pass

def test_nri_confidence_calculation():
    """Test confidence score calculation"""
    pass

def test_nri_weight_adjustment():
    """Test dynamic weight adjustment"""
    pass
```

```typescript
// Frontend tests
describe('NRIFusionDashboard', () => {
  it('should display NRI score correctly', () => {});
  it('should show modality contributions', () => {});
  it('should display risk category', () => {});
  it('should show confidence score', () => {});
  it('should handle missing modalities', () => {});
});
```

### Integration Tests

```python
async def test_nri_pipeline_end_to_end():
    """Test complete NRI fusion pipeline"""
    # 1. Submit all modality scores
    # 2. Calculate NRI
    # 3. Verify fusion result
    pass
```

## API Contract

### POST /api/v1/nri/calculate

**Request**:
```json
{
  "session_id": "uuid",
  "modalities": ["speech", "retinal", "motor", "cognitive"],
  "modality_scores": {
    "speech": 0.25,
    "retinal": 0.28,
    "motor": 0.35,
    "cognitive": 0.22
  },
  "modality_confidences": {
    "speech": 0.92,
    "retinal": 0.89,
    "motor": 0.88,
    "cognitive": 0.91
  },
  "fusion_method": "bayesian"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "nri_score": 28.5,
    "risk_category": "low",
    "confidence": 0.90,
    "confidence_interval": [24.2, 32.8],
    "modality_contributions": {
      "speech": 0.22,
      "retinal": 0.25,
      "motor": 0.32,
      "cognitive": 0.21
    },
    "modality_weights": {
      "speech": 0.25,
      "retinal": 0.25,
      "motor": 0.30,
      "cognitive": 0.20
    },
    "interpretation": "Low neurological risk. Continue regular health monitoring.",
    "recommendations": [
      "Annual follow-up assessment recommended",
      "Maintain healthy lifestyle habits",
      "Report any new symptoms to healthcare provider"
    ],
    "processing_time_ms": 50
  }
}
```

## Risk Categories

| Category | NRI Score Range | Description |
|----------|-----------------|-------------|
| Low | 0-25 | Minimal neurological risk indicators |
| Moderate | 26-50 | Some risk factors present, monitoring recommended |
| High | 51-75 | Elevated risk, clinical evaluation recommended |
| Very High | 76-100 | Significant risk, urgent clinical evaluation needed |

## Kiro Spec Template

Use this to create a Kiro specification:

```
Feature: Fix NRI Fusion Pipeline

As a developer, I want to fix all issues in the NRI fusion pipeline
so that users receive accurate multi-modal neurological risk assessments.

Requirements:
1. Fusion must handle missing modalities gracefully
2. Weights must be evidence-based and configurable
3. Risk categories must have clear thresholds
4. Results must show modality contributions
5. All tests must pass
```

## Verification Checklist

- [ ] Backend handles all modality combinations
- [ ] Backend Bayesian fusion works correctly
- [ ] Backend risk categorization is accurate
- [ ] Backend confidence calculation is correct
- [ ] Frontend displays NRI score clearly
- [ ] Frontend shows modality contributions
- [ ] Frontend explains risk category
- [ ] Frontend displays confidence meaningfully
- [ ] All unit tests pass
- [ ] Integration test passes
