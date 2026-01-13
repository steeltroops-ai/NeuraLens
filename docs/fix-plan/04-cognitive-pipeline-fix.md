# Cognitive Pipeline Fix Plan

## Overview

The Cognitive Testing pipeline handles memory and executive function assessment for MCI (Mild Cognitive Impairment) detection. This document outlines all fixes needed for both backend and frontend components.

## Current State

### Backend Components
- `backend/app/api/v1/endpoints/cognitive.py` - API endpoints
- `backend/app/ml/realtime/realtime_cognitive.py` - ML processing
- `backend/app/schemas/assessment.py` - Pydantic schemas

### Frontend Components
- `frontend/src/components/assessment/steps/CognitiveAssessmentStep.tsx` - Assessment UI
- `frontend/src/components/dashboard/CognitiveAssessment.tsx` - Dashboard view

## Issues to Fix

### Backend Issues

#### B-CG-001: Test Battery Validation
**Priority**: P0
**Description**: Test battery validation incomplete for all cognitive domains.
**Files**: `backend/app/api/v1/endpoints/cognitive.py`
**Fix**:
- Validate memory test results structure
- Validate attention test results
- Validate executive function results
- Check score ranges (0.0-1.0)

#### B-CG-002: Scoring Algorithm
**Priority**: P0
**Description**: Cognitive scoring algorithm needs calibration.
**Files**: `backend/app/ml/realtime/realtime_cognitive.py`
**Fix**:
- Implement age-adjusted scoring
- Add education-level normalization
- Calculate domain-specific scores
- Compute composite cognitive index

#### B-CG-003: Memory Assessment
**Priority**: P0
**Description**: Memory assessment metrics incomplete.
**Files**: `backend/app/ml/realtime/realtime_cognitive.py`
**Fix**:
- Calculate immediate recall score
- Calculate delayed recall score
- Measure recognition accuracy
- Add learning curve analysis

#### B-CG-004: Attention Assessment
**Priority**: P1
**Description**: Attention metrics need improvement.
**Files**: `backend/app/ml/realtime/realtime_cognitive.py`
**Fix**:
- Measure sustained attention
- Calculate selective attention
- Add divided attention score
- Include reaction time variability

#### B-CG-005: Executive Function
**Priority**: P1
**Description**: Executive function scoring incomplete.
**Files**: `backend/app/ml/realtime/realtime_cognitive.py`
**Fix**:
- Add planning score
- Calculate inhibition score
- Measure cognitive flexibility
- Include working memory component

#### B-CG-006: Error Handling
**Priority**: P0
**Description**: Missing error handling for incomplete test results.
**Files**: `backend/app/api/v1/endpoints/cognitive.py`
**Fix**:
- Handle partial test completion
- Return scores for completed domains only
- Provide recommendations for incomplete tests

### Frontend Issues

#### F-CG-001: Test Instructions
**Priority**: P0
**Description**: Test instructions not clear enough.
**Files**: `frontend/src/components/assessment/steps/CognitiveAssessmentStep.tsx`
**Fix**:
- Add step-by-step instructions
- Include practice trials
- Show example responses
- Provide audio instructions option

#### F-CG-002: Memory Test UI
**Priority**: P0
**Description**: Memory test interface needs improvement.
**Files**: `frontend/src/components/assessment/steps/CognitiveAssessmentStep.tsx`
**Fix**:
- Display word lists clearly
- Add recall input interface
- Show timer for delayed recall
- Provide recognition test UI

#### F-CG-003: Attention Test UI
**Priority**: P0
**Description**: Attention test not responsive enough.
**Files**: `frontend/src/components/assessment/steps/CognitiveAssessmentStep.tsx`
**Fix**:
- Optimize stimulus presentation timing
- Add precise reaction time measurement
- Show visual feedback on response
- Handle rapid responses correctly

#### F-CG-004: Progress Tracking
**Priority**: P1
**Description**: No progress indicator for multi-part tests.
**Files**: `frontend/src/components/assessment/steps/CognitiveAssessmentStep.tsx`
**Fix**:
- Show test battery progress
- Display current domain being tested
- Estimate remaining time
- Allow pause/resume

#### F-CG-005: Results Display
**Priority**: P1
**Description**: Results don't show domain-specific scores.
**Files**: `frontend/src/components/dashboard/CognitiveAssessment.tsx`
**Fix**:
- Display memory domain scores
- Show attention domain scores
- Display executive function scores
- Add radar chart visualization

#### F-CG-006: Accessibility
**Priority**: P1
**Description**: Cognitive tests not accessible for all users.
**Files**: `frontend/src/components/assessment/steps/CognitiveAssessmentStep.tsx`
**Fix**:
- Add screen reader support
- Provide keyboard navigation
- Allow extended time option
- Support high contrast mode

## Test Cases

### Unit Tests

```python
# Backend tests
def test_cognitive_analyze_valid_results():
    """Test cognitive analysis with valid test results"""
    pass

def test_cognitive_analyze_partial_results():
    """Test handling of partial test completion"""
    pass

def test_cognitive_memory_scoring():
    """Test memory domain scoring accuracy"""
    pass

def test_cognitive_attention_scoring():
    """Test attention domain scoring accuracy"""
    pass

def test_cognitive_executive_scoring():
    """Test executive function scoring accuracy"""
    pass

def test_cognitive_age_adjustment():
    """Test age-adjusted scoring"""
    pass
```

```typescript
// Frontend tests
describe('CognitiveAssessmentStep', () => {
  it('should display test instructions', () => {});
  it('should run memory test correctly', () => {});
  it('should measure reaction times accurately', () => {});
  it('should show progress through test battery', () => {});
  it('should display domain-specific results', () => {});
});
```

### Integration Tests

```python
async def test_cognitive_pipeline_end_to_end():
    """Test complete cognitive assessment pipeline"""
    # 1. Submit test results
    # 2. Process analysis
    # 3. Verify domain scores
    pass
```

## API Contract

### POST /api/v1/cognitive/analyze

**Request**:
```json
{
  "test_battery": ["memory", "attention", "executive"],
  "test_results": {
    "memory": {
      "immediate_recall": 0.85,
      "delayed_recall": 0.78,
      "recognition": 0.92
    },
    "attention": {
      "sustained_attention": 0.82,
      "selective_attention": 0.79,
      "divided_attention": 0.71,
      "reaction_time_ms": 450
    },
    "executive": {
      "planning": 0.76,
      "inhibition": 0.83,
      "flexibility": 0.74,
      "working_memory": 0.80
    }
  },
  "difficulty_level": "standard",
  "demographics": {
    "age": 65,
    "education_years": 16
  },
  "session_id": "uuid"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "risk_score": 0.28,
    "confidence": 0.91,
    "domain_scores": {
      "memory": 0.85,
      "attention": 0.77,
      "executive": 0.78
    },
    "composite_score": 0.80,
    "percentile": 65,
    "interpretation": "Within normal limits for age and education",
    "recommendations": [
      "Continue regular cognitive activities",
      "Follow up in 12 months"
    ],
    "processing_time_ms": 120
  }
}
```

## Kiro Spec Template

Use this to create a Kiro specification:

```
Feature: Fix Cognitive Assessment Pipeline

As a developer, I want to fix all issues in the cognitive assessment pipeline
so that users can reliably complete cognitive tests for MCI screening.

Requirements:
1. Test battery must validate all cognitive domains
2. Scoring must be age and education adjusted
3. Test UI must be clear and responsive
4. Results must display domain-specific scores
5. All tests must pass
```

## Verification Checklist

- [ ] Backend test battery validation works
- [ ] Backend scoring algorithm is accurate
- [ ] Backend age adjustment works correctly
- [ ] Frontend test instructions are clear
- [ ] Frontend memory test UI works
- [ ] Frontend attention test measures timing accurately
- [ ] Frontend shows progress through battery
- [ ] Frontend displays domain scores
- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] Accessibility audit passes
