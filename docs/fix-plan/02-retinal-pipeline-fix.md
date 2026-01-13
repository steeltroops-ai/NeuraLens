# Retinal Pipeline Fix Plan

## Overview

The Retinal Analysis pipeline handles fundus image analysis for Alzheimer's detection through retinal biomarkers. This document outlines all fixes needed for both backend and frontend components.

## Current State

### Backend Components
- `backend/app/api/v1/endpoints/retinal.py` - API endpoints
- `backend/app/ml/realtime/realtime_retinal.py` - ML processing
- `backend/app/schemas/` - Pydantic schemas

### Frontend Components
- `frontend/src/components/assessment/steps/RetinalAssessmentStep.tsx` - Assessment UI
- `frontend/src/components/dashboard/RetinalAssessment.tsx` - Dashboard view

## Issues to Fix

### Backend Issues

#### B-RT-001: Image Format Validation
**Priority**: P0
**Description**: Image format validation incomplete. Need to support JPEG, PNG, DICOM.
**Files**: `backend/app/api/v1/endpoints/retinal.py`
**Fix**:
- Add MIME type validation for JPEG, PNG
- Add DICOM file parsing support
- Validate image dimensions (min 512x512)

#### B-RT-002: Image Preprocessing
**Priority**: P0
**Description**: Image preprocessing not handling all edge cases.
**Files**: `backend/app/ml/realtime/realtime_retinal.py`
**Fix**:
- Add image normalization
- Handle different color spaces (RGB, grayscale)
- Resize to model input dimensions

#### B-RT-003: Vessel Analysis
**Priority**: P1
**Description**: Vessel segmentation accuracy needs improvement.
**Files**: `backend/app/ml/realtime/realtime_retinal.py`
**Fix**:
- Improve vessel detection algorithm
- Calculate A/V ratio more accurately
- Add vessel tortuosity measurement

#### B-RT-004: Cup-Disc Ratio
**Priority**: P1
**Description**: Optic disc detection sometimes fails.
**Files**: `backend/app/ml/realtime/realtime_retinal.py`
**Fix**:
- Add fallback detection method
- Validate detected regions
- Return confidence score for detection

#### B-RT-005: Error Handling
**Priority**: P0
**Description**: Missing error handling for corrupted images.
**Files**: `backend/app/api/v1/endpoints/retinal.py`
**Fix**:
- Add image integrity validation
- Return meaningful error messages
- Log processing failures

### Frontend Issues

#### F-RT-001: Image Upload
**Priority**: P0
**Description**: Image upload doesn't validate file size.
**Files**: `frontend/src/components/assessment/steps/RetinalAssessmentStep.tsx`
**Fix**:
- Add file size limit (max 10MB)
- Validate image dimensions client-side
- Show upload progress

#### F-RT-002: Image Preview
**Priority**: P1
**Description**: Image preview not showing correctly for all formats.
**Files**: `frontend/src/components/assessment/steps/RetinalAssessmentStep.tsx`
**Fix**:
- Handle JPEG, PNG preview
- Add zoom/pan functionality
- Show image metadata

#### F-RT-003: Drag and Drop
**Priority**: P1
**Description**: Drag and drop upload not working consistently.
**Files**: `frontend/src/components/assessment/steps/RetinalAssessmentStep.tsx`
**Fix**:
- Fix drag event handlers
- Add visual feedback on drag over
- Support multiple file selection

#### F-RT-004: Results Visualization
**Priority**: P1
**Description**: Results don't show vessel overlay on image.
**Files**: `frontend/src/components/dashboard/RetinalAssessment.tsx`
**Fix**:
- Add vessel segmentation overlay
- Highlight detected features
- Show measurement annotations

#### F-RT-005: Accessibility
**Priority**: P1
**Description**: Missing alt text and ARIA labels.
**Files**: `frontend/src/components/assessment/steps/RetinalAssessmentStep.tsx`
**Fix**:
- Add descriptive alt text
- Announce upload status
- Support keyboard file selection

## Test Cases

### Unit Tests

```python
# Backend tests
def test_retinal_analyze_valid_jpeg():
    """Test retinal analysis with valid JPEG image"""
    pass

def test_retinal_analyze_valid_png():
    """Test retinal analysis with valid PNG image"""
    pass

def test_retinal_analyze_invalid_format():
    """Test rejection of invalid image formats"""
    pass

def test_retinal_analyze_too_small():
    """Test rejection of images smaller than 512x512"""
    pass

def test_retinal_vessel_detection():
    """Test vessel segmentation accuracy"""
    pass

def test_retinal_cup_disc_ratio():
    """Test cup-disc ratio calculation"""
    pass
```

```typescript
// Frontend tests
describe('RetinalAssessmentStep', () => {
  it('should accept valid image upload', () => {});
  it('should reject oversized files', () => {});
  it('should show upload progress', () => {});
  it('should display image preview', () => {});
  it('should handle drag and drop', () => {});
});
```

### Integration Tests

```python
async def test_retinal_pipeline_end_to_end():
    """Test complete retinal analysis pipeline"""
    # 1. Upload image
    # 2. Process analysis
    # 3. Verify biomarkers
    pass
```

## API Contract

### POST /api/v1/retinal/analyze

**Request**:
```json
{
  "image_data": "base64_encoded_image",
  "format": "jpeg",
  "session_id": "uuid"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "risk_score": 0.28,
    "confidence": 0.89,
    "biomarkers": {
      "vessel_tortuosity": 0.35,
      "av_ratio": 0.72,
      "cup_disc_ratio": 0.28,
      "vessel_density": 0.65,
      "hemorrhage_count": 0,
      "drusen_detected": false
    },
    "segmentation_mask": "base64_encoded_mask",
    "processing_time_ms": 250
  }
}
```

## Kiro Spec Template

Use this to create a Kiro specification:

```
Feature: Fix Retinal Analysis Pipeline

As a developer, I want to fix all issues in the retinal analysis pipeline
so that users can reliably upload and analyze fundus images for Alzheimer's screening.

Requirements:
1. Image validation must accept JPEG, PNG formats
2. Image upload must show progress and handle errors
3. Results must display vessel overlay and all biomarkers
4. All tests must pass
```

## Verification Checklist

- [ ] Backend image validation works for JPEG, PNG
- [ ] Backend vessel detection is accurate
- [ ] Backend cup-disc ratio calculation works
- [ ] Frontend image upload with progress works
- [ ] Frontend drag and drop works
- [ ] Frontend shows image preview
- [ ] Frontend displays results with overlay
- [ ] All unit tests pass
- [ ] Integration test passes
- [ ] Accessibility audit passes
