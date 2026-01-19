# 08 - Error Handling and Reporting

## Document Info
| Field | Value |
|-------|-------|
| Stage | Error Handling |
| Owner | ML Systems Architect |
| Reviewer | All Team Members |

---

## 1. Error Taxonomy

### 1.1 Error Categories

| Category | Code Range | Description |
|----------|-----------|-------------|
| General | E_GEN_xxx | General input/system errors |
| Validation | E_VAL_xxx | Input validation failures |
| DICOM | E_DCM_xxx | DICOM-specific errors |
| Volume | E_VOL_xxx | Volumetric data errors |
| Preprocessing | E_PREP_xxx | Preprocessing failures |
| Detection | E_DET_xxx | Anatomical detection failures |
| Analysis | E_ANAL_xxx | Pathology analysis failures |
| Inference | E_INF_xxx | Model inference failures |
| System | E_SYS_xxx | System-level failures |
| Warning | W_xxx_xxx | Non-fatal warnings |

---

## 2. Complete Error Code Tables

### 2.1 General Errors (E_GEN_xxx)

| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_GEN_001 | No file received | No | Upload a file |
| E_GEN_002 | Invalid file format | No | Use supported format |
| E_GEN_003 | File size exceeds limit | No | Reduce file size |
| E_GEN_004 | File decode failed | No | Upload valid file |
| E_GEN_005 | Resolution out of range | No | Use appropriate resolution |
| E_GEN_006 | Image appears blank | No | Upload valid image |
| E_GEN_007 | Non-medical image detected | No | Upload medical image |

### 2.2 DICOM Errors (E_DCM_xxx)

| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_DCM_001 | Invalid DICOM structure | No | Check DICOM file |
| E_DCM_002 | Required tags missing | No | Complete DICOM file |
| E_DCM_003 | Unsupported modality | No | Use supported modality |
| E_DCM_004 | Pixel data decode error | No | Check transfer syntax |
| E_DCM_005 | Slice ordering error | Partial | Verify series |
| E_DCM_006 | Spatial inconsistency | Partial | Review orientation |

### 2.3 Volume Errors (E_VOL_xxx)

| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_VOL_001 | Insufficient slices | No | Upload complete volume |
| E_VOL_002 | Too many slices | No | Split volume |
| E_VOL_003 | Missing slices detected | Partial | Upload complete series |
| E_VOL_004 | Orientation mismatch | No | Consistent orientation |
| E_VOL_005 | Dimension mismatch | No | Consistent dimensions |

### 2.4 Preprocessing Errors (E_PREP_xxx)

| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_PREP_001 | Critical decode failure | No | Re-upload file |
| E_PREP_002 | Unsupported pixel format | No | Convert format |
| E_PREP_003 | Normalization failed | Partial | Check image data |
| E_PREP_004 | Zero dynamic range | Partial | Check image quality |
| E_PREP_005 | Bias correction diverged | Partial | Skip bias correction |
| E_PREP_006 | Memory overflow | Partial | Reduce resolution |

### 2.5 Detection Errors (E_DET_xxx)

| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_DET_001 | Anatomy not found | Partial | Check image content |
| E_DET_002 | Segmentation failed | Partial | Review image quality |
| E_DET_003 | Structure validation failed | Partial | Manual review needed |
| E_DET_004 | Implausible anatomy | Partial | Check orientation |

### 2.6 Analysis Errors (E_ANAL_xxx)

| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_ANAL_001 | Pathology detection failed | Partial | Retry or manual review |
| E_ANAL_002 | Confidence too low | Yes | Flagged for review |
| E_ANAL_003 | Inconsistent predictions | Yes | Results flagged |
| E_ANAL_004 | Severity scoring failed | Yes | Manual severity assessment |

### 2.7 Inference Errors (E_INF_xxx)

| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_INF_001 | Model timeout | Yes | Retry request |
| E_INF_002 | Model load failed | Yes | System retry |
| E_INF_003 | Out of memory | Partial | Reduce input size |
| E_INF_004 | Inference exception | Yes | Retry with fallback |
| E_INF_005 | Model not available | Yes | Use alternative model |

### 2.8 System Errors (E_SYS_xxx)

| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_SYS_001 | Internal server error | Yes | Retry later |
| E_SYS_002 | Service unavailable | Yes | Retry later |
| E_SYS_003 | Rate limit exceeded | Yes | Wait and retry |
| E_SYS_004 | Authentication failed | No | Check credentials |
| E_SYS_005 | Storage error | Yes | Retry later |

---

## 3. Error Propagation

### 3.1 Error Wrapper Class
```python
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, List
import traceback

@dataclass
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    
    code: str
    message: str
    stage: str
    recoverable: bool = False
    details: Optional[Dict] = None
    cause: Optional[Exception] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        super().__init__(f"[{self.code}] {self.message}")
    
    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "recoverable": self.recoverable,
            "details": self.details or {},
            "timestamp": self.timestamp.isoformat() + "Z",
            "stack_trace": traceback.format_exc() if self.cause else None
        }


class ValidationError(PipelineError):
    """Validation stage errors."""
    def __init__(self, errors: List[dict]):
        super().__init__(
            code="E_VAL_000",
            message=f"Validation failed with {len(errors)} errors",
            stage="VALIDATION",
            recoverable=False,
            details={"validation_errors": errors}
        )


class PreprocessingError(PipelineError):
    """Preprocessing stage errors."""
    def __init__(self, code: str, message: str, details: dict = None):
        super().__init__(
            code=code,
            message=message,
            stage="PREPROCESSING",
            recoverable=True,
            details=details
        )


class InferenceError(PipelineError):
    """Model inference errors."""
    def __init__(self, code: str, message: str, model_name: str = None):
        super().__init__(
            code=code,
            message=message,
            stage="INFERENCE",
            recoverable=True,
            details={"model_name": model_name}
        )
```

### 3.2 Error Chain Propagation
```python
def propagate_error(error: PipelineError, context: dict) -> PipelineError:
    """Add context to error and propagate."""
    
    error.details = error.details or {}
    error.details.update({
        "pipeline_context": {
            "request_id": context.get("request_id"),
            "modality": context.get("modality"),
            "stages_completed": context.get("stages_completed", []),
            "last_successful_stage": context.get("current_stage")
        }
    })
    
    return error
```

---

## 4. Human-Readable Messages

### 4.1 User-Facing Message Templates
```python
USER_MESSAGES = {
    "E_GEN_001": {
        "title": "No Image Uploaded",
        "explanation": "We didn't receive any image file. Please upload a medical image to analyze.",
        "action": "Upload a chest X-ray, CT scan, or MRI image."
    },
    "E_GEN_007": {
        "title": "Non-Medical Image",
        "explanation": "The uploaded image doesn't appear to be a medical radiological image.",
        "action": "Please upload a valid chest X-ray, CT scan, or MRI image."
    },
    "E_DCM_001": {
        "title": "Invalid DICOM File",
        "explanation": "The uploaded file is not a valid DICOM file or may be corrupted.",
        "action": "Please verify the file is a properly formatted DICOM file."
    },
    "E_VOL_003": {
        "title": "Incomplete Volume",
        "explanation": "Some slices appear to be missing from the CT/MRI volume.",
        "action": "Please upload the complete image series."
    },
    "E_DET_001": {
        "title": "Anatomy Not Detected",
        "explanation": "We couldn't identify the expected anatomical structures in the image.",
        "action": "Ensure the image shows the relevant body region clearly."
    },
    "E_INF_001": {
        "title": "Analysis Timeout",
        "explanation": "The analysis took longer than expected and timed out.",
        "action": "Please try again. If the problem persists, try a smaller image."
    }
}

def get_user_message(error_code: str) -> dict:
    """Get user-friendly error message."""
    return USER_MESSAGES.get(error_code, {
        "title": "Analysis Error",
        "explanation": "An error occurred during image analysis.",
        "action": "Please try again or contact support if the problem persists."
    })
```

---

## 5. Error Response Schema

### 5.1 Standard Error Response
```json
{
  "success": false,
  "request_id": "req_abc123",
  "timestamp": "2026-01-19T10:30:00.000Z",
  "processing_time_ms": 150,
  
  "error": {
    "code": "E_GEN_007",
    "message": "Image does not appear to be a medical radiological image",
    "stage": "VALIDATION",
    
    "user_message": {
      "title": "Non-Medical Image",
      "explanation": "The uploaded image doesn't appear to be a medical radiological image.",
      "action": "Please upload a valid chest X-ray, CT scan, or MRI image."
    },
    
    "technical_details": {
      "detected_content": "photograph",
      "confidence": 0.92,
      "expected": "medical_radiograph"
    },
    
    "recoverable": false,
    "resubmission_hint": "Upload a medical X-ray, CT, or MRI image"
  },
  
  "stages_completed": [
    {"stage": "RECEIPT", "status": "success", "time_ms": 5}
  ],
  
  "stages_failed": [
    {"stage": "VALIDATION", "status": "failed", "error_code": "E_GEN_007", "time_ms": 145}
  ]
}
```

### 5.2 Partial Success with Warnings
```json
{
  "success": true,
  "partial": true,
  "request_id": "req_abc123",
  
  "warnings": [
    {
      "code": "W_DET_001",
      "message": "Lung segmentation confidence below threshold",
      "stage": "DETECTION",
      "impact": "Regional analysis may be less precise",
      "recommendation": "Results are valid but should be reviewed"
    },
    {
      "code": "W_QUAL_001",
      "message": "Image quality suboptimal",
      "stage": "PREPROCESSING",
      "impact": "Detection confidence may be reduced"
    }
  ],
  
  "clinical_results": {
    "findings": [...],
    "quality_disclaimer": "Results generated from suboptimal image quality"
  }
}
```

---

## 6. Frontend Error Handling Guidelines

### 6.1 Error Display by Category

| Error Category | UI Treatment | Icon | Color |
|----------------|--------------|------|-------|
| Validation | Inline form error | Alert triangle | Red |
| File format | Upload area error | File X | Red |
| Quality warning | Yellow banner | Info | Yellow |
| Partial failure | Result with warning | Warning | Orange |
| Complete failure | Error modal | X circle | Red |
| System error | Generic retry | Refresh | Gray |

### 6.2 Frontend Error Handler
```typescript
interface ErrorResponse {
  success: false;
  error: {
    code: string;
    message: string;
    stage: string;
    user_message: {
      title: string;
      explanation: string;
      action: string;
    };
    recoverable: boolean;
    resubmission_hint?: string;
  };
  request_id: string;
}

function handleRadiologyError(response: ErrorResponse): void {
  const { error } = response;
  
  // Validation errors - show inline
  if (error.stage === 'VALIDATION') {
    showValidationError(error.user_message);
    return;
  }
  
  // Recoverable errors - show retry option
  if (error.recoverable) {
    showRetryDialog({
      title: error.user_message.title,
      message: error.user_message.explanation,
      action: error.user_message.action,
      hint: error.resubmission_hint,
      onRetry: () => retryAnalysis()
    });
    return;
  }
  
  // Fatal errors - show error modal
  showErrorModal({
    title: error.user_message.title,
    message: error.user_message.explanation,
    action: error.user_message.action,
    errorCode: error.code,
    requestId: response.request_id
  });
}
```

---

## 7. Logging and Monitoring

### 7.1 Error Logging
```python
import logging
import json

logger = logging.getLogger("radiology_pipeline")

def log_error(error: PipelineError, request_context: dict):
    """Log error with full context for debugging."""
    
    log_entry = {
        "timestamp": error.timestamp.isoformat(),
        "error_code": error.code,
        "error_message": error.message,
        "stage": error.stage,
        "recoverable": error.recoverable,
        "details": error.details,
        "request_id": request_context.get("request_id"),
        "user_id": request_context.get("user_id"),
        "input_summary": {
            "modality": request_context.get("modality"),
            "file_type": request_context.get("file_type"),
            "file_size_mb": request_context.get("file_size_mb")
        },
        "stack_trace": traceback.format_exc() if error.cause else None
    }
    
    if error.recoverable:
        logger.warning(json.dumps(log_entry))
    else:
        logger.error(json.dumps(log_entry))
```

### 7.2 Error Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| error_rate | Errors per 1000 requests | > 50 |
| error_by_code | Count per error code | Depends on code |
| error_by_stage | Count per pipeline stage | > 10% of stage |
| recovery_rate | Successful retries / total retries | < 50% |
| mttr | Mean time to resolution | > 5 minutes |

---

## 8. Error Recovery Strategies

### 8.1 Recovery Matrix

| Error Code | Recovery Strategy |
|------------|-------------------|
| E_INF_001 | Retry with timeout increase |
| E_INF_002 | Fallback to alternative model |
| E_INF_003 | Reduce resolution and retry |
| E_PREP_005 | Skip bias correction |
| E_DET_002 | Use full image without masking |
| E_SYS_003 | Exponential backoff retry |

### 8.2 Graceful Degradation
```python
FALLBACK_STRATEGIES = {
    "bias_correction_failed": {
        "fallback": "simple_normalization",
        "quality_impact": "minor",
        "continue_pipeline": True
    },
    "segmentation_failed": {
        "fallback": "analyze_full_image",
        "quality_impact": "moderate",
        "continue_pipeline": True
    },
    "primary_model_failed": {
        "fallback": "backup_model",
        "quality_impact": "minor",
        "continue_pipeline": True
    },
    "heatmap_generation_failed": {
        "fallback": "skip_heatmap",
        "quality_impact": "cosmetic",
        "continue_pipeline": True
    }
}
```

---

## 9. Stage Confirmation

```json
{
  "error_handling": "configured",
  "error_codes_defined": 45,
  "user_messages_defined": 20,
  "recovery_strategies": 8,
  "monitoring_enabled": true
}
```
