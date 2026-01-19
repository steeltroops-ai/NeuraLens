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
| Validation | E_VAL_xxx | Input validation failures |
| Preprocessing | E_PREP_xxx | Signal/image processing failures |
| Detection | E_DET_xxx | Structure detection failures |
| Analysis | E_ANAL_xxx | Functional analysis failures |
| Inference | E_INF_xxx | Model inference failures |
| System | E_SYS_xxx | System-level failures |

### 1.2 Complete Error Code Table

#### Validation Errors
| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_VAL_001 | No valid modality provided | No | Upload echo or ECG |
| E_VAL_002 | Invalid file format | No | Use supported format |
| E_VAL_003 | File size exceeds limit | No | Reduce file size |
| E_VAL_004 | Invalid sample rate | No | Use 100-1000 Hz |
| E_VAL_005 | Signal duration too short | No | Record longer signal |
| E_VAL_006 | Image resolution too low | No | Use higher resolution |
| E_VAL_007 | Corrupted file detected | No | Re-upload file |
| E_VAL_008 | Invalid metadata schema | Yes | Fix JSON format |

#### Preprocessing Errors
| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_PREP_001 | All frames below quality | Partial | Use clearer images |
| E_PREP_002 | Video decode failed | No | Convert video format |
| E_PREP_003 | Insufficient cardiac cycles | Partial | Record longer video |
| E_PREP_004 | Signal flatline detected | No | Check ECG electrodes |
| E_PREP_005 | Excessive noise in signal | Partial | Reduce interference |
| E_PREP_006 | Baseline wander extreme | Yes | Retry with filtering |

#### Detection Errors
| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_DET_001 | View not recognizable | Partial | Use standard views |
| E_DET_002 | LV not detected | Partial | Ensure LV visible |
| E_DET_003 | Temporal inconsistency | Yes | Retry analysis |
| E_DET_004 | Implausible anatomy | Partial | Check image quality |

#### Analysis Errors
| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_ANAL_001 | EF calculation failed | Partial | Manual review needed |
| E_ANAL_002 | R-peak detection failed | Partial | Check signal quality |
| E_ANAL_003 | HRV computation failed | Yes | Verify beat detection |
| E_ANAL_004 | Inconsistent metrics | Yes | Results flagged |

#### Inference Errors
| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_INF_001 | Model timeout | Yes | Retry request |
| E_INF_002 | Model load failed | Yes | System retry |
| E_INF_003 | Out of memory | Yes | Reduce input size |
| E_INF_004 | Inference failed | Yes | Retry with fallback |

#### System Errors
| Code | Message | Recoverable | User Action |
|------|---------|-------------|-------------|
| E_SYS_001 | Internal server error | Yes | Retry later |
| E_SYS_002 | Service unavailable | Yes | Retry later |
| E_SYS_003 | Rate limit exceeded | Yes | Wait and retry |
| E_SYS_004 | Authentication failed | No | Check credentials |

---

## 2. Error Propagation

### 2.1 Error Wrapping
```python
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    
    def __init__(
        self,
        code: str,
        message: str,
        stage: str,
        recoverable: bool = False,
        details: dict = None,
        cause: Exception = None
    ):
        self.code = code
        self.message = message
        self.stage = stage
        self.recoverable = recoverable
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.utcnow()
        
        super().__init__(f"[{code}] {message}")
    
    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "recoverable": self.recoverable,
            "details": self.details,
            "timestamp": self.timestamp.isoformat() + "Z"
        }


class ValidationError(PipelineError):
    def __init__(self, errors: List[dict]):
        super().__init__(
            code="E_VAL_000",
            message=f"Validation failed with {len(errors)} errors",
            stage="VALIDATION",
            recoverable=False,
            details={"validation_errors": errors}
        )


class PreprocessingError(PipelineError):
    def __init__(self, code: str, message: str, details: dict = None):
        super().__init__(
            code=code,
            message=message,
            stage="PREPROCESSING",
            recoverable=True,
            details=details
        )
```

### 2.2 Error Chain
```python
def propagate_error(error: PipelineError, context: dict) -> PipelineError:
    """Add context to error and propagate."""
    
    error.details.update({
        "pipeline_context": context,
        "stages_completed": context.get("stages_completed", []),
        "last_successful_stage": context.get("current_stage")
    })
    
    return error
```

---

## 3. Human-Readable Explanations

### 3.1 User-Facing Messages
```python
USER_MESSAGES = {
    "E_VAL_001": {
        "title": "No Input Provided",
        "explanation": "We didn't receive any analyzable data. Please upload either an echocardiogram image/video or an ECG recording.",
        "action": "Upload at least one type of cardiac data to proceed."
    },
    "E_VAL_005": {
        "title": "ECG Recording Too Short",
        "explanation": "The ECG recording needs to be at least 5 seconds long to perform accurate heart rate variability analysis.",
        "action": "Please provide a longer ECG recording (recommended: 30 seconds or more)."
    },
    "E_PREP_005": {
        "title": "Signal Quality Issue",
        "explanation": "We detected significant noise in your ECG signal, which may affect the accuracy of the analysis.",
        "action": "Try recording in a quieter environment, ensure good electrode contact, and avoid movement during recording."
    },
    "E_DET_002": {
        "title": "Heart Structure Not Visible",
        "explanation": "We couldn't clearly identify the left ventricle in the echocardiogram images.",
        "action": "Please ensure the echocardiogram shows a clear view of the heart chambers. Standard views like Apical 4-Chamber work best."
    }
}

def get_user_message(error_code: str) -> dict:
    """Get user-friendly error message."""
    return USER_MESSAGES.get(error_code, {
        "title": "Processing Error",
        "explanation": "An error occurred during analysis.",
        "action": "Please try again or contact support."
    })
```

---

## 4. Error Response Schema

### 4.1 Standard Error Response
```json
{
  "success": false,
  "error": {
    "code": "E_VAL_005",
    "message": "Signal duration 3.5s below minimum 5s",
    "stage": "VALIDATION",
    "timestamp": "2026-01-19T10:30:00.000Z",
    
    "user_message": {
      "title": "ECG Recording Too Short",
      "explanation": "The ECG recording needs to be at least 5 seconds long.",
      "action": "Please provide a longer ECG recording."
    },
    
    "technical_details": {
      "actual_duration_sec": 3.5,
      "minimum_duration_sec": 5,
      "sample_rate_hz": 500,
      "samples_received": 1750
    },
    
    "recoverable": false,
    "resubmission_hint": "Increase ECG duration to at least 5 seconds",
    
    "pipeline_context": {
      "stages_completed": ["RECEIPT"],
      "last_successful_stage": "RECEIPT",
      "modalities_received": ["ecg_signal"]
    }
  },
  "request_id": "req_abc123",
  "processing_time_ms": 45
}
```

### 4.2 Partial Success Response
```json
{
  "success": true,
  "partial": true,
  "warnings": [
    {
      "code": "W_DET_001",
      "message": "Echo analysis skipped due to low quality",
      "stage": "DETECTION",
      "impact": "EF and wall motion not available"
    }
  ],
  "data": {
    "ecg_analysis": {
      "heart_rate_bpm": 72,
      "rhythm": "Normal Sinus Rhythm"
    },
    "echo_analysis": null
  },
  "request_id": "req_abc123"
}
```

---

## 5. Frontend Error Handling

### 5.1 Error Display Guidelines
| Error Category | UI Treatment |
|----------------|--------------|
| Validation | Inline form errors |
| Quality warnings | Yellow banner, continue option |
| Partial failure | Show available results + warning |
| Complete failure | Error modal with action |
| System error | Generic retry message |

### 5.2 Frontend Error Schema
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

function handleError(error: ErrorResponse) {
  if (error.error.stage === 'VALIDATION') {
    showValidationError(error.error.user_message);
  } else if (error.error.recoverable) {
    showRetryDialog(error.error.user_message, error.error.resubmission_hint);
  } else {
    showErrorModal(error.error.user_message);
  }
}
```

---

## 6. Logging and Monitoring

### 6.1 Error Logging
```python
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
            "modalities": request_context.get("modalities", []),
            "file_sizes": request_context.get("file_sizes", {})
        },
        "stack_trace": traceback.format_exc() if error.cause else None
    }
    
    logger.error(json.dumps(log_entry))
```

### 6.2 Error Metrics
| Metric | Description |
|--------|-------------|
| error_rate | Errors per 1000 requests |
| error_by_code | Count per error code |
| error_by_stage | Count per pipeline stage |
| recovery_rate | Successful retries / total retries |
| mttr | Mean time to resolution |
