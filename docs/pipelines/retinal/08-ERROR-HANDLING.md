# Retinal Pipeline - Error Handling & Reporting

## Document Info
| Field | Value |
|-------|-------|
| Version | 4.0.0 |
| Pipeline Stage | 8 - Error Handling |

---

## 1. Error Taxonomy

### 1.1 Error Categories
| Category | Code Range | Description |
|----------|------------|-------------|
| Validation | VAL_001-099 | Input validation failures |
| Preprocessing | PRE_001-099 | Image preprocessing failures |
| Model | MOD_001-099 | ML model inference failures |
| Anatomical | ANA_001-099 | Anatomical detection failures |
| Clinical | CLI_001-099 | Clinical assessment failures |
| System | SYS_001-099 | System/infrastructure failures |

### 1.2 Complete Error Code Registry

#### Validation Errors (VAL)
| Code | Message | User Guidance |
|------|---------|---------------|
| VAL_001 | Invalid file format | Upload JPEG, PNG, or TIFF image |
| VAL_002 | Corrupted file header | Re-export or re-capture the image |
| VAL_003 | Invalid file extension | Rename with correct extension |
| VAL_010 | Resolution too low | Minimum 512x512 pixels required |
| VAL_011 | Resolution warning | 1024x1024 recommended for best results |
| VAL_012 | Resolution too high | Maximum 8192x8192 pixels |
| VAL_020 | Poor illumination | Recapture with proper lighting |
| VAL_021 | Uneven illumination | Adjust fundus camera settings |
| VAL_022 | Overexposed image | Reduce flash intensity |
| VAL_023 | Underexposed image | Increase flash intensity |
| VAL_030 | No fundus detected | Ensure proper patient positioning |
| VAL_040 | Not a retinal image | Upload fundus photograph only |
| VAL_050 | Empty file | Upload non-empty image file |
| VAL_051 | Undecodable image | File may be corrupted |

#### Preprocessing Errors (PRE)
| Code | Message | Recovery |
|------|---------|----------|
| PRE_001 | Color normalization failed | Continue with original colors |
| PRE_002 | CLAHE enhancement failed | Skip contrast enhancement |
| PRE_003 | Artifact removal failed | Continue with artifacts |
| PRE_010 | Quality gate failed | Image unusable for analysis |

#### Model Errors (MOD)
| Code | Message | Recovery |
|------|---------|----------|
| MOD_001 | Model loading failed | System error - retry later |
| MOD_002 | Inference timeout | Reduce image size, retry |
| MOD_003 | Invalid output shape | Model mismatch - contact support |
| MOD_010 | Ensemble disagreement | Report with high uncertainty |
| MOD_020 | GPU memory exceeded | Fall back to CPU inference |

#### Anatomical Errors (ANA)
| Code | Message | Impact |
|------|---------|--------|
| ANA_001 | Optic disc not detected | CDR unavailable, reduced confidence |
| ANA_002 | Macula not detected | DME assessment limited |
| ANA_003 | Vessel segmentation failed | Vessel biomarkers unavailable |
| ANA_010 | Anatomical inconsistency | Flag for manual review |
| ANA_020 | Multiple disc candidates | Use highest confidence |

#### Clinical Errors (CLI)
| Code | Message | Recovery |
|------|---------|----------|
| CLI_001 | DR grading failed | Report "Unable to grade" |
| CLI_002 | Risk calculation error | Use default risk level |
| CLI_003 | Recommendation generation failed | Provide generic recommendations |

#### System Errors (SYS)
| Code | Message | Recovery |
|------|---------|----------|
| SYS_001 | Database connection failed | Retry with backoff |
| SYS_002 | Storage write failed | Retry, then fail |
| SYS_003 | Memory allocation failed | Reduce batch size |
| SYS_010 | Rate limit exceeded | Queue and retry |

---

## 2. Error Classification

### 2.1 Severity Levels
```python
class ErrorSeverity(str, Enum):
    FATAL = "fatal"      # Pipeline cannot continue
    ERROR = "error"      # Stage failed, may recover
    WARNING = "warning"  # Issue noted, continuing
    INFO = "info"        # Informational only
```

### 2.2 Error Classification Logic
```python
def classify_error(error_code: str, context: dict) -> dict:
    category = error_code.split("_")[0]
    
    classification = {
        "code": error_code,
        "category": category,
        "severity": determine_severity(error_code, context),
        "recoverable": is_recoverable(error_code),
        "user_actionable": is_user_actionable(error_code),
        "resubmission_helps": resubmission_recommended(error_code),
    }
    
    return classification

def determine_severity(code: str, context: dict) -> ErrorSeverity:
    FATAL_CODES = ["VAL_001", "VAL_050", "VAL_051", "MOD_001", "PRE_010"]
    ERROR_CODES = ["VAL_010", "VAL_040", "MOD_002", "ANA_010"]
    WARNING_CODES = ["VAL_011", "PRE_001", "PRE_002", "ANA_001"]
    
    if code in FATAL_CODES:
        return ErrorSeverity.FATAL
    elif code in ERROR_CODES:
        return ErrorSeverity.ERROR
    elif code in WARNING_CODES:
        return ErrorSeverity.WARNING
    else:
        return ErrorSeverity.INFO
```

---

## 3. Error Propagation

### 3.1 Error Chain
```python
@dataclass
class PipelineError:
    stage: str
    error_type: str
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    cause: Optional["PipelineError"] = None  # For chained errors

    def to_response(self) -> dict:
        return {
            "stage": self.stage,
            "error_type": self.error_type,
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }
```

### 3.2 Error Aggregation
```python
def aggregate_errors(errors: List[PipelineError]) -> dict:
    """Aggregate multiple errors into summary."""
    if not errors:
        return None
    
    primary_error = errors[0]  # First error is primary
    
    return {
        "primary_error": primary_error.to_response(),
        "error_count": len(errors),
        "all_errors": [e.to_response() for e in errors],
        "stages_affected": list(set(e.stage for e in errors)),
        "most_severe": max(errors, key=lambda e: severity_rank(e.code)).code,
    }
```

---

## 4. Human-Readable Messages

### 4.1 Message Templates
```python
ERROR_MESSAGES = {
    "VAL_001": {
        "title": "Invalid File Format",
        "description": "The uploaded file is not a supported image format.",
        "guidance": "Please upload a JPEG, PNG, or TIFF fundus photograph.",
        "technical": "Expected MIME type image/jpeg, image/png, or image/tiff."
    },
    "VAL_010": {
        "title": "Image Resolution Too Low",
        "description": "The image resolution is below the minimum required for accurate analysis.",
        "guidance": "Please upload an image with at least 512x512 pixels. 1024x1024 or higher is recommended.",
        "technical": "Minimum resolution: 512x512. Provided: {width}x{height}."
    },
    "VAL_040": {
        "title": "Not a Retinal Image",
        "description": "The uploaded image does not appear to be a fundus photograph.",
        "guidance": "Please upload a retinal fundus image captured with a fundus camera.",
        "technical": "Fundus detection confidence: {confidence}. Threshold: 0.5."
    },
    "PRE_010": {
        "title": "Image Quality Too Low",
        "description": "The image quality is too poor for reliable analysis.",
        "guidance": "Please recapture the fundus image with better focus, lighting, and patient positioning.",
        "technical": "Quality score: {quality_score}. Minimum required: 0.3."
    },
    "MOD_002": {
        "title": "Analysis Timeout",
        "description": "The analysis took too long and was interrupted.",
        "guidance": "Please try again. If the problem persists, try with a smaller image.",
        "technical": "Inference timeout after {timeout_ms}ms."
    },
    "ANA_001": {
        "title": "Optic Disc Not Detected",
        "description": "The optic disc could not be reliably detected in the image.",
        "guidance": "Analysis will continue with reduced accuracy for glaucoma-related biomarkers.",
        "technical": "Optic disc detection confidence below threshold."
    },
}

def format_error_message(code: str, context: dict = None) -> dict:
    template = ERROR_MESSAGES.get(code, {
        "title": "Unknown Error",
        "description": "An unexpected error occurred.",
        "guidance": "Please try again or contact support.",
        "technical": f"Error code: {code}"
    })
    
    # Format with context if provided
    if context:
        for key in ["description", "guidance", "technical"]:
            template[key] = template[key].format(**context)
    
    return template
```

---

## 5. Machine-Readable Error Codes

### 5.1 Error Response Structure
```python
@dataclass
class ErrorResponse:
    # Primary identification
    code: str                    # e.g., "VAL_010"
    category: str                # e.g., "validation"
    
    # Human-readable
    title: str                   # e.g., "Image Resolution Too Low"
    message: str                 # User-friendly description
    guidance: str                # What user should do
    
    # Machine-readable
    stage: str                   # Pipeline stage where error occurred
    severity: str                # fatal, error, warning, info
    recoverable: bool            # Can pipeline continue?
    
    # Action guidance
    resubmission_recommended: bool
    retry_recommended: bool
    contact_support: bool
    
    # Technical details (optional)
    technical_details: Optional[dict] = None
    
    # Timestamps
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
```

---

## 6. Frontend-Compatible Response Schemas

### 6.1 Error Response JSON
```json
{
  "success": false,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-01-19T08:00:00Z",
  
  "error": {
    "code": "VAL_010",
    "category": "validation",
    "title": "Image Resolution Too Low",
    "message": "The image resolution is below the minimum required for accurate analysis.",
    "guidance": "Please upload an image with at least 512x512 pixels.",
    
    "stage": "input_validation",
    "severity": "error",
    "recoverable": false,
    
    "resubmission_recommended": true,
    "retry_recommended": false,
    "contact_support": false,
    
    "technical_details": {
      "provided_resolution": [256, 256],
      "minimum_resolution": [512, 512],
      "recommended_resolution": [1024, 1024]
    }
  },
  
  "stages_completed": [],
  "processing_time_ms": 125
}
```

### 6.2 Warning Response (Analysis Continues)
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  
  "warnings": [
    {
      "code": "ANA_001",
      "title": "Optic Disc Not Detected",
      "message": "Glaucoma biomarkers may be less accurate.",
      "affected_metrics": ["cup_disc_ratio", "rnfl_status"]
    }
  ],
  
  "results": {
    "...": "normal results with reduced confidence for affected metrics"
  },
  
  "confidence_adjustments": {
    "cup_disc_ratio": 0.5,
    "overall": 0.85
  }
}
```

### 6.3 Frontend Error Display Mapping
```typescript
interface FrontendErrorDisplay {
  // For toast/alert
  title: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
  
  // For detailed view
  guidance: string;
  technicalDetails?: Record<string, unknown>;
  
  // For action buttons
  actions: Array<{
    label: string;
    action: 'resubmit' | 'retry' | 'contact_support' | 'dismiss';
  }>;
}

function mapErrorToDisplay(error: ErrorResponse): FrontendErrorDisplay {
  const actions = [];
  
  if (error.resubmission_recommended) {
    actions.push({ label: 'Upload New Image', action: 'resubmit' });
  }
  if (error.retry_recommended) {
    actions.push({ label: 'Try Again', action: 'retry' });
  }
  if (error.contact_support) {
    actions.push({ label: 'Contact Support', action: 'contact_support' });
  }
  actions.push({ label: 'Dismiss', action: 'dismiss' });
  
  return {
    title: error.title,
    message: error.message,
    severity: error.severity === 'fatal' ? 'error' : error.severity,
    guidance: error.guidance,
    technicalDetails: error.technical_details,
    actions,
  };
}
```
