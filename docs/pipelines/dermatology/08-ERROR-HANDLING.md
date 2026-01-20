# 08 - Error Handling and Reporting

## Purpose
Define comprehensive error taxonomy, propagation patterns, and user-facing error responses for the dermatology pipeline.

---

## 1. Error Taxonomy

### 1.1 Error Categories

```python
class ErrorCategory(Enum):
    """Top-level error categories."""
    
    VALIDATION = "validation"           # Input validation errors
    PREPROCESSING = "preprocessing"     # Image preprocessing errors
    SEGMENTATION = "segmentation"       # Lesion detection/segmentation errors
    CLASSIFICATION = "classification"   # Model inference errors
    SCORING = "scoring"                 # Risk scoring errors
    EXPLANATION = "explanation"         # Explanation generation errors
    SYSTEM = "system"                   # Infrastructure errors
    TIMEOUT = "timeout"                 # Processing timeout errors
```

### 1.2 Error Code Structure

```
Error Code Format: E_{CATEGORY}_{SUBCODE}

Categories:
  VAL = Validation
  PRE = Preprocessing
  SEG = Segmentation
  CLS = Classification
  SCR = Scoring
  EXP = Explanation
  SYS = System
  TMO = Timeout

Example: E_VAL_011 = Validation error, image too blurry
```

---

## 2. Complete Error Code Table

### 2.1 Validation Errors (E_VAL_xxx)

| Code | Message | User Message | Recoverable | Action |
|------|---------|--------------|-------------|--------|
| `E_VAL_001` | Invalid file type | "This file type is not supported. Please upload a JPEG, PNG, or HEIC image." | Yes | Upload different file |
| `E_VAL_002` | File too large | "Your image is too large (max 50MB). Please use a smaller image." | Yes | Compress or resize |
| `E_VAL_003` | File too small | "The uploaded file is too small to be a valid image." | Yes | Upload valid image |
| `E_VAL_004` | Corrupted file | "The image file appears to be corrupted. Please try uploading again." | Yes | Re-upload |
| `E_VAL_005` | Empty file | "No image data was received. Please select an image and try again." | Yes | Upload image |
| `E_VAL_010` | Resolution too low | "Image resolution is too low for reliable analysis. Please use at least 2MP." | Yes | Higher resolution |
| `E_VAL_011` | Image too blurry | "The image is too blurry. Please retake with better focus." | Yes | Retake photo |
| `E_VAL_012` | Image overexposed | "The image is overexposed (too bright). Please retake with less light." | Yes | Retake photo |
| `E_VAL_013` | Image underexposed | "The image is too dark. Please improve lighting and retake." | Yes | Retake photo |
| `E_VAL_014` | Uneven lighting | "Uneven lighting detected. Please use diffuse, even lighting." | Yes | Retake photo |
| `E_VAL_015` | Severe color cast | "Strong color distortion detected. Please check lighting and camera settings." | Yes | Retake photo |
| `E_VAL_020` | No skin detected | "This image does not appear to show skin. Please upload a skin lesion image." | Yes | Upload skin image |
| `E_VAL_021` | No lesion detected | "No lesion could be detected. Please ensure the lesion is clearly visible." | Yes | Retake photo |
| `E_VAL_022` | Multiple lesions | "Multiple lesions detected. Please submit one lesion at a time." | Yes | Crop to single |
| `E_VAL_023` | Lesion at edge | "The lesion is too close to the edge. Please center the lesion." | Yes | Retake centered |
| `E_VAL_024` | Excessive occlusion | "The lesion is partially hidden. Please remove hair or obstructions." | Yes | Clear and retake |
| `E_VAL_025` | Extreme blur | "Image is extremely blurry. Unable to process." | Yes | Retake with care |

### 2.2 Preprocessing Errors (E_PRE_xxx)

| Code | Message | User Message | Recoverable | Action |
|------|---------|--------------|-------------|--------|
| `E_PRE_001` | Color correction failed | "Color processing failed. Proceeding with reduced accuracy." | Warning | Continue |
| `E_PRE_002` | Illumination correction failed | "Lighting correction could not be applied." | Warning | Continue |
| `E_PRE_003` | Hair removal failed | "Hair removal processing failed." | Warning | Continue |
| `E_PRE_004` | Contrast enhancement failed | "Image enhancement failed." | Warning | Continue |
| `E_PRE_005` | Resize failed | "Image resizing failed." | No | System error |
| `E_PRE_010` | Memory exceeded | "Image too large to process. Please use a smaller image." | Yes | Smaller image |

### 2.3 Segmentation Errors (E_SEG_xxx)

| Code | Message | User Message | Recoverable | Action |
|------|---------|--------------|-------------|--------|
| `E_SEG_001` | No lesion detected | "No lesion could be identified in the image." | Yes | Retake photo |
| `E_SEG_002` | Low confidence segmentation | "Lesion boundary could not be reliably determined." | Warning | Continue with caution |
| `E_SEG_003` | Fragmented segmentation | "Lesion appears fragmented. Using bounding box." | Warning | Continue |
| `E_SEG_004` | Lesion too small | "Detected lesion is too small for reliable analysis." | Yes | Closer photo |
| `E_SEG_005` | Lesion too large | "Lesion fills too much of the image. Please include some surrounding skin." | Yes | Retake farther |
| `E_SEG_006` | Segmentation model failed | "Lesion detection system error." | No | Retry/report |
| `E_SEG_007` | Implausible geometry | "Detected shape does not appear to be a skin lesion." | Yes | Retake |

### 2.4 Classification Errors (E_CLS_xxx)

| Code | Message | User Message | Recoverable | Action |
|------|---------|--------------|-------------|--------|
| `E_CLS_001` | Model inference failed | "Analysis model encountered an error." | No | Retry |
| `E_CLS_002` | Ensemble disagreement | "Models produced inconsistent results." | Warning | Expert review |
| `E_CLS_003` | Low confidence | "Analysis confidence is low." | Warning | Continue with caution |
| `E_CLS_004` | GPU memory error | "Processing resources unavailable." | No | Retry later |
| `E_CLS_005` | Model not loaded | "Analysis model is not available." | No | System error |

### 2.5 Scoring Errors (E_SCR_xxx)

| Code | Message | User Message | Recoverable | Action |
|------|---------|--------------|-------------|--------|
| `E_SCR_001` | Feature extraction failed | "Unable to extract lesion features." | No | Retry |
| `E_SCR_002` | ABCDE computation failed | "Feature scoring failed." | Warning | Use model only |
| `E_SCR_003` | Risk calculation failed | "Risk score computation failed." | No | Retry |
| `E_SCR_004` | Prior comparison failed | "Could not compare to previous image." | Warning | Skip comparison |

### 2.6 Explanation Errors (E_EXP_xxx)

| Code | Message | User Message | Recoverable | Action |
|------|---------|--------------|-------------|--------|
| `E_EXP_001` | Explanation generation failed | "AI explanation could not be generated." | Warning | Use template |
| `E_EXP_002` | LLM timeout | "Explanation service timeout." | Warning | Use template |
| `E_EXP_003` | Heatmap generation failed | "Visual explanation could not be created." | Warning | Omit heatmap |

### 2.7 System Errors (E_SYS_xxx)

| Code | Message | User Message | Recoverable | Action |
|------|---------|--------------|-------------|--------|
| `E_SYS_001` | Internal server error | "An unexpected error occurred. Please try again." | No | Retry |
| `E_SYS_002` | Service unavailable | "The analysis service is temporarily unavailable." | No | Retry later |
| `E_SYS_003` | Database error | "Data storage error occurred." | No | Retry |
| `E_SYS_004` | Memory exhausted | "Server resources exhausted." | No | Retry later |
| `E_SYS_005` | GPU unavailable | "GPU processing unavailable." | No | Retry later |

### 2.8 Timeout Errors (E_TMO_xxx)

| Code | Message | User Message | Recoverable | Action |
|------|---------|--------------|-------------|--------|
| `E_TMO_001` | Processing timeout | "Analysis took too long. Please try again with a simpler image." | Yes | Retry |
| `E_TMO_002` | Upload timeout | "Image upload timed out. Please check your connection." | Yes | Retry |
| `E_TMO_003` | Model timeout | "Model processing timed out." | No | Retry |

---

## 3. Error Response Schema

### 3.1 Error Response Structure

```typescript
interface ErrorResponse {
  success: false;
  error: {
    code: string;                    // e.g., "E_VAL_011"
    category: string;                // e.g., "validation"
    message: string;                 // Technical message
    
    user_message: {
      title: string;                 // Short title
      explanation: string;           // Detailed explanation
      action: string;                // What user should do
    };
    
    recoverable: boolean;            // Can user retry?
    retry_recommended: boolean;      // Should they retry?
    
    // For retake guidance
    image_guidance?: {
      issue: string;
      tips: string[];
      example_image_url?: string;
    };
    
    // Technical details (for logging)
    details?: {
      stage: string;
      timestamp: string;
      request_id: string;
      stack_trace?: string;          // Only in dev mode
    };
  };
  
  // Partial results if available
  partial_results?: {
    stages_completed: string[];
    last_successful_output?: any;
  };
}
```

### 3.2 Example Error Responses

#### Validation Error (Blurry Image)
```json
{
  "success": false,
  "error": {
    "code": "E_VAL_011",
    "category": "validation",
    "message": "Image blur score 15.2 below threshold 30",
    
    "user_message": {
      "title": "Image Out of Focus",
      "explanation": "The image appears blurry, which would significantly reduce the accuracy of lesion analysis. Sharp, focused images are essential for reliable skin cancer screening.",
      "action": "Please retake the photo with the lesion in sharp focus."
    },
    
    "recoverable": true,
    "retry_recommended": true,
    
    "image_guidance": {
      "issue": "blur",
      "tips": [
        "Hold your phone or camera steady, or use a tripod",
        "Tap on the lesion to focus before taking the photo",
        "Use your camera's macro mode if available",
        "Ensure adequate lighting so the camera can focus",
        "Take multiple photos and select the sharpest one"
      ]
    },
    
    "details": {
      "stage": "validation",
      "timestamp": "2026-01-20T10:30:00Z",
      "request_id": "derm_abc123"
    }
  }
}
```

#### Segmentation Error (No Lesion)
```json
{
  "success": false,
  "error": {
    "code": "E_SEG_001",
    "category": "segmentation",
    "message": "No lesion detected with confidence > 0.5",
    
    "user_message": {
      "title": "No Lesion Detected",
      "explanation": "Our AI could not identify a distinct lesion in your image. This may happen if the lesion is very small, blends with surrounding skin, or the image quality is insufficient.",
      "action": "Please ensure the lesion is clearly visible and centered in the image."
    },
    
    "recoverable": true,
    "retry_recommended": true,
    
    "image_guidance": {
      "issue": "no_lesion",
      "tips": [
        "Center the lesion in the frame",
        "Get closer - the lesion should fill about 30-50% of the image",
        "Ensure good contrast between the lesion and surrounding skin",
        "Use even lighting without shadows on the lesion",
        "If the lesion is very small, consider using a dermatoscope attachment"
      ]
    }
  },
  
  "partial_results": {
    "stages_completed": ["validation", "preprocessing"],
    "last_successful_output": {
      "image_quality": 0.75,
      "preprocessing_confidence": 0.82
    }
  }
}
```

#### Classification Error (Low Confidence)
```json
{
  "success": false,
  "error": {
    "code": "E_CLS_003",
    "category": "classification",
    "message": "Classification confidence 0.35 below threshold 0.50",
    
    "user_message": {
      "title": "Uncertain Analysis",
      "explanation": "Our AI analysis has low confidence in its assessment. This could be due to unusual lesion characteristics or image quality issues. For safety, we recommend professional evaluation.",
      "action": "Please consult a dermatologist for in-person examination."
    },
    
    "recoverable": false,
    "retry_recommended": false,
    
    "details": {
      "stage": "classification",
      "timestamp": "2026-01-20T10:32:00Z",
      "request_id": "derm_abc123"
    }
  },
  
  "partial_results": {
    "stages_completed": ["validation", "preprocessing", "segmentation", "classification"],
    "last_successful_output": {
      "segmentation_confidence": 0.78,
      "melanoma_probability": 0.42,
      "classification_confidence": 0.35
    }
  }
}
```

---

## 4. Error Propagation

### 4.1 Propagation Flow

```python
class ErrorPropagator:
    """
    Handles error propagation through pipeline stages.
    """
    
    def __init__(self):
        self.error_chain: List[PipelineError] = []
    
    def capture(self, error: Exception, stage: str, context: dict = None):
        """Capture and classify an error."""
        pipeline_error = PipelineError(
            original_error=error,
            stage=stage,
            timestamp=datetime.utcnow(),
            context=context or {}
        )
        
        # Classify error
        pipeline_error.code = self._classify_error(error, stage)
        pipeline_error.category = self._get_category(pipeline_error.code)
        pipeline_error.recoverable = self._is_recoverable(pipeline_error.code)
        
        self.error_chain.append(pipeline_error)
        
        # Log error
        logger.error(
            f"[{stage}] {pipeline_error.code}: {str(error)}",
            extra={"request_id": context.get("request_id")}
        )
        
        return pipeline_error
    
    def should_continue(self, error: PipelineError) -> bool:
        """Determine if pipeline should continue after error."""
        # Critical errors stop pipeline
        if error.code.startswith("E_VAL_02"):  # Content validation
            return False
        if error.code.startswith("E_SEG_001"):  # No lesion
            return False
        if error.code.startswith("E_SYS"):  # System errors
            return False
        
        # Warning-level errors allow continuation
        if error.code.endswith("W"):  # Warning suffix
            return True
        
        # Default: stop
        return False
    
    def get_user_response(self) -> ErrorResponse:
        """Generate user-facing error response."""
        if len(self.error_chain) == 0:
            return None
        
        # Use most severe error
        primary_error = self._get_primary_error()
        
        return ErrorResponse(
            success=False,
            error=self._format_error(primary_error),
            partial_results=self._get_partial_results()
        )
    
    def _classify_error(self, error: Exception, stage: str) -> str:
        """Map exception to error code."""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Validation errors
        if stage == "validation":
            if "blur" in error_msg:
                return "E_VAL_011"
            elif "resolution" in error_msg:
                return "E_VAL_010"
            elif "file" in error_msg:
                return "E_VAL_001"
            elif "skin" in error_msg:
                return "E_VAL_020"
        
        # Segmentation errors
        if stage == "segmentation":
            if "no lesion" in error_msg or "not detected" in error_msg:
                return "E_SEG_001"
            elif "confidence" in error_msg:
                return "E_SEG_002"
        
        # Classification errors
        if stage == "classification":
            if "cuda" in error_msg or "gpu" in error_msg:
                return "E_CLS_004"
            elif "timeout" in error_msg:
                return "E_TMO_003"
        
        # Default system error
        return "E_SYS_001"
```

### 4.2 Error Aggregation

```python
class ErrorAggregator:
    """
    Aggregates multiple errors into a coherent response.
    """
    
    def aggregate(self, errors: List[PipelineError]) -> AggregatedError:
        """Aggregate multiple errors."""
        if len(errors) == 0:
            return None
        
        if len(errors) == 1:
            return AggregatedError(
                primary=errors[0],
                secondary=[],
                total_count=1
            )
        
        # Sort by severity
        sorted_errors = sorted(
            errors, 
            key=lambda e: self._severity_score(e),
            reverse=True
        )
        
        primary = sorted_errors[0]
        secondary = sorted_errors[1:3]  # Up to 2 secondary errors
        
        return AggregatedError(
            primary=primary,
            secondary=secondary,
            total_count=len(errors),
            categories=list(set(e.category for e in errors)),
            all_recoverable=all(e.recoverable for e in errors)
        )
    
    def _severity_score(self, error: PipelineError) -> int:
        """Calculate error severity score."""
        # Higher score = more severe
        severity = {
            "validation": 100,      # User can fix
            "segmentation": 80,
            "classification": 70,
            "scoring": 60,
            "explanation": 40,      # Non-critical
            "timeout": 50,
            "system": 90            # Infrastructure issue
        }
        
        base_score = severity.get(error.category, 50)
        
        # Adjust for recoverability
        if not error.recoverable:
            base_score += 20
        
        return base_score
```

---

## 5. Human-Readable Messages

### 5.1 Message Templates

```python
ERROR_MESSAGES = {
    "E_VAL_011": {
        "title": "Image Out of Focus",
        "explanation": (
            "The image appears blurry, which would significantly reduce the "
            "accuracy of lesion analysis. Sharp, focused images are essential "
            "for reliable skin cancer screening."
        ),
        "action": "Please retake the photo with the lesion in sharp focus.",
        "tips": [
            "Hold your phone or camera steady, or use a tripod",
            "Tap on the lesion to focus before taking the photo",
            "Use your camera's macro mode if available",
            "Ensure adequate lighting so the camera can focus"
        ]
    },
    
    "E_VAL_020": {
        "title": "No Skin Detected",
        "explanation": (
            "This image does not appear to contain skin. Our analysis is "
            "designed specifically for skin lesion images."
        ),
        "action": "Please upload a close-up photograph of the skin lesion.",
        "tips": [
            "Ensure the skin area is clearly visible",
            "The lesion should be the main focus of the image",
            "Avoid images of clothing, objects, or non-skin areas"
        ]
    },
    
    "E_SEG_001": {
        "title": "No Lesion Detected",
        "explanation": (
            "Our AI could not identify a distinct lesion in your image. "
            "This may happen if the lesion is very small, has low contrast "
            "with surrounding skin, or is partially hidden."
        ),
        "action": "Please ensure the lesion is clearly visible and centered.",
        "tips": [
            "Center the lesion in the frame",
            "Get closer - the lesion should be clearly visible",
            "Ensure good contrast between lesion and surrounding skin",
            "Use even lighting without shadows on the lesion"
        ]
    },
    
    "E_CLS_003": {
        "title": "Uncertain Analysis",
        "explanation": (
            "Our AI analysis has low confidence in its assessment of this "
            "lesion. This could be due to unusual characteristics or image "
            "quality. For your safety, professional evaluation is recommended."
        ),
        "action": "Please consult a dermatologist for in-person examination.",
        "tips": [
            "Do not delay seeing a doctor based on this result",
            "Bring a printed copy of any images to your appointment",
            "Note any changes you've observed in the lesion"
        ]
    }
}
```

### 5.2 Dynamic Message Generation

```python
class MessageGenerator:
    """
    Generates user-friendly error messages.
    """
    
    def generate(
        self, 
        error_code: str, 
        context: dict = None
    ) -> UserMessage:
        """Generate user message for error."""
        template = ERROR_MESSAGES.get(error_code)
        
        if template is None:
            return self._default_message(error_code)
        
        message = UserMessage(
            title=template["title"],
            explanation=template["explanation"],
            action=template["action"],
            tips=template.get("tips", [])
        )
        
        # Customize based on context
        if context:
            message = self._customize_message(message, context)
        
        return message
    
    def _customize_message(
        self, 
        message: UserMessage, 
        context: dict
    ) -> UserMessage:
        """Customize message based on context."""
        # Add specific values if available
        if "blur_score" in context:
            message.explanation += (
                f" Your image blur score was {context['blur_score']:.1f}, "
                f"but we need at least {context.get('threshold', 30)}."
            )
        
        if "quality_score" in context:
            message.explanation += (
                f" Overall image quality: {context['quality_score']*100:.0f}%."
            )
        
        return message
    
    def _default_message(self, error_code: str) -> UserMessage:
        """Generate default message for unknown errors."""
        return UserMessage(
            title="Processing Error",
            explanation=(
                "An error occurred while analyzing your image. "
                "This may be a temporary issue."
            ),
            action="Please try again. If the problem persists, contact support.",
            tips=[
                "Try uploading a different image",
                "Check your internet connection",
                "Try again in a few minutes"
            ]
        )
```

---

## 6. Frontend-Compatible Response Format

### 6.1 Standardized Response

```typescript
// TypeScript interface for frontend
interface DermatologyErrorResponse {
  success: false;
  
  error: {
    // Machine-readable
    code: string;
    category: ErrorCategory;
    recoverable: boolean;
    
    // Human-readable
    title: string;
    message: string;
    action: string;
    
    // UI helpers
    icon?: "warning" | "error" | "info";
    color?: "red" | "yellow" | "blue";
    showRetry?: boolean;
    
    // Guidance
    tips?: string[];
    exampleImageUrl?: string;
  };
  
  // Processing info
  metadata: {
    requestId: string;
    timestamp: string;
    processingTimeMs: number;
    stagesCompleted: string[];
  };
}
```

### 6.2 Response Builder

```python
class FrontendResponseBuilder:
    """
    Builds frontend-compatible error responses.
    """
    
    def build(
        self, 
        error: PipelineError, 
        tracker: StageTracker
    ) -> dict:
        """Build frontend response."""
        user_message = MessageGenerator().generate(
            error.code, 
            error.context
        )
        
        return {
            "success": False,
            
            "error": {
                "code": error.code,
                "category": error.category,
                "recoverable": error.recoverable,
                
                "title": user_message.title,
                "message": user_message.explanation,
                "action": user_message.action,
                
                "icon": self._get_icon(error),
                "color": self._get_color(error),
                "showRetry": error.recoverable,
                
                "tips": user_message.tips,
                "exampleImageUrl": self._get_example_url(error.code)
            },
            
            "metadata": {
                "requestId": error.context.get("request_id"),
                "timestamp": error.timestamp.isoformat(),
                "processingTimeMs": tracker.get_summary()["total_duration_ms"],
                "stagesCompleted": [
                    s.stage for s in tracker.stages.values() 
                    if s.status == "success"
                ]
            }
        }
    
    def _get_icon(self, error: PipelineError) -> str:
        """Get appropriate icon for error."""
        if error.category in ["validation", "preprocessing"]:
            return "warning"
        elif error.category in ["system", "timeout"]:
            return "error"
        else:
            return "info"
    
    def _get_color(self, error: PipelineError) -> str:
        """Get color for error display."""
        if error.recoverable:
            return "yellow"
        else:
            return "red"
```
