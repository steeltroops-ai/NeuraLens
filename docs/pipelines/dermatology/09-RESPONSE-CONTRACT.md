# 09 - Frontend Response Contract

## Purpose
Define the complete API response contracts for successful and failed dermatology analyses.

---

## 1. Success Response Schema

### 1.1 Complete Success Response

```typescript
interface DermatologySuccessResponse {
  success: true;
  
  // Receipt confirmation
  receipt: {
    acknowledged: true;
    request_id: string;
    timestamp: string;                    // ISO 8601
    image_hash: string;                   // SHA256
    processing_time_ms: number;
  };
  
  // Processing stages
  stages: {
    completed: StageInfo[];
    total_stages: number;
    total_duration_ms: number;
  };
  
  // Lesion detection
  lesion: {
    detected: true;
    confidence: number;                   // 0-1
    bounding_box: BoundingBox;
    center: Coordinate;
    
    // Geometry
    geometry: {
      area_mm2: number;
      diameter_mm: number;
      major_axis_mm: number;
      minor_axis_mm: number;
      circularity: number;                // 0-1
      asymmetry_index: number;            // 0-1
    };
    
    // Segmentation mask (optional)
    mask_base64?: string;                 // PNG, base64
  };
  
  // Risk classification
  risk: {
    tier: RiskTier;                       // 1-5
    tier_name: string;                    // "CRITICAL" | "HIGH" | etc.
    score: number;                        // 0-100
    
    // Urgency
    urgency: string;                      // "24-48 hours" | "1-2 weeks" | etc.
    action: string;                       // Recommended action
    
    // Escalations
    escalations: Escalation[];
    requires_urgent_action: boolean;
  };
  
  // Classification results
  classification: {
    // Melanoma
    melanoma: {
      probability: number;                // 0-1
      classification: MelanomaClass;
      confidence: number;
    };
    
    // Benign/Malignant
    malignancy: {
      classification: "benign" | "malignant";
      benign_probability: number;
      malignant_probability: number;
      confidence: number;
    };
    
    // Subtype
    subtype: {
      primary: string;                    // Most likely type
      primary_probability: number;
      all_predictions: SubtypePrediction[];
    };
  };
  
  // ABCDE Analysis
  abcde: {
    score: number;                        // 0-1
    criteria_met: number;                 // 0-5
    
    asymmetry: ABCDEScore;
    border: ABCDEScore;
    color: ABCDEScore;
    diameter: ABCDEScore;
    evolution: ABCDEScore;
  };
  
  // Visualizations (optional)
  visualizations?: {
    heatmap_base64?: string;              // Grad-CAM heatmap
    overlay_base64?: string;              // Annotated image
    segmentation_overlay_base64?: string; // Lesion boundary overlay
  };
  
  // AI Explanation (optional)
  explanation?: {
    summary: string;
    detailed: string;
    recommendations: string[];
    disclaimers: string[];
  };
  
  // Longitudinal comparison (if prior exists)
  longitudinal?: LongitudinalComparison;
  
  // Confidence and quality
  quality: {
    image_quality: number;                // 0-1
    analysis_confidence: number;          // 0-1
    model_agreement: number;              // 0-1 (ensemble)
    warnings: string[];
  };
  
  // Metadata
  metadata: {
    model_versions: {
      segmentation: string;
      classification: string;
      ensemble: string;
    };
    pipeline_version: string;
    timestamp: string;
  };
}
```

### 1.2 Supporting Types

```typescript
interface StageInfo {
  name: string;
  status: "success" | "warning" | "skipped";
  duration_ms: number;
  confidence?: number;
  warnings?: string[];
}

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface Coordinate {
  x: number;
  y: number;
}

type RiskTier = 1 | 2 | 3 | 4 | 5;

type MelanomaClass = 
  | "unlikely"
  | "low_suspicion"
  | "moderate_suspicion"
  | "high_suspicion";

interface SubtypePrediction {
  subtype: string;
  probability: number;
  is_malignant: boolean;
}

interface ABCDEScore {
  score: number;                          // 0-1
  is_concerning: boolean;
  classification: string;                 // "normal" | "mild" | "moderate" | "severe"
  details?: string;
}

interface Escalation {
  rule: string;
  action: string;
  reason: string;
  priority: number;                       // 1=highest
}

interface LongitudinalComparison {
  prior_analysis_id: string;
  time_delta_days: number;
  
  changes: {
    size: {
      grew: boolean;
      percentage_change: number;
    };
    color: {
      changed: boolean;
      new_colors: string[];
    };
    shape: {
      changed: boolean;
      became_more_irregular: boolean;
    };
  };
  
  growth_rate?: {
    monthly_mm: number;
    doubling_time_days?: number;
    risk_level: string;
  };
  
  overall_change: "stable" | "notable" | "concerning" | "significant";
}
```

---

## 2. Failure Response Schema

### 2.1 Complete Failure Response

```typescript
interface DermatologyFailureResponse {
  success: false;
  
  // Error details
  error: {
    code: string;                         // e.g., "E_VAL_011"
    category: ErrorCategory;
    
    // User-facing
    title: string;
    message: string;
    action: string;
    
    // Guidance
    tips?: string[];
    example_image_url?: string;
    
    // Recovery
    recoverable: boolean;
    retry_recommended: boolean;
  };
  
  // Failure context
  failure: {
    stage: string;                        // Stage that failed
    timestamp: string;
    request_id: string;
  };
  
  // Partial results (if any)
  partial_results?: {
    stages_completed: StageInfo[];
    
    // Any successful outputs
    validation?: ValidationResult;
    preprocessing?: PreprocessingResult;
    lesion?: LesionDetection;
    
    // Last known good state
    last_confidence?: number;
  };
  
  // Image retake guidance (if applicable)
  retake_guidance?: {
    recommended: boolean;
    issue_type: string;
    tips: string[];
    quality_requirements: {
      min_resolution: string;
      focus_requirement: string;
      lighting: string;
    };
  };
  
  // Metadata
  metadata: {
    request_id: string;
    timestamp: string;
    processing_time_ms: number;
  };
}

type ErrorCategory = 
  | "validation"
  | "preprocessing"
  | "segmentation"
  | "classification"
  | "scoring"
  | "explanation"
  | "system"
  | "timeout";
```

---

## 3. Example Responses

### 3.1 Complete Success Response

```json
{
  "success": true,
  
  "receipt": {
    "acknowledged": true,
    "request_id": "derm_20260120_abc123",
    "timestamp": "2026-01-20T10:30:00Z",
    "image_hash": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "processing_time_ms": 3450
  },
  
  "stages": {
    "completed": [
      {"name": "validation", "status": "success", "duration_ms": 120, "confidence": 0.95},
      {"name": "preprocessing", "status": "success", "duration_ms": 450, "confidence": 0.88},
      {"name": "segmentation", "status": "success", "duration_ms": 890, "confidence": 0.92},
      {"name": "feature_extraction", "status": "success", "duration_ms": 320},
      {"name": "classification", "status": "success", "duration_ms": 750, "confidence": 0.87},
      {"name": "scoring", "status": "success", "duration_ms": 180},
      {"name": "explanation", "status": "success", "duration_ms": 740}
    ],
    "total_stages": 7,
    "total_duration_ms": 3450
  },
  
  "lesion": {
    "detected": true,
    "confidence": 0.92,
    "bounding_box": {
      "x": 156,
      "y": 203,
      "width": 245,
      "height": 238
    },
    "center": {"x": 278, "y": 322},
    
    "geometry": {
      "area_mm2": 42.5,
      "diameter_mm": 7.3,
      "major_axis_mm": 8.1,
      "minor_axis_mm": 6.5,
      "circularity": 0.72,
      "asymmetry_index": 0.38
    },
    
    "mask_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
  },
  
  "risk": {
    "tier": 3,
    "tier_name": "MODERATE",
    "score": 45.2,
    
    "urgency": "1-3 months",
    "action": "Dermatology consultation recommended",
    
    "escalations": [
      {
        "rule": "multiple_concerning_features",
        "action": "priority_referral",
        "reason": "3 of 5 ABCDE criteria met",
        "priority": 2
      }
    ],
    "requires_urgent_action": false
  },
  
  "classification": {
    "melanoma": {
      "probability": 0.32,
      "classification": "low_suspicion",
      "confidence": 0.78
    },
    
    "malignancy": {
      "classification": "benign",
      "benign_probability": 0.68,
      "malignant_probability": 0.32,
      "confidence": 0.72
    },
    
    "subtype": {
      "primary": "dysplastic_nevus",
      "primary_probability": 0.45,
      "all_predictions": [
        {"subtype": "dysplastic_nevus", "probability": 0.45, "is_malignant": false},
        {"subtype": "melanoma", "probability": 0.25, "is_malignant": true},
        {"subtype": "benign_nevus", "probability": 0.20, "is_malignant": false}
      ]
    }
  },
  
  "abcde": {
    "score": 0.52,
    "criteria_met": 3,
    
    "asymmetry": {
      "score": 0.45,
      "is_concerning": true,
      "classification": "moderately_asymmetric",
      "details": "Asymmetry detected along horizontal axis"
    },
    "border": {
      "score": 0.62,
      "is_concerning": true,
      "classification": "moderately_irregular",
      "details": "Irregular border with 3 notches detected"
    },
    "color": {
      "score": 0.55,
      "is_concerning": true,
      "classification": "multiple_colors",
      "details": "4 distinct colors detected: brown, tan, black, red"
    },
    "diameter": {
      "score": 0.30,
      "is_concerning": true,
      "classification": "concerning",
      "details": "Diameter 7.3mm exceeds 6mm threshold"
    },
    "evolution": {
      "score": 0.20,
      "is_concerning": false,
      "classification": "stable",
      "details": "No prior image available for comparison"
    }
  },
  
  "visualizations": {
    "heatmap_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "overlay_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "segmentation_overlay_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
  },
  
  "explanation": {
    "summary": "This lesion shows some features that warrant professional evaluation, though the overall risk appears moderate.",
    
    "detailed": "## Analysis Summary\n\nYour skin lesion has been analyzed using our AI dermatology screening system. Here's what we found:\n\n### Key Findings\n\n**Asymmetry**: The lesion shows moderate asymmetry, meaning one half does not mirror the other. This is one of the features dermatologists look for during skin checks.\n\n**Border**: The border appears somewhat irregular with a few notched areas. While many benign moles can have irregular borders, this finding adds to the overall assessment.\n\n**Color**: We detected 4 different color tones within the lesion (brown, tan, black, and hints of red). Multiple colors can sometimes indicate abnormal cell growth.\n\n**Diameter**: At 7.3mm, this lesion exceeds the typical 6mm guideline often used in screening. Larger lesions warrant closer attention.\n\n### What This Means\n\nBased on our analysis, this lesion has features that merit evaluation by a dermatologist. While many lesions with these characteristics turn out to be benign (non-cancerous), professional examination can provide definitive assessment.\n\n### Risk Level: MODERATE\n\nWe recommend scheduling a dermatology appointment within the next 1-3 months for professional evaluation.",
    
    "recommendations": [
      "Schedule a dermatology appointment within 1-3 months",
      "Take photos monthly to track any changes",
      "Note any new symptoms (itching, bleeding, crusting)",
      "Bring these results to your appointment"
    ],
    
    "disclaimers": [
      "This AI analysis is for screening purposes only and is NOT a medical diagnosis.",
      "Only a qualified dermatologist can diagnose skin conditions after proper examination.",
      "If you notice rapid changes, bleeding, or other concerning symptoms, seek immediate medical attention.",
      "This tool is intended to support, not replace, professional medical judgment."
    ]
  },
  
  "quality": {
    "image_quality": 0.88,
    "analysis_confidence": 0.85,
    "model_agreement": 0.82,
    "warnings": [
      "Minor hair occlusion detected - analysis may be slightly affected"
    ]
  },
  
  "metadata": {
    "model_versions": {
      "segmentation": "unet_effb4_v2.1.0",
      "classification": "effnet_isic_v3.0.2",
      "ensemble": "derm_ensemble_v1.5.0"
    },
    "pipeline_version": "1.0.0",
    "timestamp": "2026-01-20T10:30:03.450Z"
  }
}
```

### 3.2 Validation Failure Response

```json
{
  "success": false,
  
  "error": {
    "code": "E_VAL_011",
    "category": "validation",
    
    "title": "Image Out of Focus",
    "message": "The image appears blurry, which would significantly reduce the accuracy of lesion analysis. Sharp, focused images are essential for reliable skin cancer screening.",
    "action": "Please retake the photo with the lesion in sharp focus.",
    
    "tips": [
      "Hold your phone or camera steady, or use a tripod",
      "Tap on the lesion to focus before taking the photo",
      "Use your camera's macro mode if available",
      "Ensure adequate lighting so the camera can focus",
      "Take multiple photos and select the sharpest one"
    ],
    
    "recoverable": true,
    "retry_recommended": true
  },
  
  "failure": {
    "stage": "validation",
    "timestamp": "2026-01-20T10:30:00.120Z",
    "request_id": "derm_20260120_def456"
  },
  
  "partial_results": {
    "stages_completed": [
      {"name": "file_validation", "status": "success", "duration_ms": 45}
    ],
    
    "validation": {
      "file_valid": true,
      "resolution_valid": true,
      "focus_valid": false,
      "focus_score": 18.5,
      "required_focus_score": 30.0
    }
  },
  
  "retake_guidance": {
    "recommended": true,
    "issue_type": "blur",
    "tips": [
      "Ensure the lesion is in sharp focus before taking the photo",
      "Use macro/close-up mode on your camera",
      "Hold the camera steady or use a surface to stabilize",
      "Good lighting helps the camera focus better"
    ],
    "quality_requirements": {
      "min_resolution": "1920x1080 (2MP)",
      "focus_requirement": "Lesion details clearly visible",
      "lighting": "Even, diffuse lighting without harsh shadows"
    }
  },
  
  "metadata": {
    "request_id": "derm_20260120_def456",
    "timestamp": "2026-01-20T10:30:00.120Z",
    "processing_time_ms": 120
  }
}
```

### 3.3 Partial Success Response

```json
{
  "success": true,
  
  "receipt": {
    "acknowledged": true,
    "request_id": "derm_20260120_ghi789",
    "timestamp": "2026-01-20T10:35:00Z",
    "image_hash": "sha256:...",
    "processing_time_ms": 4200
  },
  
  "stages": {
    "completed": [
      {"name": "validation", "status": "success", "duration_ms": 115},
      {"name": "preprocessing", "status": "warning", "duration_ms": 520, "warnings": ["Hair removal had limited effect"]},
      {"name": "segmentation", "status": "warning", "duration_ms": 950, "confidence": 0.65, "warnings": ["Using bounding box due to low segmentation confidence"]},
      {"name": "feature_extraction", "status": "success", "duration_ms": 380},
      {"name": "classification", "status": "success", "duration_ms": 820, "confidence": 0.72},
      {"name": "scoring", "status": "success", "duration_ms": 190},
      {"name": "explanation", "status": "skipped", "duration_ms": 0}
    ],
    "total_stages": 7,
    "total_duration_ms": 4200
  },
  
  "lesion": {
    "detected": true,
    "confidence": 0.65,
    "bounding_box": {"x": 180, "y": 220, "width": 200, "height": 195},
    "center": {"x": 280, "y": 317},
    
    "geometry": {
      "area_mm2": 38.2,
      "diameter_mm": 6.9,
      "major_axis_mm": 7.5,
      "minor_axis_mm": 6.1,
      "circularity": 0.78,
      "asymmetry_index": 0.28
    }
  },
  
  "risk": {
    "tier": 4,
    "tier_name": "LOW",
    "score": 28.5,
    "urgency": "Routine",
    "action": "Monitor and re-evaluate in 6 months",
    "escalations": [],
    "requires_urgent_action": false
  },
  
  "classification": {
    "melanoma": {
      "probability": 0.15,
      "classification": "unlikely",
      "confidence": 0.72
    },
    "malignancy": {
      "classification": "benign",
      "benign_probability": 0.85,
      "malignant_probability": 0.15,
      "confidence": 0.70
    },
    "subtype": {
      "primary": "benign_nevus",
      "primary_probability": 0.62,
      "all_predictions": [
        {"subtype": "benign_nevus", "probability": 0.62, "is_malignant": false},
        {"subtype": "melanoma", "probability": 0.15, "is_malignant": true}
      ]
    }
  },
  
  "abcde": {
    "score": 0.25,
    "criteria_met": 1,
    "asymmetry": {"score": 0.28, "is_concerning": false, "classification": "mildly_asymmetric"},
    "border": {"score": 0.22, "is_concerning": false, "classification": "slightly_irregular"},
    "color": {"score": 0.18, "is_concerning": false, "classification": "uniform"},
    "diameter": {"score": 0.35, "is_concerning": true, "classification": "concerning"},
    "evolution": {"score": 0.15, "is_concerning": false, "classification": "stable"}
  },
  
  "quality": {
    "image_quality": 0.68,
    "analysis_confidence": 0.70,
    "model_agreement": 0.75,
    "warnings": [
      "Hair partially obscures lesion - consider retaking after hair removal",
      "Segmentation confidence lower than optimal",
      "AI explanation skipped due to processing constraints"
    ]
  },
  
  "metadata": {
    "model_versions": {
      "segmentation": "unet_effb4_v2.1.0",
      "classification": "effnet_isic_v3.0.2",
      "ensemble": "derm_ensemble_v1.5.0"
    },
    "pipeline_version": "1.0.0",
    "timestamp": "2026-01-20T10:35:04.200Z"
  }
}
```

---

## 4. Response Guidelines

### 4.1 Required Fields

| Field | Always Present | Notes |
|-------|----------------|-------|
| `success` | Yes | Boolean indicator |
| `receipt.request_id` | Yes | For tracking |
| `receipt.timestamp` | Yes | For logging |
| `stages` | Yes | Processing info |
| `risk.tier` | On success | Risk classification |
| `classification.melanoma` | On success | Primary result |
| `quality.warnings` | Yes | May be empty |
| `error` | On failure | Error details |

### 4.2 Optional Fields

| Field | Condition | Notes |
|-------|-----------|-------|
| `visualizations` | If requested | Heatmaps, overlays |
| `explanation` | If requested & succeeded | AI text |
| `longitudinal` | If prior provided | Comparison |
| `lesion.mask_base64` | If segmentation high confidence | Binary mask |
| `partial_results` | On failure after partial processing | Salvaged data |

### 4.3 Image Encoding

All images returned as **base64-encoded PNG**:
- Heatmaps: 512x512, RGBA (transparency for overlay)
- Overlays: Original resolution, RGB
- Masks: Original resolution, grayscale (0 or 255)

```javascript
// Frontend usage
const img = new Image();
img.src = `data:image/png;base64,${response.visualizations.heatmap_base64}`;
```
