# Retinal Pipeline - Frontend Response Contract

## Document Info
| Field | Value |
|-------|-------|
| Version | 4.0.0 |
| Pipeline Stage | 9 - API Response |

---

## 1. Success Response Schema

### 1.1 Complete Success Response
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-01-19T08:00:00Z",
  "processing_time_ms": 1850,
  
  "receipt_confirmation": {
    "image_received": true,
    "image_size_bytes": 245678,
    "image_dimensions": [1024, 1024],
    "received_at": "2026-01-19T08:00:00Z"
  },
  
  "stages_completed": [
    "input_validation",
    "image_preprocessing",
    "quality_assessment",
    "anatomical_detection",
    "biomarker_extraction",
    "dr_grading",
    "risk_calculation",
    "heatmap_generation",
    "clinical_assessment",
    "output_formatting"
  ],
  
  "stages_timing_ms": {
    "input_validation": 45,
    "image_preprocessing": 320,
    "quality_assessment": 85,
    "anatomical_detection": 450,
    "biomarker_extraction": 380,
    "dr_grading": 250,
    "risk_calculation": 15,
    "heatmap_generation": 180,
    "clinical_assessment": 95,
    "output_formatting": 30
  },
  
  "image_quality": {
    "score": 0.88,
    "grade": "good",
    "usable": true,
    "issues": [],
    "components": {
      "sharpness": 0.92,
      "illumination": 0.85,
      "snr_db": 32.5,
      "glare_percentage": 0.8
    }
  },
  
  "diabetic_retinopathy": {
    "grade": 0,
    "grade_name": "No DR",
    "probability": 0.92,
    "probabilities": {
      "no_dr": 0.92,
      "mild_npdr": 0.05,
      "moderate_npdr": 0.02,
      "severe_npdr": 0.008,
      "proliferative_dr": 0.002
    },
    "referral_urgency": "routine_12_months",
    "referral_description": "Routine screening in 12 months",
    "four_two_one_rule": {
      "hemorrhages_4_quadrants": false,
      "venous_beading_2_quadrants": false,
      "irma_1_quadrant": false,
      "severe_npdr_criteria_met": false
    }
  },
  
  "diabetic_macular_edema": {
    "present": false,
    "csme_criteria_met": false,
    "severity": "none",
    "macular_thickness_um": 285,
    "confidence": 0.88
  },
  
  "risk_assessment": {
    "overall_score": 15.5,
    "category": "low",
    "confidence": 0.91,
    "primary_finding": "No significant abnormality",
    "component_scores": {
      "dr_grade": 0,
      "cdr": 12,
      "vessel_abnormality": 8,
      "lesion_burden": 0,
      "macular_involvement": 0
    },
    "referral_urgency": "routine_12_months"
  },
  
  "biomarkers": {
    "vessels": {
      "tortuosity_index": {
        "value": 1.12,
        "normal_range": [1.0, 1.15],
        "status": "normal",
        "confidence": 0.89
      },
      "av_ratio": {
        "value": 0.68,
        "normal_range": [0.65, 0.75],
        "status": "normal",
        "confidence": 0.87
      },
      "vessel_density": {
        "value": 0.78,
        "normal_range": [0.60, 0.85],
        "status": "normal",
        "confidence": 0.91
      }
    },
    "optic_disc": {
      "cup_disc_ratio": {
        "value": 0.32,
        "normal_range": [0.1, 0.4],
        "status": "normal",
        "confidence": 0.93
      },
      "disc_area_mm2": {
        "value": 2.1,
        "normal_range": [1.5, 2.5],
        "status": "normal"
      }
    },
    "lesions": {
      "hemorrhage_count": {
        "value": 0,
        "threshold": 0,
        "status": "normal"
      },
      "microaneurysm_count": {
        "value": 0,
        "threshold": 0,
        "status": "normal"
      },
      "exudate_area_percent": {
        "value": 0.0,
        "threshold": 1.0,
        "status": "normal"
      }
    }
  },
  
  "findings": [
    {
      "type": "Normal fundus appearance",
      "location": "general",
      "severity": "normal",
      "description": "No visible retinal pathology",
      "icd10_code": null
    }
  ],
  
  "differential_diagnoses": [],
  
  "recommendations": [
    "Retinal examination appears normal",
    "Continue routine diabetic screening annually",
    "Maintain blood glucose and blood pressure control"
  ],
  
  "clinical_summary": "Fundus examination reveals a healthy retina with no evidence of diabetic retinopathy, macular edema, or glaucomatous changes. All vascular biomarkers are within normal limits. Routine annual screening is recommended.",
  
  "heatmap_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "lesion_overlay_base64": null,
  
  "model_versions": {
    "dr_classifier": "efficientnet_b4_dr_v1.0.0",
    "vessel_segmentation": "unet_vessels_v1.0.0",
    "optic_disc": "unet_od_v1.0.0"
  },
  
  "confidence_scores": {
    "overall": 0.91,
    "dr_grading": 0.92,
    "biomarkers": 0.89,
    "anatomical": 0.93
  },
  
  "warnings": []
}
```

---

## 2. Failure Response Schema

### 2.1 Complete Failure Response
```json
{
  "success": false,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-01-19T08:00:00Z",
  "processing_time_ms": 125,
  
  "receipt_confirmation": {
    "image_received": true,
    "image_size_bytes": 12456,
    "image_dimensions": [256, 256],
    "received_at": "2026-01-19T08:00:00Z"
  },
  
  "failure_stage": "input_validation",
  
  "error": {
    "code": "VAL_010",
    "category": "validation",
    "title": "Image Resolution Too Low",
    "message": "The image resolution is below the minimum required for accurate analysis.",
    "guidance": "Please upload an image with at least 512x512 pixels. 1024x1024 or higher is recommended for best results.",
    
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
  "stages_failed": ["input_validation"],
  
  "stages_timing_ms": {
    "input_validation": 125
  },
  
  "partial_results": null
}
```

---

## 3. Partial Success Response

### 3.1 Analysis with Warnings
```json
{
  "success": true,
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2026-01-19T08:00:00Z",
  
  "warnings": [
    {
      "code": "ANA_001",
      "stage": "anatomical_detection",
      "title": "Optic Disc Not Detected",
      "message": "Glaucoma-related biomarkers may be less accurate.",
      "affected_metrics": ["cup_disc_ratio", "disc_area_mm2", "rim_area_mm2"],
      "confidence_reduction": 0.3
    },
    {
      "code": "VAL_011",
      "stage": "input_validation",
      "title": "Suboptimal Resolution",
      "message": "Image resolution is acceptable but below recommended.",
      "guidance": "1024x1024 or higher is recommended for best accuracy."
    }
  ],
  
  "confidence_adjustments": {
    "overall": 0.75,
    "cup_disc_ratio": 0.5,
    "glaucoma_risk": 0.5
  },
  
  "results": {
    "diabetic_retinopathy": { "...": "normal results" },
    "biomarkers": {
      "optic_disc": {
        "cup_disc_ratio": {
          "value": null,
          "status": "unavailable",
          "reason": "Optic disc detection failed"
        }
      }
    }
  }
}
```

---

## 4. Response Field Specifications

### 4.1 Required Fields (Success)
| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Always `true` for success |
| `session_id` | string (UUID) | Unique analysis session ID |
| `timestamp` | string (ISO8601) | Response timestamp |
| `processing_time_ms` | integer | Total processing time |
| `receipt_confirmation` | object | Confirms image received |
| `stages_completed` | array[string] | List of completed stages |
| `diabetic_retinopathy` | object | DR grading results |
| `risk_assessment` | object | Overall risk score |
| `biomarkers` | object | All extracted biomarkers |
| `recommendations` | array[string] | Clinical recommendations |

### 4.2 Required Fields (Failure)
| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Always `false` for failure |
| `session_id` | string (UUID) | Unique session ID |
| `timestamp` | string (ISO8601) | Response timestamp |
| `failure_stage` | string | Stage where failure occurred |
| `error` | object | Error details |
| `error.code` | string | Machine-readable error code |
| `error.message` | string | Human-readable message |
| `error.resubmission_recommended` | boolean | Should user resubmit? |

### 4.3 Optional Fields
| Field | Type | When Included |
|-------|------|---------------|
| `heatmap_base64` | string | When heatmap generation succeeds |
| `lesion_overlay_base64` | string | When lesions are detected |
| `prior_comparison` | object | When prior image exists |
| `warnings` | array | When non-fatal issues occur |
| `partial_results` | object | When analysis partially succeeds |

---

## 5. Visual Overlay Specifications

### 5.1 Heatmap Format
```json
{
  "heatmap_base64": "data:image/png;base64,iVBORw0KGgo...",
  "heatmap_dimensions": [512, 512],
  "colormap": "jet",
  "opacity_recommended": 0.5,
  "legend": {
    "low_attention": "#0000FF",
    "medium_attention": "#00FF00",
    "high_attention": "#FF0000"
  }
}
```

### 5.2 Lesion Overlay Format
```json
{
  "lesion_overlay_base64": "data:image/png;base64,...",
  "lesion_annotations": [
    {
      "type": "microaneurysm",
      "bbox": [120, 340, 135, 355],
      "confidence": 0.87,
      "color": "#FF0000"
    },
    {
      "type": "hemorrhage",
      "bbox": [200, 180, 250, 230],
      "confidence": 0.92,
      "color": "#FFA500"
    }
  ],
  "legend": {
    "microaneurysm": "#FF0000",
    "hemorrhage": "#FFA500",
    "hard_exudate": "#FFFF00",
    "cotton_wool_spot": "#FFFFFF"
  }
}
```

---

## 6. TypeScript Interface Definitions

```typescript
// Main Response Types
interface RetinalAnalysisResponse {
  success: boolean;
  session_id: string;
  timestamp: string;
  processing_time_ms: number;
  
  receipt_confirmation: ReceiptConfirmation;
  stages_completed: PipelineStage[];
  stages_timing_ms: Record<PipelineStage, number>;
  
  // Success-only fields
  image_quality?: ImageQuality;
  diabetic_retinopathy?: DRResult;
  diabetic_macular_edema?: DMEResult;
  risk_assessment?: RiskAssessment;
  biomarkers?: CompleteBiomarkers;
  findings?: ClinicalFinding[];
  recommendations?: string[];
  clinical_summary?: string;
  heatmap_base64?: string;
  confidence_scores?: Record<string, number>;
  
  // Failure-only fields
  failure_stage?: PipelineStage;
  error?: ErrorResponse;
  
  // Both
  warnings?: Warning[];
}

interface ReceiptConfirmation {
  image_received: boolean;
  image_size_bytes: number;
  image_dimensions: [number, number];
  received_at: string;
}

interface ErrorResponse {
  code: string;
  category: string;
  title: string;
  message: string;
  guidance: string;
  stage: string;
  severity: 'fatal' | 'error' | 'warning';
  recoverable: boolean;
  resubmission_recommended: boolean;
  retry_recommended: boolean;
  contact_support: boolean;
  technical_details?: Record<string, unknown>;
}

interface Warning {
  code: string;
  stage: string;
  title: string;
  message: string;
  affected_metrics?: string[];
  guidance?: string;
}
```
