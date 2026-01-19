# 09 - Frontend Response Contract

## Document Info
| Field | Value |
|-------|-------|
| Stage | Response Formatting |
| Owner | All Team Members |
| Reviewer | Radiologist |

---

## 1. Success Response Schema

### 1.1 Complete Success Response
```json
{
  "success": true,
  "request_id": "req_rad_abc123def456",
  "timestamp": "2026-01-19T10:30:00.000Z",
  "processing_time_ms": 2450,
  
  "receipt": {
    "acknowledged": true,
    "modality_received": "chest_xray",
    "body_region": "chest",
    "is_volumetric": false,
    "file_hash": "sha256:abc123...",
    "file_size_mb": 2.4
  },
  
  "stages_completed": [
    {"stage": "RECEIPT", "status": "success", "time_ms": 5},
    {"stage": "VALIDATION", "status": "success", "time_ms": 120},
    {"stage": "PREPROCESSING", "status": "success", "time_ms": 450},
    {"stage": "DETECTION", "status": "success", "time_ms": 380},
    {"stage": "ANALYSIS", "status": "success", "time_ms": 980},
    {"stage": "AGGREGATION", "status": "success", "time_ms": 45},
    {"stage": "SCORING", "status": "success", "time_ms": 70},
    {"stage": "FORMATTING", "status": "success", "time_ms": 400}
  ],
  
  "clinical_results": {
    "modality_processed": "chest_xray",
    
    "primary_finding": {
      "condition": "Pneumonia",
      "probability": 72.5,
      "severity": "high",
      "description": "Right lower lobe consolidation consistent with pneumonia"
    },
    
    "all_predictions": {
      "Atelectasis": 12.3,
      "Consolidation": 68.2,
      "Infiltration": 45.6,
      "Pneumothorax": 2.1,
      "Edema": 8.5,
      "Emphysema": 3.2,
      "Fibrosis": 4.8,
      "Effusion": 35.4,
      "Pneumonia": 72.5,
      "Pleural_Thickening": 5.1,
      "Cardiomegaly": 18.2,
      "Nodule": 6.3,
      "Mass": 2.8,
      "Hernia": 0.5,
      "Lung Lesion": 4.4,
      "Fracture": 1.2,
      "Lung Opacity": 52.5,
      "Enlarged Cardiomediastinum": 12.8
    },
    
    "findings": [
      {
        "id": "finding_001",
        "condition": "Pneumonia",
        "probability": 72.5,
        "severity": "high",
        "confidence": 0.85,
        "location": "Right lower lobe",
        "description": "Consolidation pattern consistent with pneumonia",
        "radiological_features": ["Consolidation", "Air bronchograms"],
        "is_critical": true
      },
      {
        "id": "finding_002",
        "condition": "Pleural Effusion",
        "probability": 35.4,
        "severity": "moderate",
        "confidence": 0.78,
        "location": "Right costophrenic angle",
        "description": "Small right pleural effusion",
        "radiological_features": ["Costophrenic blunting", "Meniscus sign"],
        "is_critical": false
      }
    ],
    
    "anatomical_findings": {
      "lungs": {
        "left_lung": {"status": "normal", "confidence": 0.92},
        "right_lung": {"status": "abnormal", "confidence": 0.88, "findings": ["consolidation"]}
      },
      "heart": {
        "cardiothoracic_ratio": 0.48,
        "status": "normal",
        "cardiomegaly": false
      },
      "mediastinum": {
        "status": "normal",
        "width_normal": true
      }
    },
    
    "risk_assessment": {
      "risk_score": 55.2,
      "risk_category": "high",
      "risk_color": "orange",
      "urgency": "priority",
      "confidence": 0.84
    },
    
    "recommendations": [
      "Clinical correlation recommended for right lower lobe findings",
      "Consider antibiotic therapy if clinically indicated",
      "Follow-up imaging in 4-6 weeks to document resolution",
      "Evaluate for possible associated pleural effusion"
    ]
  },
  
  "quality_assessment": {
    "overall_quality": "good",
    "quality_score": 0.88,
    "positioning": "adequate",
    "exposure": "satisfactory",
    "inspiration": "adequate",
    "rotation": "minimal",
    "limitations": []
  },
  
  "visualizations": {
    "heatmap": {
      "available": true,
      "format": "base64_png",
      "data": "iVBORw0KGgoAAAANSUhEUgAA...",
      "target_condition": "Pneumonia"
    },
    "overlay": {
      "available": true,
      "format": "base64_png",
      "data": "iVBORw0KGgoAAAANSUhEUgAA...",
      "opacity": 0.5
    },
    "segmentation": {
      "available": true,
      "structures": ["left_lung", "right_lung", "heart"],
      "format": "base64_png"
    }
  },
  
  "metadata": {
    "model_used": "densenet121-res224-all",
    "model_version": "1.0.0",
    "ensemble_size": 1,
    "calibration_applied": true,
    "dicom_tags_extracted": false
  }
}
```

---

## 2. Failure Response Schema

### 2.1 Complete Failure Response
```json
{
  "success": false,
  "request_id": "req_rad_abc123def456",
  "timestamp": "2026-01-19T10:30:00.000Z",
  "processing_time_ms": 150,
  
  "error": {
    "code": "E_GEN_007",
    "message": "Image does not appear to be a medical radiological image",
    "stage": "VALIDATION",
    "stage_index": 1,
    
    "user_message": {
      "title": "Non-Medical Image Detected",
      "explanation": "The uploaded image doesn't appear to be a medical radiological image such as an X-ray, CT scan, or MRI.",
      "action": "Please upload a valid medical image for analysis."
    },
    
    "technical_details": {
      "detected_content_type": "photograph",
      "medical_confidence": 0.12,
      "expected_types": ["chest_xray", "ct_scan", "mri"]
    },
    
    "recoverable": false,
    "resubmission_hint": "Upload a chest X-ray, CT, or MRI image in JPEG, PNG, or DICOM format"
  },
  
  "stages_completed": [
    {"stage": "RECEIPT", "status": "success", "time_ms": 5}
  ],
  
  "stages_failed": [
    {"stage": "VALIDATION", "status": "failed", "time_ms": 145, "error_code": "E_GEN_007"}
  ]
}
```

### 2.2 Partial Success Response
```json
{
  "success": true,
  "partial": true,
  "request_id": "req_rad_abc123def456",
  "timestamp": "2026-01-19T10:30:00.000Z",
  "processing_time_ms": 2100,
  
  "warnings": [
    {
      "code": "W_DET_001",
      "message": "Lung segmentation confidence below threshold",
      "stage": "DETECTION",
      "impact": "Regional analysis may be less precise",
      "recommendation": "Results are valid but manual review recommended"
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
    "quality_disclaimer": "Results generated from suboptimal image quality. Manual review recommended."
  },
  
  "visualizations": {
    "heatmap": {
      "available": true,
      "confidence_warning": true
    }
  }
}
```

---

## 3. Field Specifications

### 3.1 Required Fields (Success)

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Always true for success |
| request_id | string | Unique request identifier |
| timestamp | ISO8601 | Response timestamp |
| processing_time_ms | integer | Total processing time |
| receipt | object | Input acknowledgment |
| stages_completed | array | List of completed stages |
| clinical_results | object | Analysis results |
| quality_assessment | object | Image quality metrics |

### 3.2 Required Fields (Failure)

| Field | Type | Description |
|-------|------|-------------|
| success | boolean | Always false for failure |
| request_id | string | Unique request identifier |
| timestamp | ISO8601 | Response timestamp |
| error | object | Error details |
| error.code | string | Machine-readable error code |
| error.message | string | Technical error message |
| error.stage | string | Stage where error occurred |
| error.user_message | object | Human-readable explanation |
| error.recoverable | boolean | Whether retry may help |

### 3.3 Optional Fields

| Field | Condition | Description |
|-------|-----------|-------------|
| visualizations | If requested | Heatmaps, overlays, segmentation |
| metadata | If DICOM | DICOM tags and model info |
| warnings | If partial success | Non-fatal issues |
| partial | If some analysis failed | Indicates incomplete results |

---

## 4. Confidence and Severity Levels

### 4.1 Confidence Reporting

| Level | Range | Interpretation | Color |
|-------|-------|----------------|-------|
| High | >= 0.85 | Reliable result | Green |
| Medium | 0.70-0.84 | Acceptable, monitor | Yellow |
| Low | 0.50-0.69 | Review recommended | Orange |
| Very Low | < 0.50 | Flagged for review | Red |

### 4.2 Severity Levels

| Severity | Description | UI Treatment |
|----------|-------------|--------------|
| critical | Immediate attention needed | Red badge, alert |
| high | Significant finding | Orange badge |
| moderate | Notable finding | Yellow badge |
| low | Minor finding | Gray badge |
| normal | No abnormality | Green badge |

### 4.3 Risk Categories

| Category | Score Range | Color | Action |
|----------|-------------|-------|--------|
| critical | 75-100 | Red | Urgent review |
| high | 50-74 | Orange | Priority follow-up |
| moderate | 25-49 | Yellow | Routine follow-up |
| low | 0-24 | Green | Routine monitoring |

---

## 5. Visualization Outputs

### 5.1 Heatmap Format
```json
{
  "heatmap": {
    "available": true,
    "format": "base64_png",
    "data": "iVBORw0KGgoAAAANSUhEUgAA...",
    "dimensions": [224, 224],
    "target_condition": "Pneumonia",
    "colormap": "jet",
    "min_value": 0.0,
    "max_value": 1.0
  }
}
```

### 5.2 Overlay Format
```json
{
  "overlay": {
    "available": true,
    "format": "base64_png",
    "data": "iVBORw0KGgoAAAANSUhEUgAA...",
    "original_dimensions": [1024, 1024],
    "overlay_dimensions": [224, 224],
    "default_opacity": 0.5
  }
}
```

### 5.3 Segmentation Mask Format
```json
{
  "segmentation": {
    "available": true,
    "structures": {
      "left_lung": {
        "mask_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
        "color": "#3b82f6",
        "area_pixels": 45230
      },
      "right_lung": {
        "mask_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
        "color": "#10b981",
        "area_pixels": 48120
      },
      "heart": {
        "mask_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
        "color": "#ef4444",
        "area_pixels": 18450
      }
    }
  }
}
```

---

## 6. Volumetric Data Response

### 6.1 CT/MRI Volume Response
```json
{
  "success": true,
  "request_id": "req_rad_vol_123",
  
  "receipt": {
    "modality_received": "ct",
    "body_region": "chest",
    "is_volumetric": true,
    "slice_count": 256,
    "volume_dimensions": [512, 512, 256],
    "spacing_mm": [0.7, 0.7, 1.0]
  },
  
  "clinical_results": {
    "volume_analysis": {
      "slices_analyzed": 256,
      "aggregation_method": "noisy_or"
    },
    
    "lesions_detected": [
      {
        "id": "lesion_001",
        "type": "nodule",
        "center_voxel": [245, 180, 120],
        "center_physical_mm": [171.5, 126.0, 120.0],
        "size_mm": [8.2, 7.5, 6.8],
        "volume_ml": 0.22,
        "probability": 0.78,
        "malignancy_risk": "low",
        "slice_range": [115, 125]
      }
    ],
    
    "organ_volumes": {
      "left_lung_ml": 2450,
      "right_lung_ml": 2680,
      "heart_ml": 520
    },
    
    "slice_findings": {
      "summary": "Findings concentrated in slices 115-140",
      "key_slices": [120, 125, 130]
    }
  },
  
  "visualizations": {
    "key_slice_heatmaps": {
      "120": "iVBORw0KGgoAAAANSUhEUgAA...",
      "125": "iVBORw0KGgoAAAANSUhEUgAA...",
      "130": "iVBORw0KGgoAAAANSUhEUgAA..."
    },
    "3d_render_available": false
  }
}
```

---

## 7. TypeScript Interface

```typescript
// Response Types
interface RadiologySuccessResponse {
  success: true;
  request_id: string;
  timestamp: string;
  processing_time_ms: number;
  
  receipt: {
    acknowledged: boolean;
    modality_received: 'chest_xray' | 'ct' | 'mri';
    body_region: string;
    is_volumetric: boolean;
    file_hash: string;
    file_size_mb: number;
  };
  
  stages_completed: StageResult[];
  clinical_results: ClinicalResults;
  quality_assessment: QualityAssessment;
  visualizations?: Visualizations;
  metadata?: ResponseMetadata;
}

interface ClinicalResults {
  modality_processed: string;
  primary_finding: Finding;
  all_predictions: Record<string, number>;
  findings: Finding[];
  anatomical_findings: AnatomicalFindings;
  risk_assessment: RiskAssessment;
  recommendations: string[];
}

interface Finding {
  id: string;
  condition: string;
  probability: number;
  severity: 'critical' | 'high' | 'moderate' | 'low' | 'normal';
  confidence: number;
  location?: string;
  description: string;
  radiological_features?: string[];
  is_critical: boolean;
}

interface RiskAssessment {
  risk_score: number;
  risk_category: 'critical' | 'high' | 'moderate' | 'low';
  risk_color: 'red' | 'orange' | 'yellow' | 'green';
  urgency: 'urgent' | 'priority' | 'routine';
  confidence: number;
}

interface RadiologyErrorResponse {
  success: false;
  request_id: string;
  timestamp: string;
  processing_time_ms: number;
  
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
  
  stages_completed: StageResult[];
  stages_failed: StageResult[];
}

type RadiologyResponse = RadiologySuccessResponse | RadiologyErrorResponse;
```

---

## 8. API Versioning

### 8.1 Version Header
```
X-API-Version: 2026-01-19
Content-Type: application/json
```

### 8.2 Response Size Limits

| Field | Max Size | Notes |
|-------|----------|-------|
| heatmap_base64 | 500 KB | Per heatmap |
| overlay_base64 | 1 MB | Per overlay |
| all visualizations | 5 MB | Total |
| Total response | 10 MB | Compressed |

---

## 9. Stage Confirmation

```json
{
  "stage_complete": "FORMATTING",
  "stage_id": 7,
  "status": "success",
  "timestamp": "2026-01-19T10:30:04.500Z",
  "summary": {
    "response_size_kb": 245,
    "visualizations_included": true,
    "heatmap_generated": true
  },
  "next_stage": "COMPLETE"
}
```
