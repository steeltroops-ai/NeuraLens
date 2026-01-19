# 09 - Frontend Response Contract

## Document Info
| Field | Value |
|-------|-------|
| Stage | Response Formatting |
| Owner | All Team Members |
| Reviewer | Cardiologist |

---

## 1. Success Response Schema

### 1.1 Complete Success Response
```json
{
  "success": true,
  "request_id": "req_abc123def456",
  "timestamp": "2026-01-19T10:30:00.000Z",
  "processing_time_ms": 2450,
  
  "receipt": {
    "acknowledged": true,
    "modalities_received": ["echo_video", "ecg_signal", "clinical_metadata"],
    "file_hashes": {
      "echo_video": "sha256:abc123...",
      "ecg_signal": "sha256:def456..."
    }
  },
  
  "stages_completed": [
    {"stage": "RECEIPT", "status": "success", "time_ms": 5},
    {"stage": "VALIDATION", "status": "success", "time_ms": 120},
    {"stage": "PREPROCESSING", "status": "success", "time_ms": 800},
    {"stage": "DETECTION", "status": "success", "time_ms": 600},
    {"stage": "ANALYSIS", "status": "success", "time_ms": 450},
    {"stage": "FUSION", "status": "success", "time_ms": 50},
    {"stage": "SCORING", "status": "success", "time_ms": 25},
    {"stage": "FORMATTING", "status": "success", "time_ms": 100}
  ],
  
  "clinical_results": {
    "echo_analysis": {
      "ejection_fraction": {
        "value": 58.2,
        "unit": "%",
        "interpretation": "normal",
        "confidence": 0.89,
        "reference_range": {"min": 55, "max": 70}
      },
      "wall_motion": {
        "global_score": 1.0,
        "interpretation": "normal",
        "abnormal_segments": [],
        "confidence": 0.85
      },
      "chamber_assessment": {
        "lv_dilated": false,
        "la_dilated": false,
        "rv_dilated": false
      },
      "view_quality": {
        "primary_view": "A4C",
        "view_confidence": 0.92,
        "frames_analyzed": 450
      }
    },
    
    "ecg_analysis": {
      "rhythm_analysis": {
        "classification": "Normal Sinus Rhythm",
        "heart_rate_bpm": 72,
        "regularity": "regular",
        "confidence": 0.94,
        "r_peaks_detected": 42
      },
      "hrv_metrics": {
        "time_domain": {
          "rmssd_ms": 42.5,
          "sdnn_ms": 68.3,
          "pnn50_percent": 18.2,
          "mean_rr_ms": 833
        },
        "interpretation": {
          "autonomic_balance": "normal",
          "parasympathetic": "adequate",
          "sympathetic": "normal"
        }
      },
      "intervals": {
        "pr_interval_ms": 165,
        "qrs_duration_ms": 95,
        "qt_interval_ms": 380,
        "qtc_ms": 412,
        "all_normal": true
      },
      "arrhythmias": []
    },
    
    "findings": [
      {
        "id": "finding_001",
        "type": "observation",
        "severity": "normal",
        "title": "Normal Sinus Rhythm",
        "description": "Regular rhythm with rate 60-100 bpm",
        "source": "ecg"
      },
      {
        "id": "finding_002",
        "type": "observation",
        "severity": "normal",
        "title": "Normal Ejection Fraction",
        "description": "EF 58% indicates normal systolic function",
        "source": "echo"
      }
    ],
    
    "risk_assessment": {
      "risk_score": 12.5,
      "risk_category": "low",
      "risk_factors": [],
      "confidence": 0.91
    },
    
    "recommendations": [
      "ECG shows normal sinus rhythm",
      "Heart rate variability indicates healthy autonomic function",
      "Ejection fraction within normal limits",
      "No action required - continue routine monitoring"
    ]
  },
  
  "quality_assessment": {
    "overall_quality": "good",
    "echo_quality": {
      "signal_quality_score": 0.88,
      "frames_usable_percent": 94
    },
    "ecg_quality": {
      "signal_quality_score": 0.92,
      "noise_level_db": -35,
      "usable_segments_percent": 98
    }
  },
  
  "visualizations": {
    "ecg_waveform": {
      "available": true,
      "data": [0.12, 0.15, 0.18, ...],
      "sample_rate": 500,
      "r_peak_indices": [125, 542, 958, ...],
      "annotations": [
        {"type": "r_peak", "index": 125, "label": "R"},
        {"type": "interval", "start": 100, "end": 115, "label": "PR"}
      ]
    },
    "overlays": {
      "lv_contour_available": true,
      "gradcam_available": true
    }
  },
  
  "metadata": {
    "patient_context_used": true,
    "age_years": 65,
    "sex": "male",
    "clinical_history_factors": ["hypertension"]
  }
}
```

---

## 2. Failure Response Schema

### 2.1 Complete Failure Response
```json
{
  "success": false,
  "request_id": "req_abc123def456",
  "timestamp": "2026-01-19T10:30:00.000Z",
  "processing_time_ms": 150,
  
  "error": {
    "code": "E_VAL_005",
    "message": "Signal duration 3.5s below minimum 5s",
    "stage": "VALIDATION",
    "stage_index": 1,
    
    "user_message": {
      "title": "ECG Recording Too Short",
      "explanation": "The ECG recording needs to be at least 5 seconds long to perform accurate heart rate variability analysis.",
      "action": "Please provide a longer ECG recording (recommended: 30 seconds or more)."
    },
    
    "technical_details": {
      "actual_duration_sec": 3.5,
      "minimum_duration_sec": 5,
      "sample_rate_hz": 500
    },
    
    "recoverable": false,
    "resubmission_hint": "Increase ECG duration to at least 5 seconds"
  },
  
  "stages_completed": [
    {"stage": "RECEIPT", "status": "success", "time_ms": 5}
  ],
  
  "stages_failed": [
    {"stage": "VALIDATION", "status": "failed", "time_ms": 145, "error_code": "E_VAL_005"}
  ]
}
```

### 2.2 Partial Success Response
```json
{
  "success": true,
  "partial": true,
  "request_id": "req_abc123def456",
  "timestamp": "2026-01-19T10:30:00.000Z",
  "processing_time_ms": 1850,
  
  "warnings": [
    {
      "code": "W_DET_001",
      "message": "Echo detection confidence below threshold",
      "stage": "DETECTION",
      "impact": "Echo analysis results may be less reliable",
      "recommendation": "Consider uploading clearer echo images"
    }
  ],
  
  "clinical_results": {
    "echo_analysis": {
      "available": true,
      "confidence_warning": true,
      "ejection_fraction": {
        "value": 52.0,
        "confidence": 0.55,
        "flagged_for_review": true
      }
    },
    "ecg_analysis": {
      "available": true,
      "rhythm_analysis": {
        "classification": "Normal Sinus Rhythm",
        "heart_rate_bpm": 72,
        "confidence": 0.94
      }
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
| stages_completed | array | List of completed stages |
| clinical_results | object | Analysis results |
| risk_assessment | object | Risk score and factors |
| quality_assessment | object | Signal/image quality |

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
| visualizations | If requested | ECG waveform, overlays |
| metadata | If provided | Patient context used |
| warnings | If partial success | Non-fatal issues |
| partial | If some analysis failed | Indicates incomplete results |

---

## 4. Confidence Levels

### 4.1 Confidence Reporting
| Level | Range | Interpretation |
|-------|-------|----------------|
| High | >= 0.85 | Reliable result |
| Medium | 0.70-0.84 | Acceptable, monitor |
| Low | 0.50-0.69 | Review recommended |
| Very Low | < 0.50 | Flagged for review |

### 4.2 Confidence Display
```json
{
  "confidence": 0.89,
  "confidence_level": "high",
  "confidence_display": {
    "percentage": 89,
    "description": "High confidence",
    "color": "green",
    "icon": "check-circle"
  }
}
```

---

## 5. Visual Overlays

### 5.1 ECG Annotations
```json
{
  "ecg_annotations": [
    {
      "type": "r_peak",
      "sample_index": 125,
      "time_sec": 0.25,
      "marker_style": {"color": "#ef4444", "size": 6}
    },
    {
      "type": "interval",
      "name": "PR",
      "start_index": 100,
      "end_index": 115,
      "duration_ms": 30,
      "style": {"color": "#3b82f6", "thickness": 2}
    },
    {
      "type": "abnormal_beat",
      "sample_index": 958,
      "label": "PVC",
      "style": {"color": "#f59e0b", "highlight": true}
    }
  ]
}
```

### 5.2 Echo Overlays
```json
{
  "echo_overlays": {
    "lv_contour": {
      "available": true,
      "format": "svg_path",
      "frame_indices": [0, 15, 30],
      "paths": ["M 112 140 ..."]
    },
    "gradcam": {
      "available": true,
      "format": "base64_png",
      "data": "iVBORw0KGgo..."
    }
  }
}
```

---

## 6. API Versioning

### 6.1 Version Header
```
X-API-Version: 2026-01-19
Content-Type: application/json
```

### 6.2 Backward Compatibility
| Version | Changes |
|---------|---------|
| 2026-01-19 | Added echo analysis |
| 2025-12-01 | Initial ECG-only version |

---

## 7. Response Size Limits

| Field | Max Size | Notes |
|-------|----------|-------|
| ecg_waveform.data | 50,000 samples | 100 seconds @ 500Hz |
| overlays (total) | 1 MB | Base64 encoded |
| Total response | 5 MB | Compressed |
