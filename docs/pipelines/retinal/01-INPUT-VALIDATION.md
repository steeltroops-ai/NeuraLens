# Retinal Pipeline - Input Interface & Validation Specification

## Document Info
| Field | Value |
|-------|-------|
| Version | 4.0.0 |
| Authors | Ophthalmologist, CV Engineer, ML Architect |
| Last Updated | 2026-01-19 |

---

## 1. Input Interface

### 1.1 API Endpoint
```
POST /api/retinal/analyze
Content-Type: multipart/form-data
```

### 1.2 Request Schema
```json
{
  "image": "File (required) - Fundus photograph",
  "session_id": "string (optional) - UUID for tracking",
  "eye": "string (optional) - 'left' | 'right' | 'unknown'",
  "patient_age": "integer (optional) - For age-adjusted analysis"
}
```

---

## 2. Validation Checklist

### 2.1 File Type Validation
| Check | Criteria | Error Code |
|-------|----------|------------|
| MIME Type | `image/jpeg`, `image/png`, `image/tiff` | `VAL_001` |
| Magic Bytes | FFD8 (JPEG), 89504E47 (PNG) | `VAL_002` |
| Extension | `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff` | `VAL_003` |

### 2.2 Resolution Validation
| Check | Criteria | Error Code |
|-------|----------|------------|
| Minimum | 512 x 512 pixels | `VAL_010` |
| Recommended | 1024 x 1024+ pixels | `VAL_011` (warning) |
| Maximum | 8192 x 8192 pixels | `VAL_012` |
| Aspect Ratio | 0.8 - 1.2 (near-square) | `VAL_013` |

### 2.3 Illumination Validation
| Check | Criteria | Error Code |
|-------|----------|------------|
| Mean Intensity | 40-220 (8-bit range) | `VAL_020` |
| Uniformity | CV < 0.4 across quadrants | `VAL_021` |
| Over-exposure | <5% pixels at 255 | `VAL_022` |
| Under-exposure | <20% pixels at 0 | `VAL_023` |

### 2.4 Field-of-View Validation
| Check | Criteria | Error Code |
|-------|----------|------------|
| Circular FOV | Fundus circle detected | `VAL_030` |
| Coverage | >60% image area | `VAL_031` |
| Centering | Optic disc within frame | `VAL_032` |

### 2.5 Retina Detection (vs Non-Fundus)
| Check | Method | Error Code |
|-------|--------|------------|
| Color Profile | Red channel dominance | `VAL_040` |
| Vessel Pattern | Edge detection confirms vessels | `VAL_041` |
| Optic Disc | Circular bright region detected | `VAL_042` |
| Non-Fundus Rejection | CNN classifier confidence <0.5 | `VAL_043` |

### 2.6 Corruption Detection
| Check | Criteria | Error Code |
|-------|----------|------------|
| File Size | >1KB | `VAL_050` |
| Decodable | OpenCV/PIL can read | `VAL_051` |
| Dimensions | width>0, height>0 | `VAL_052` |
| Color Channels | 3 channels (RGB) | `VAL_053` |

---

## 3. Error Response Schema

### 3.1 Validation Error Response
```json
{
  "success": false,
  "error": {
    "stage": "input_validation",
    "error_type": "validation_error",
    "code": "VAL_001",
    "message": "Invalid file format. Expected JPEG or PNG.",
    "details": {
      "provided_mime": "application/pdf",
      "accepted_formats": ["image/jpeg", "image/png", "image/tiff"]
    },
    "resubmission_recommended": true,
    "guidance": "Please upload a fundus photograph in JPEG or PNG format."
  },
  "session_id": "uuid",
  "timestamp": "2026-01-19T08:00:00Z"
}
```

### 3.2 Error Code Registry
| Code | Category | Message | Resubmit? |
|------|----------|---------|-----------|
| `VAL_001` | Format | Invalid file format | Yes |
| `VAL_002` | Format | Corrupted file header | Yes |
| `VAL_010` | Resolution | Image too small | Yes |
| `VAL_020` | Quality | Poor illumination | Yes |
| `VAL_030` | FOV | No fundus detected | Yes |
| `VAL_040` | Content | Not a retinal image | Yes |
| `VAL_050` | Corruption | Empty or corrupted file | Yes |

---

## 4. Validation Flow

```
Image Upload
    |
    v
[File Size Check] --fail--> Error VAL_050
    |
    v
[Format Check] --fail--> Error VAL_001/002
    |
    v
[Decode Image] --fail--> Error VAL_051
    |
    v
[Resolution Check] --fail--> Error VAL_010
    |
    v
[Color Space Check] --fail--> Error VAL_053
    |
    v
[Illumination Check] --fail--> Error VAL_020-023
    |
    v
[Fundus Detection] --fail--> Error VAL_040-043
    |
    v
[Quality Score] --low--> Warning (proceed with caution)
    |
    v
VALIDATION PASSED --> Preprocessing Stage
```

---

## 5. Input Contract Summary

```python
INPUT_CONTRACT = {
    "file_constraints": {
        "max_size_mb": 15,
        "min_size_bytes": 1024,
        "formats": ["jpeg", "png", "tiff"],
    },
    "resolution_constraints": {
        "min": (512, 512),
        "max": (8192, 8192),
        "recommended": (1024, 1024),
        "aspect_ratio_range": (0.8, 1.2),
    },
    "quality_constraints": {
        "min_quality_score": 0.3,
        "recommended_quality_score": 0.6,
        "sharpness_threshold": 100,
        "snr_min_db": 15,
    },
    "content_constraints": {
        "must_be_fundus": True,
        "fundus_confidence_min": 0.5,
    }
}
```
