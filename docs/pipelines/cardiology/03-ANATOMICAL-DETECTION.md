# 03 - Anatomical Structure Detection

## Document Info
| Field | Value |
|-------|-------|
| Stage | 3 - Anatomical Detection |
| Owner | Computer Vision Engineer |
| Reviewer | Cardiologist |

---

## 1. Detection Targets

### 1.1 Cardiac Chambers
| Structure | View | Priority | Method |
|-----------|------|----------|--------|
| Left Ventricle (LV) | A4C, PLAX, PSAX | P1 | Segmentation |
| Right Ventricle (RV) | A4C | P1 | Segmentation |
| Left Atrium (LA) | A4C, PLAX | P2 | Segmentation |
| Right Atrium (RA) | A4C | P2 | Segmentation |

### 1.2 Ventricular Walls
| Wall Segment | Location | Clinical Relevance |
|--------------|----------|-------------------|
| Septal | LV/RV border | Motion abnormality |
| Anterior | LV front | LAD territory |
| Lateral | LV side | LCx territory |
| Inferior | LV bottom | RCA territory |
| Posterior | LV back | RCA/LCx territory |
| Apex | LV tip | Apical aneurysm |

### 1.3 Valves
| Valve | View | Detection Method |
|-------|------|-----------------|
| Mitral | A4C, PLAX | Keypoint detection |
| Aortic | PLAX, A3C | Keypoint detection |
| Tricuspid | A4C | Keypoint detection |
| Pulmonic | PSAX | Keypoint detection |

---

## 2. Detection Strategy

### 2.1 Segmentation vs Keypoint

**Segmentation (for chambers):**
- Full boundary delineation
- Area/volume calculation
- U-Net or Mask R-CNN architecture
- Output: Pixel-wise mask

**Keypoint Detection (for valves, landmarks):**
- Specific anatomical points
- HRNet or pose estimation model
- Output: (x, y, confidence) per point

### 2.2 Model Architecture

```
EchoNet-Dynamic Style Architecture
----------------------------------
Input: 224x224 grayscale frame
    |
    v
ResNet-50 Backbone (pretrained ImageNet)
    |
    v
Feature Pyramid Network
    |
    +-- Chamber Head --> Segmentation masks (LV, RV, LA, RA)
    |
    +-- Landmark Head --> Keypoints (valve annuli, apex)
    |
    +-- Quality Head --> Frame quality score
```

---

## 3. Anatomical Map Representation

### 3.1 Output Format
```json
{
  "frame_id": 0,
  "timestamp_ms": 0,
  "structures": {
    "left_ventricle": {
      "mask": "base64_encoded_mask",
      "area_pixels": 4250,
      "centroid": [112, 140],
      "confidence": 0.94,
      "phase": "end_diastole"
    },
    "right_ventricle": {
      "mask": "base64_encoded_mask",
      "area_pixels": 1890,
      "centroid": [80, 135],
      "confidence": 0.88
    },
    "mitral_valve": {
      "annulus_points": [[95, 160], [130, 158]],
      "leaflet_points": [[112, 145]],
      "confidence": 0.91
    }
  },
  "coordinate_system": "image_plane",
  "normalized_axes": {
    "apex": [112, 200],
    "base": [112, 100],
    "lateral": [160, 150],
    "septal": [65, 150]
  }
}
```

---

## 4. Validation Rules

### 4.1 Physiological Constraints
| Constraint | Rule | Action if Violated |
|------------|------|-------------------|
| LV area | 1000-15000 px (at 224x224) | Flag as abnormal |
| RV/LV ratio | 0.3-0.8 | Flag if outside range |
| LA/LV ratio | 0.3-0.6 | Flag if enlarged |
| Chamber overlap | < 5% | Reject detection |

### 4.2 Implausible Shape Detection
- **Convexity check:** LV should be roughly elliptical
- **Aspect ratio:** Height/width within 1.2-2.5
- **Smoothness:** Boundary gradient continuity

### 4.3 Temporal Consistency
- **Frame-to-frame change:** Area change < 20% between adjacent frames
- **Centroid drift:** < 20 pixels between frames
- **Phase coherence:** Contraction/relaxation pattern consistent

---

## 5. Coordinate Systems

### 5.1 Image Plane (pixels)
- Origin: Top-left corner
- X: Left to right
- Y: Top to bottom
- Units: Pixels

### 5.2 Normalized Cardiac Axes
- Origin: LV apex
- Long axis: Apex to mitral annulus center
- Short axis: Perpendicular to long axis
- Units: Normalized 0-1

---

## 6. Stage Output

```json
{
  "stage_complete": "DETECTION",
  "stage_id": 3,
  "status": "success",
  "frames_analyzed": 450,
  "structures_detected": {
    "left_ventricle": {"frames": 448, "mean_confidence": 0.92},
    "right_ventricle": {"frames": 445, "mean_confidence": 0.87},
    "mitral_valve": {"frames": 440, "mean_confidence": 0.89}
  },
  "validation": {
    "physiological_constraints": "passed",
    "temporal_consistency": "passed",
    "shape_validation": "passed"
  },
  "next_stage": "ANALYSIS"
}
```
