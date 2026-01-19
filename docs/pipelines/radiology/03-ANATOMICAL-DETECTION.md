# 03 - Anatomical Structure Detection

## Document Info
| Field | Value |
|-------|-------|
| Stage | 3 - Anatomical Detection |
| Owner | Computer Vision Engineer |
| Reviewer | Radiologist |

---

## 1. Overview

### 1.1 Purpose
Detect and segment anatomical structures to:
- Provide spatial context for pathology localization
- Validate image content and orientation
- Enable region-specific analysis
- Support clinical reporting with precise locations

### 1.2 Anatomical Targets by Modality

| Modality | Body Region | Structures Detected |
|----------|-------------|---------------------|
| Chest X-Ray | Thorax | Lungs, Heart silhouette, Ribs, Clavicles, Mediastinum |
| CT Chest | Thorax | Lungs, Lobes, Heart, Aorta, Airways, Vertebrae |
| CT Abdomen | Abdomen | Liver, Spleen, Kidneys, Pancreas, Aorta |
| MRI Brain | Head | Gray matter, White matter, CSF, Ventricles, Major structures |
| MRI Spine | Spine | Vertebrae, Discs, Spinal cord, Nerve roots |

---

## 2. Detection Approaches

### 2.1 Segmentation vs Detection Tradeoffs

| Approach | Pros | Cons | Use When |
|----------|------|------|----------|
| **Semantic Segmentation** | Precise boundaries, full coverage | Slower, memory intensive | Volumetric analysis, lesion volumes |
| **Instance Segmentation** | Individual objects | Complex post-processing | Multiple similar structures |
| **Bounding Box Detection** | Fast, simple | No precise boundaries | Localization only |
| **Landmark Detection** | Very fast | Sparse information | Quick validation |

### 2.2 Recommended Approach by Structure

| Structure | Approach | Model Type |
|-----------|----------|------------|
| Lungs | Semantic segmentation | U-Net |
| Heart | Bounding box + contour | DenseNet + regression |
| Liver | Semantic segmentation | 3D U-Net |
| Brain structures | Multi-class segmentation | nnU-Net |
| Ribs/Vertebrae | Instance segmentation | Mask R-CNN |

---

## 3. Coordinate Systems

### 3.1 Pixel Space
- Origin: Top-left corner
- Units: Pixels
- Axes: (row, column) or (y, x)

### 3.2 Physical Space (DICOM)
- Origin: Patient-specific reference
- Units: Millimeters
- Coordinate system: Patient-based (LPS or RAS)

### 3.3 Transformation
```python
def pixel_to_physical(pixel_coords, spacing, origin):
    """Convert pixel coordinates to physical space."""
    physical = []
    for i, p in enumerate(pixel_coords):
        physical.append(origin[i] + p * spacing[i])
    return tuple(physical)

def physical_to_pixel(physical_coords, spacing, origin):
    """Convert physical coordinates to pixel space."""
    pixel = []
    for i, p in enumerate(physical_coords):
        pixel.append(int((p - origin[i]) / spacing[i]))
    return tuple(pixel)
```

---

## 4. Chest X-Ray Anatomical Detection

### 4.1 Lung Segmentation
```python
class LungSegmentorXRay:
    """Lung segmentation for chest X-rays."""
    
    def __init__(self):
        self.model = self._load_model()
    
    def segment(self, image: np.ndarray) -> dict:
        """
        Segment lung fields.
        
        Returns:
            dict with 'left_lung', 'right_lung', 'combined' masks
        """
        # Preprocess
        img_resized = cv2.resize(image, (256, 256))
        img_norm = img_resized / 255.0
        
        # Predict
        mask = self.model.predict(img_norm[np.newaxis, ..., np.newaxis])
        
        # Post-process
        mask_binary = (mask[0, ..., 0] > 0.5).astype(np.uint8)
        
        # Separate left and right
        midline = mask_binary.shape[1] // 2
        left_lung = mask_binary.copy()
        left_lung[:, :midline] = 0
        
        right_lung = mask_binary.copy()
        right_lung[:, midline:] = 0
        
        # Resize to original
        combined = cv2.resize(mask_binary, (image.shape[1], image.shape[0]))
        
        return {
            "combined": combined,
            "left_lung": cv2.resize(left_lung, (image.shape[1], image.shape[0])),
            "right_lung": cv2.resize(right_lung, (image.shape[1], image.shape[0])),
            "lung_area_ratio": np.sum(combined) / combined.size
        }
```

### 4.2 Heart Silhouette Detection
```python
class HeartDetectorXRay:
    """Detect heart silhouette in chest X-ray."""
    
    def detect(self, image: np.ndarray, lung_mask: np.ndarray) -> dict:
        """
        Detect heart region.
        
        Returns:
            dict with bounding box, CTR (cardiothoracic ratio)
        """
        # Heart is typically in the lower medial region between lungs
        h, w = image.shape[:2]
        
        # Find lung boundaries
        lung_rows = np.any(lung_mask, axis=1)
        lung_cols = np.any(lung_mask, axis=0)
        
        lung_left = np.argmax(lung_cols)
        lung_right = w - np.argmax(lung_cols[::-1])
        
        thorax_width = lung_right - lung_left
        
        # Heart is approximately in center-left
        heart_left = lung_left + int(0.3 * thorax_width)
        heart_right = lung_left + int(0.7 * thorax_width)
        heart_top = int(0.3 * h)
        heart_bottom = int(0.8 * h)
        
        # Refine using intensity (heart is denser)
        roi = image[heart_top:heart_bottom, heart_left:heart_right]
        
        # Calculate cardiothoracic ratio (simplified)
        heart_width = heart_right - heart_left
        ctr = heart_width / thorax_width
        
        return {
            "bbox": [heart_left, heart_top, heart_right, heart_bottom],
            "cardiothoracic_ratio": ctr,
            "cardiomegaly": ctr > 0.5,
            "confidence": 0.85
        }
```

---

## 5. CT Anatomical Detection

### 5.1 Lung Lobe Segmentation
```python
class LungLobeSegmentorCT:
    """Segment lung lobes in CT volume."""
    
    LOBES = ["RUL", "RML", "RLL", "LUL", "LLL"]  # Right/Left Upper/Middle/Lower
    
    def segment(self, volume: np.ndarray) -> dict:
        """
        Segment lung lobes.
        
        Returns:
            dict with lobe masks and volumes
        """
        # First segment lungs
        lung_mask = self._segment_lungs(volume)
        
        # Then segment lobes using fissure detection
        lobe_masks = self._segment_lobes(volume, lung_mask)
        
        # Calculate volumes
        voxel_volume = 1.0  # mm^3, adjust based on spacing
        volumes = {}
        for lobe, mask in lobe_masks.items():
            volumes[f"{lobe}_volume_ml"] = np.sum(mask) * voxel_volume / 1000
        
        return {
            "lung_mask": lung_mask,
            "lobe_masks": lobe_masks,
            "volumes": volumes,
            "total_lung_volume_ml": np.sum(lung_mask) * voxel_volume / 1000
        }
```

### 5.2 Organ Segmentation (Abdomen)
```python
class AbdominalOrganSegmentor:
    """Segment abdominal organs in CT."""
    
    ORGANS = ["liver", "spleen", "left_kidney", "right_kidney", "pancreas"]
    
    def segment(self, volume: np.ndarray) -> dict:
        """Multi-organ segmentation using nnU-Net style model."""
        
        # Preprocess
        volume_norm = self._normalize(volume)
        
        # Run segmentation model
        segmentation = self._run_model(volume_norm)
        
        # Extract individual organs
        organ_masks = {}
        organ_volumes = {}
        
        for i, organ in enumerate(self.ORGANS):
            mask = (segmentation == i + 1).astype(np.uint8)
            organ_masks[organ] = mask
            organ_volumes[f"{organ}_volume_ml"] = np.sum(mask) * self.voxel_volume / 1000
        
        return {
            "organ_masks": organ_masks,
            "volumes": organ_volumes,
            "segmentation_map": segmentation
        }
```

---

## 6. MRI Brain Structure Detection

### 6.1 Brain Parcellation
```python
class BrainParcellator:
    """Segment brain structures in MRI."""
    
    STRUCTURES = [
        "gray_matter", "white_matter", "csf",
        "left_hippocampus", "right_hippocampus",
        "left_thalamus", "right_thalamus",
        "left_caudate", "right_caudate",
        "brainstem", "cerebellum"
    ]
    
    def segment(self, volume: np.ndarray) -> dict:
        """
        Parcellate brain into anatomical structures.
        
        Uses atlas-based or deep learning segmentation.
        """
        # Skull strip first
        brain_mask = self._skull_strip(volume)
        
        # Segment tissues
        tissue_probs = self._segment_tissues(volume, brain_mask)
        
        # Parcellate structures
        parcellation = self._parcellate(volume, brain_mask)
        
        return {
            "brain_mask": brain_mask,
            "tissue_probabilities": tissue_probs,
            "parcellation": parcellation,
            "structure_volumes": self._calculate_volumes(parcellation)
        }
```

---

## 7. Cross-Slice Consistency (Volumetric)

### 7.1 Consistency Checks
```python
def validate_volume_consistency(masks: np.ndarray) -> dict:
    """Validate anatomical consistency across slices."""
    
    results = {
        "is_consistent": True,
        "issues": []
    }
    
    # Check for size continuity
    areas = [np.sum(masks[i]) for i in range(masks.shape[0])]
    
    # Detect sudden jumps in area
    for i in range(1, len(areas) - 1):
        if areas[i] > 0:
            ratio_prev = areas[i] / max(areas[i-1], 1)
            ratio_next = areas[i] / max(areas[i+1], 1)
            
            if ratio_prev > 3 or ratio_prev < 0.33:
                results["issues"].append({
                    "type": "area_discontinuity",
                    "slice": i,
                    "ratio": ratio_prev
                })
    
    # Check for fragmentation
    for i in range(masks.shape[0]):
        if np.sum(masks[i]) > 0:
            labeled, num = label(masks[i])
            if num > 2:  # Allow 2 for bilateral structures
                results["issues"].append({
                    "type": "fragmentation",
                    "slice": i,
                    "fragments": num
                })
    
    results["is_consistent"] = len(results["issues"]) == 0
    return results
```

---

## 8. Anatomical Validation

### 8.1 Plausibility Checks

| Structure | Check | Normal Range | Error Code |
|-----------|-------|--------------|------------|
| Lung volume | Total volume | 4-7 L | W_ANAT_001 |
| Heart | CTR ratio | 0.35-0.50 | W_ANAT_002 |
| Liver volume | Volume | 1200-1800 mL | W_ANAT_003 |
| Brain | Total volume | 1100-1400 mL | W_ANAT_004 |
| Kidney | Size asymmetry | < 20% diff | W_ANAT_005 |

### 8.2 Spatial Relationship Validation
```python
def validate_spatial_relationships(structures: dict) -> list:
    """Check anatomical spatial relationships are valid."""
    
    issues = []
    
    # Heart should be left of center in chest
    if "heart" in structures and "lungs" in structures:
        heart_center = get_centroid(structures["heart"])
        lung_center = get_centroid(structures["lungs"])
        
        if heart_center[0] > lung_center[0]:
            issues.append({
                "type": "spatial_anomaly",
                "message": "Heart appears right of center (possible dextrocardia or image flip)",
                "severity": "warning"
            })
    
    # Liver should be right-sided
    if "liver" in structures:
        liver_center = get_centroid(structures["liver"])
        if liver_center[0] > structures["liver"].shape[1] / 2:
            issues.append({
                "type": "spatial_anomaly", 
                "message": "Liver appears left-sided (possible situs inversus)",
                "severity": "warning"
            })
    
    return issues
```

---

## 9. Output Specification

### 9.1 Anatomical Map Representation
```python
@dataclass
class AnatomicalMap:
    """Anatomical detection output."""
    
    # Structure masks (dict of name -> mask)
    structure_masks: Dict[str, np.ndarray]
    
    # Bounding boxes (dict of name -> [x1, y1, x2, y2])
    bounding_boxes: Dict[str, List[int]]
    
    # Centroids (dict of name -> [x, y, z])
    centroids: Dict[str, List[float]]
    
    # Volumes/areas (dict of name -> value in mL or cm^2)
    measurements: Dict[str, float]
    
    # Validation results
    validation: Dict[str, Any]
    
    # Confidence per structure
    confidences: Dict[str, float]
```

### 9.2 JSON Output Example
```json
{
  "structures": {
    "left_lung": {
      "mask_available": true,
      "bbox": [50, 100, 250, 450],
      "centroid": [150, 275],
      "area_cm2": 145.5,
      "confidence": 0.92
    },
    "right_lung": {
      "mask_available": true,
      "bbox": [280, 95, 480, 445],
      "centroid": [380, 270],
      "area_cm2": 152.3,
      "confidence": 0.94
    },
    "heart": {
      "mask_available": false,
      "bbox": [180, 200, 350, 420],
      "cardiothoracic_ratio": 0.48,
      "confidence": 0.87
    }
  },
  "validation": {
    "is_valid": true,
    "issues": []
  }
}
```

---

## 10. Stage Confirmation

```json
{
  "stage_complete": "DETECTION",
  "stage_id": 3,
  "status": "success",
  "timestamp": "2026-01-19T10:30:02.000Z",
  "summary": {
    "structures_detected": ["left_lung", "right_lung", "heart"],
    "all_structures_valid": true,
    "average_confidence": 0.91
  },
  "next_stage": "ANALYSIS"
}
```
