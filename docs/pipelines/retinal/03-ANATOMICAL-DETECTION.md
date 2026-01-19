# Retinal Pipeline - Anatomical Structure Detection

## Document Info
| Field | Value |
|-------|-------|
| Version | 4.0.0 |
| Pipeline Stage | 3 - Anatomical Detection |

---

## 1. Detection Modules Overview

```
Preprocessed Image
       |
       +---> [Optic Disc Detector] --> center, radius, cup boundary
       |
       +---> [Macula Detector] --> fovea center, macular region
       |
       +---> [Vascular Tree Segmenter] --> vessel mask, A/V classification
       |
       +---> [Fovea Localizer] --> precise foveal center
       |
       v
[Anatomical Consistency Check] --> Validate spatial relationships
       |
       v
Anatomical Map Output
```

---

## 2. Optic Disc Detection

### 2.1 Model Specification
| Attribute | Value |
|-----------|-------|
| Architecture | U-Net with ResNet34 encoder |
| Input Size | 512 x 512 |
| Output | Binary mask + center + radius |
| Training Data | REFUGE, ORIGA datasets |

### 2.2 Detection Pipeline
```python
class OpticDiscDetector:
    def detect(self, image: np.ndarray) -> OpticDiscResult:
        # 1. Segmentation model inference
        disc_mask = self.segmentation_model(image)
        
        # 2. Fit ellipse to segmented region
        contours = cv2.findContours(disc_mask, ...)
        ellipse = cv2.fitEllipse(contours[0])
        
        # 3. Cup segmentation within disc
        cup_mask = self.cup_segmentation_model(disc_crop)
        
        # 4. Calculate CDR
        cdr = cup_area / disc_area
        
        return OpticDiscResult(
            detected=True,
            center=(ellipse[0][0], ellipse[0][1]),
            radius=int((ellipse[1][0] + ellipse[1][1]) / 4),
            cup_to_disc_ratio=cdr,
            disc_area_mm2=pixels_to_mm2(disc_area),
            confidence=model_confidence
        )
```

### 2.3 Coordinate System
- Origin: Top-left of image
- X: Left to right (0 to width)
- Y: Top to bottom (0 to height)
- Units: Pixels (convert to mm using 30 pixels/degree assumption)

---

## 3. Macula Detection

### 3.1 Model Specification
| Attribute | Value |
|-----------|-------|
| Method | Anatomical relationship to optic disc |
| Fallback | Intensity-based detection |
| Output | Fovea center, macular region bounds |

### 3.2 Detection Algorithm
```python
def detect_macula(image: np.ndarray, optic_disc: OpticDiscResult) -> MacularResult:
    # Macula is typically 2-2.5 disc diameters temporal to disc
    disc_diameter = optic_disc.radius * 2
    
    # Estimate macula location (for right eye)
    macula_x = optic_disc.center[0] - (2.5 * disc_diameter)
    macula_y = optic_disc.center[1]
    
    # Refine using darkest region search
    search_region = image[
        int(macula_y - disc_diameter):int(macula_y + disc_diameter),
        int(macula_x - disc_diameter):int(macula_x + disc_diameter)
    ]
    
    # Red channel shows fovea as darker spot
    red_channel = search_region[:,:,0]
    fovea_loc = find_local_minimum(red_channel)
    
    return MacularResult(
        detected=True,
        center=adjusted_fovea_location,
        thickness_um=estimate_thickness(image, fovea_loc),
        confidence=calculate_confidence()
    )
```

---

## 4. Vascular Tree Segmentation

### 4.1 Model Specification
| Attribute | Value |
|-----------|-------|
| Architecture | U-Net / SA-UNet |
| Input Size | 512 x 512 (tiled for larger) |
| Output | Binary vessel mask |
| Training Data | DRIVE, STARE, CHASE_DB1 |
| Expected IoU | >0.80 |

### 4.2 Segmentation vs Detection Tradeoffs
| Approach | Pros | Cons | Use Case |
|----------|------|------|----------|
| Segmentation | Precise vessel boundaries | Slower, higher memory | Biomarker extraction |
| Detection | Faster, lighter | Less precise | Screening only |

**Recommendation**: Use segmentation for clinical biomarker extraction.

### 4.3 Artery/Vein Classification
```python
def classify_vessels(image: np.ndarray, vessel_mask: np.ndarray) -> dict:
    # Color-based A/V classification
    # Arteries: Brighter red, narrower
    # Veins: Darker red, wider
    
    vessel_colors = extract_vessel_colors(image, vessel_mask)
    
    # K-means clustering into 2 classes
    kmeans = KMeans(n_clusters=2)
    labels = kmeans.fit_predict(vessel_colors)
    
    # Assign based on brightness (arteries brighter)
    artery_label = identify_brighter_cluster(kmeans, labels)
    
    return {
        "artery_mask": vessel_mask * (labels == artery_label),
        "vein_mask": vessel_mask * (labels != artery_label),
        "confidence": silhouette_score(vessel_colors, labels)
    }
```

---

## 5. Fovea Center Localization

### 5.1 Algorithm
```python
def localize_fovea(image: np.ndarray, macula: MacularResult) -> tuple:
    # Fovea is darkest point within foveal avascular zone
    green_channel = image[:,:,1]  # Green shows best contrast
    
    # Search within 0.5 disc diameter of estimated macula
    search_radius = 50  # pixels
    roi = extract_circular_roi(green_channel, macula.center, search_radius)
    
    # Apply Gaussian blur and find minimum
    smoothed = cv2.GaussianBlur(roi, (15, 15), 0)
    min_loc = np.unravel_index(np.argmin(smoothed), smoothed.shape)
    
    return adjust_to_image_coords(min_loc, macula.center, search_radius)
```

---

## 6. Anatomical Map Representation

### 6.1 Output Schema
```python
@dataclass
class AnatomicalMap:
    # Optic Disc
    optic_disc_center: Tuple[int, int]  # (x, y) pixels
    optic_disc_radius: int  # pixels
    optic_cup_center: Tuple[int, int]
    optic_cup_radius: int
    cup_to_disc_ratio: float
    
    # Macula
    fovea_center: Tuple[int, int]
    macula_radius: int  # ~1500 microns = ~50 pixels
    
    # Vessels
    vessel_mask: np.ndarray  # Binary mask
    artery_mask: np.ndarray  # Artery subset
    vein_mask: np.ndarray  # Vein subset
    
    # Metadata
    image_dimensions: Tuple[int, int]
    pixel_to_mm_ratio: float
    laterality: str  # "OD" (right) or "OS" (left)
    confidence_scores: Dict[str, float]
```

---

## 7. Anatomical Consistency Validation

### 7.1 Impossible Geometry Checks
```python
def validate_anatomy(anatomy: AnatomicalMap) -> list:
    violations = []
    
    # 1. Disc-Macula distance check (should be ~15 degrees / ~4.5mm)
    dm_distance = euclidean(anatomy.optic_disc_center, anatomy.fovea_center)
    expected_dm = anatomy.image_dimensions[0] * 0.25  # ~25% of image width
    if not (0.15 < dm_distance / anatomy.image_dimensions[0] < 0.35):
        violations.append({
            "type": "DISC_MACULA_DISTANCE",
            "message": "Disc-macula distance outside expected range",
            "severity": "warning"
        })
    
    # 2. CDR sanity check (should be 0.1 - 0.9)
    if not (0.1 <= anatomy.cup_to_disc_ratio <= 0.9):
        violations.append({
            "type": "CDR_OUT_OF_RANGE",
            "message": f"CDR {anatomy.cup_to_disc_ratio} is anatomically implausible",
            "severity": "error"
        })
    
    # 3. Vessel density check (should have visible vessels)
    vessel_ratio = anatomy.vessel_mask.sum() / anatomy.vessel_mask.size
    if vessel_ratio < 0.05:
        violations.append({
            "type": "LOW_VESSEL_DENSITY",
            "message": "Vessel detection failed or severely reduced vasculature",
            "severity": "warning"
        })
    
    # 4. Macula should not overlap with optic disc
    dm_overlap = check_circle_overlap(
        anatomy.optic_disc_center, anatomy.optic_disc_radius,
        anatomy.fovea_center, anatomy.macula_radius
    )
    if dm_overlap:
        violations.append({
            "type": "MACULA_DISC_OVERLAP",
            "message": "Detected macula overlaps with optic disc",
            "severity": "error"
        })
    
    return violations
```

### 7.2 Validation Response
| Violation | Severity | Action |
|-----------|----------|--------|
| CDR out of range | Error | Reduce confidence, flag for review |
| Low vessel density | Warning | Continue with reduced biomarker confidence |
| Disc-macula distance | Warning | Check laterality assignment |
| Macula-disc overlap | Error | Re-run detection with alternate parameters |
