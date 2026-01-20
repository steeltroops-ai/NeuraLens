# 03 - Lesion Localization and Segmentation

## Purpose
Define the detection and segmentation modules that locate, extract, and validate lesion boundaries from preprocessed skin images.

---

## Module Overview

```
+------------------+     +------------------+     +------------------+
|  PREPROCESSED    |---->| LESION           |---->| BOUNDARY         |
|  IMAGE           |     | DETECTION        |     | REFINEMENT       |
+------------------+     +------------------+     +------------------+
                                                          |
        +-------------------------------------------------+
        |
        v
+------------------+     +------------------+     +------------------+
|  SEMANTIC        |---->| INSTANCE         |---->| GEOMETRY         |
|  SEGMENTATION    |     | SEGMENTATION     |     | EXTRACTION       |
+------------------+     +------------------+     +------------------+
                                                          |
        +-------------------------------------------------+
        |
        v
+------------------+     +------------------+
|  VALIDATION &    |---->| SEGMENTATION     |
|  CONFIDENCE      |     | OUTPUT           |
+------------------+     +------------------+
```

---

## 1. Lesion Detection Module

### Purpose
Locate the primary lesion region in the image using object detection techniques.

### Algorithm: YOLO-based Detection + Fallback

```python
class LesionDetector:
    """
    Detects lesion bounding boxes in skin images.
    Uses YOLO architecture with fallback to traditional methods.
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ):
        self.model = self._load_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """Detect lesions in image."""
        # Primary detection using deep learning
        detections = self._dl_detect(image)
        
        # Fallback to traditional methods if no detections
        if len(detections) == 0:
            detections = self._traditional_detect(image)
            source = "traditional"
        else:
            source = "deep_learning"
        
        # Select primary lesion (highest confidence, most centered)
        primary_lesion = self._select_primary(detections, image.shape)
        
        return DetectionResult(
            detections=detections,
            primary_lesion=primary_lesion,
            detection_source=source,
            confidence=primary_lesion.confidence if primary_lesion else 0.0
        )
    
    def _dl_detect(self, image: np.ndarray) -> List[LesionBoundingBox]:
        """Deep learning based detection."""
        # Preprocess for model
        blob = cv2.dnn.blobFromImage(
            image, 1/255.0, (416, 416), 
            swapRB=True, crop=False
        )
        self.model.setInput(blob)
        outputs = self.model.forward(self.model.getUnconnectedOutLayersNames())
        
        detections = []
        h, w = image.shape[:2]
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Extract bounding box
                    cx, cy, bw, bh = detection[:4]
                    x = int((cx - bw/2) * w)
                    y = int((cy - bh/2) * h)
                    width = int(bw * w)
                    height = int(bh * h)
                    
                    detections.append(LesionBoundingBox(
                        x=x, y=y, width=width, height=height,
                        confidence=float(confidence),
                        class_id=class_id
                    ))
        
        # Apply NMS
        if len(detections) > 1:
            detections = self._apply_nms(detections)
        
        return detections
    
    def _traditional_detect(self, image: np.ndarray) -> List[LesionBoundingBox]:
        """Fallback detection using color and edge analysis."""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Otsu's thresholding on L channel
        _, binary = cv2.threshold(
            lab[:, :, 0], 0, 255, 
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            img_area = image.shape[0] * image.shape[1]
            
            # Filter by reasonable size
            if 0.001 * img_area < area < 0.8 * img_area:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append(LesionBoundingBox(
                    x=x, y=y, width=w, height=h,
                    confidence=0.6,  # Lower confidence for traditional
                    class_id=0
                ))
        
        return detections
    
    def _select_primary(
        self, 
        detections: List[LesionBoundingBox], 
        image_shape: tuple
    ) -> Optional[LesionBoundingBox]:
        """Select the primary lesion from multiple detections."""
        if len(detections) == 0:
            return None
        if len(detections) == 1:
            return detections[0]
        
        h, w = image_shape[:2]
        center_x, center_y = w / 2, h / 2
        
        # Score by confidence and centrality
        scored = []
        for det in detections:
            det_center_x = det.x + det.width / 2
            det_center_y = det.y + det.height / 2
            
            # Distance from center (normalized)
            dist = np.sqrt(
                ((det_center_x - center_x) / w)**2 + 
                ((det_center_y - center_y) / h)**2
            )
            centrality = 1.0 - min(dist, 1.0)
            
            # Combined score
            score = det.confidence * 0.6 + centrality * 0.4
            scored.append((score, det))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
```

---

## 2. Semantic Segmentation Module

### Purpose
Produce pixel-level lesion masks using deep semantic segmentation.

### Architecture: U-Net with EfficientNet Encoder

```python
class LesionSegmenter:
    """
    Performs semantic segmentation to produce lesion masks.
    Uses U-Net architecture with pretrained encoder.
    """
    
    def __init__(
        self,
        model_path: str,
        encoder: str = "efficientnet-b4",
        threshold: float = 0.5
    ):
        self.model = self._load_model(model_path)
        self.encoder = encoder
        self.threshold = threshold
        self.input_size = (512, 512)
    
    def segment(
        self, 
        image: np.ndarray, 
        detection: Optional[LesionBoundingBox] = None
    ) -> SegmentationResult:
        """Segment lesion in image."""
        # Prepare input
        input_tensor = self._preprocess(image)
        
        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Generate binary mask
        mask = (probabilities > self.threshold).astype(np.uint8) * 255
        
        # Post-process mask
        mask = self._postprocess_mask(mask)
        
        # Compute confidence map
        confidence_map = self._compute_confidence(probabilities)
        
        # Validate segmentation
        validation = self._validate_segmentation(mask, image.shape)
        
        return SegmentationResult(
            mask=mask,
            probability_map=probabilities,
            confidence_map=confidence_map,
            validation=validation,
            mean_confidence=float(np.mean(probabilities[mask > 0])) if np.any(mask > 0) else 0.0
        )
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize
        resized = cv2.resize(image, self.input_size)
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        normalized = (normalized - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # To tensor
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
        
        return tensor
    
    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Clean up segmentation mask."""
        # Fill holes
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, 
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        )
        
        # Remove small connected components
        mask = self._remove_small_components(mask, min_size=100)
        
        # Keep only largest component
        mask = self._keep_largest_component(mask)
        
        return mask
    
    def _compute_confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """Compute per-pixel confidence scores."""
        # Confidence is higher when probability is far from 0.5
        confidence = 2 * np.abs(probabilities - 0.5)
        return confidence
    
    def _remove_small_components(self, mask: np.ndarray, min_size: int) -> np.ndarray:
        """Remove connected components smaller than min_size."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                mask[labels == i] = 0
        
        return mask
    
    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        
        if num_labels <= 1:
            return mask
        
        # Find largest component (excluding background)
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        output = np.zeros_like(mask)
        output[labels == largest_idx] = 255
        
        return output
```

---

## 3. Boundary Refinement Module

### Purpose
Refine segmentation boundaries for accurate lesion delineation.

### Algorithm: Active Contour (Snake) + CRF

```python
class BoundaryRefiner:
    """
    Refines lesion boundaries using active contours and CRF.
    """
    
    def __init__(
        self,
        alpha: float = 0.015,   # Snake - Elasticity
        beta: float = 10.0,     # Snake - Rigidity
        gamma: float = 0.001,   # Snake - Viscosity
        crf_iterations: int = 5
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.crf_iterations = crf_iterations
    
    def refine(
        self, 
        image: np.ndarray, 
        initial_mask: np.ndarray
    ) -> RefinementResult:
        """Refine segmentation boundary."""
        # Extract initial contour
        contours, _ = cv2.findContours(
            initial_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        
        if len(contours) == 0:
            return RefinementResult(mask=initial_mask, refined=False)
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Apply active contour
        refined_contour = self._active_contour(image, contour)
        
        # Create refined mask
        refined_mask = np.zeros_like(initial_mask)
        cv2.drawContours(refined_mask, [refined_contour], -1, 255, -1)
        
        # Apply CRF for final refinement
        final_mask = self._crf_refinement(image, refined_mask)
        
        # Compute improvement metrics
        iou = self._compute_iou(initial_mask, final_mask)
        boundary_distance = self._compute_boundary_distance(initial_mask, final_mask)
        
        return RefinementResult(
            mask=final_mask,
            refined=True,
            contour=refined_contour,
            iou_with_initial=iou,
            boundary_shift=boundary_distance
        )
    
    def _active_contour(
        self, 
        image: np.ndarray, 
        initial_contour: np.ndarray
    ) -> np.ndarray:
        """Apply active contour evolution."""
        # Convert to grayscale for gradient computation
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute gradient magnitude
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        
        # Invert so edges attract
        external_energy = -cv2.GaussianBlur(gradient_magnitude, (5, 5), 0)
        
        # Initialize snake points
        snake = initial_contour.squeeze().astype(np.float64)
        
        # Iterate
        for _ in range(100):
            snake = self._snake_iteration(snake, external_energy)
        
        return snake.astype(np.int32).reshape(-1, 1, 2)
    
    def _crf_refinement(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> np.ndarray:
        """Apply Conditional Random Field for boundary refinement."""
        try:
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_softmax
            
            # Create CRF
            d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
            
            # Unary potentials
            probs = np.stack([1 - mask/255, mask/255], axis=0).astype(np.float32)
            unary = unary_from_softmax(probs)
            d.setUnaryEnergy(unary)
            
            # Pairwise potentials
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(
                sxy=50, srgb=13, rgbim=image, compat=10
            )
            
            # Inference
            Q = d.inference(self.crf_iterations)
            refined = np.argmax(Q, axis=0).reshape(image.shape[:2])
            
            return (refined * 255).astype(np.uint8)
            
        except ImportError:
            # Fall back to morphological refinement
            return self._morphological_refinement(mask)
    
    def _morphological_refinement(self, mask: np.ndarray) -> np.ndarray:
        """Fallback morphological refinement."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
        return refined
```

---

## 4. Geometry Extraction Module

### Purpose
Extract geometric properties from segmented lesion for clinical assessment.

### Measurements

```python
class GeometryExtractor:
    """
    Extracts geometric measurements from lesion mask.
    """
    
    PIXELS_PER_MM = 10  # Default calibration, adjustable
    
    def extract(
        self, 
        mask: np.ndarray, 
        image: np.ndarray,
        calibration: Optional[float] = None
    ) -> GeometryResult:
        """Extract geometric properties from lesion mask."""
        if calibration:
            self.PIXELS_PER_MM = calibration
        
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        
        if len(contours) == 0:
            return GeometryResult(valid=False, error="No lesion contour found")
        
        contour = max(contours, key=cv2.contourArea)
        
        # Basic measurements
        area_px = cv2.contourArea(contour)
        perimeter_px = cv2.arcLength(contour, True)
        
        # Bounding geometry
        x, y, w, h = cv2.boundingRect(contour)
        
        # Center and diameter
        M = cv2.moments(contour)
        if M["m00"] > 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = x + w//2, y + h//2
        
        # Fit ellipse for major/minor axes
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (cx, cy), (minor_axis, major_axis), angle = ellipse
        else:
            major_axis = max(w, h)
            minor_axis = min(w, h)
            angle = 0
        
        # Equivalent diameter
        equiv_diameter_px = np.sqrt(4 * area_px / np.pi)
        
        # Circularity
        circularity = 4 * np.pi * area_px / (perimeter_px ** 2) if perimeter_px > 0 else 0
        
        # Asymmetry indices
        asymmetry = self._compute_asymmetry(mask, contour, center_x, center_y)
        
        # Border irregularity
        border_irregularity = self._compute_border_irregularity(contour)
        
        # Convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area_px / hull_area if hull_area > 0 else 0
        
        # Convert to mm
        area_mm2 = area_px / (self.PIXELS_PER_MM ** 2)
        diameter_mm = equiv_diameter_px / self.PIXELS_PER_MM
        major_mm = major_axis / self.PIXELS_PER_MM
        minor_mm = minor_axis / self.PIXELS_PER_MM
        
        return GeometryResult(
            valid=True,
            contour=contour,
            center=(center_x, center_y),
            bounding_box=(x, y, w, h),
            
            # Pixel measurements
            area_pixels=area_px,
            perimeter_pixels=perimeter_px,
            diameter_pixels=equiv_diameter_px,
            
            # Calibrated measurements (mm)
            area_mm2=area_mm2,
            diameter_mm=diameter_mm,
            major_axis_mm=major_mm,
            minor_axis_mm=minor_mm,
            
            # Shape features
            circularity=circularity,
            asymmetry_index=asymmetry,
            border_irregularity=border_irregularity,
            solidity=solidity,
            aspect_ratio=major_axis / minor_axis if minor_axis > 0 else 1.0,
            orientation=angle
        )
    
    def _compute_asymmetry(
        self, 
        mask: np.ndarray, 
        contour: np.ndarray,
        cx: int, 
        cy: int
    ) -> float:
        """Compute asymmetry index (0=symmetric, 1=asymmetric)."""
        # Horizontal asymmetry
        left_half = mask[:, :cx]
        right_half = cv2.flip(mask[:, cx:], 1)
        
        # Align sizes
        min_w = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, -min_w:]
        right_half = right_half[:, :min_w]
        
        h_asymmetry = 1 - self._compute_overlap(left_half, right_half)
        
        # Vertical asymmetry
        top_half = mask[:cy, :]
        bottom_half = cv2.flip(mask[cy:, :], 0)
        
        min_h = min(top_half.shape[0], bottom_half.shape[0])
        top_half = top_half[-min_h:, :]
        bottom_half = bottom_half[:min_h, :]
        
        v_asymmetry = 1 - self._compute_overlap(top_half, bottom_half)
        
        return (h_asymmetry + v_asymmetry) / 2
    
    def _compute_overlap(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute overlap ratio between two masks."""
        intersection = np.sum((mask1 > 0) & (mask2 > 0))
        union = np.sum((mask1 > 0) | (mask2 > 0))
        return intersection / union if union > 0 else 1.0
    
    def _compute_border_irregularity(self, contour: np.ndarray) -> float:
        """Compute border irregularity score."""
        # Compute curvature at each point
        contour_squeezed = contour.squeeze()
        n_points = len(contour_squeezed)
        
        if n_points < 10:
            return 0.0
        
        curvatures = []
        for i in range(n_points):
            prev_idx = (i - 5) % n_points
            next_idx = (i + 5) % n_points
            
            p1 = contour_squeezed[prev_idx]
            p2 = contour_squeezed[i]
            p3 = contour_squeezed[next_idx]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cross = np.cross(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                curvature = np.abs(cross) / (norm1 * norm2)
                curvatures.append(curvature)
        
        if len(curvatures) == 0:
            return 0.0
        
        # Irregularity = variance in curvature
        return float(np.std(curvatures))
```

---

## 5. Validation Module

### Purpose
Validate segmentation outputs for clinical plausibility.

```python
class SegmentationValidator:
    """
    Validates segmentation results for clinical plausibility.
    """
    
    # Thresholds
    MIN_LESION_RATIO = 0.001     # Minimum lesion/image ratio
    MAX_LESION_RATIO = 0.80      # Maximum lesion/image ratio
    MIN_DIAMETER_MM = 1.0        # Minimum plausible diameter
    MAX_DIAMETER_MM = 100.0      # Maximum plausible diameter
    MIN_CIRCULARITY = 0.1        # Minimum circularity
    MAX_FRAGMENTATION = 3        # Maximum allowed fragments
    
    def validate(
        self, 
        mask: np.ndarray,
        geometry: GeometryResult,
        image_shape: tuple
    ) -> ValidationResult:
        """Validate segmentation output."""
        issues = []
        
        # Check lesion size ratio
        img_area = image_shape[0] * image_shape[1]
        lesion_ratio = geometry.area_pixels / img_area
        
        if lesion_ratio < self.MIN_LESION_RATIO:
            issues.append(ValidationIssue(
                code="SEG_V01",
                severity="error",
                message="Lesion too small - may not be correctly detected"
            ))
        elif lesion_ratio > self.MAX_LESION_RATIO:
            issues.append(ValidationIssue(
                code="SEG_V02",
                severity="warning",
                message="Lesion occupies most of image - ensure proper framing"
            ))
        
        # Check plausible diameter
        if geometry.diameter_mm < self.MIN_DIAMETER_MM:
            issues.append(ValidationIssue(
                code="SEG_V03",
                severity="warning",
                message="Lesion diameter very small - verify calibration"
            ))
        elif geometry.diameter_mm > self.MAX_DIAMETER_MM:
            issues.append(ValidationIssue(
                code="SEG_V04",
                severity="warning",
                message="Lesion diameter unusually large"
            ))
        
        # Check circularity
        if geometry.circularity < self.MIN_CIRCULARITY:
            issues.append(ValidationIssue(
                code="SEG_V05",
                severity="warning",
                message="Lesion has very irregular shape - verify segmentation"
            ))
        
        # Check fragmentation
        num_components = self._count_components(mask)
        if num_components > self.MAX_FRAGMENTATION:
            issues.append(ValidationIssue(
                code="SEG_V06",
                severity="error",
                message="Segmentation appears fragmented"
            ))
        
        # Check border proximity
        if self._touches_border(mask):
            issues.append(ValidationIssue(
                code="SEG_V07",
                severity="warning",
                message="Lesion touches image border - may be truncated"
            ))
        
        # Determine overall validity
        errors = [i for i in issues if i.severity == "error"]
        valid = len(errors) == 0
        
        return ValidationResult(
            valid=valid,
            issues=issues,
            confidence=1.0 - len(issues) * 0.1
        )
    
    def _count_components(self, mask: np.ndarray) -> int:
        """Count connected components in mask."""
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask)
        return num_labels - 1  # Exclude background
    
    def _touches_border(self, mask: np.ndarray) -> bool:
        """Check if mask touches image borders."""
        h, w = mask.shape
        return (
            np.any(mask[0, :] > 0) or
            np.any(mask[-1, :] > 0) or
            np.any(mask[:, 0] > 0) or
            np.any(mask[:, -1] > 0)
        )
```

---

## 6. Output Schema

```python
@dataclass
class LesionMaskOutput:
    """Complete lesion segmentation output."""
    
    # Detection
    detection: DetectionResult
    
    # Segmentation
    mask: np.ndarray                      # Binary mask (H x W)
    probability_map: np.ndarray           # Probability at each pixel
    confidence_map: np.ndarray            # Confidence at each pixel
    
    # Geometry
    geometry: GeometryResult
    
    # Refinement
    refinement: RefinementResult
    
    # Validation
    validation: ValidationResult
    
    # Coordinate system
    coordinate_system: str = "image"      # "image" or "normalized"
    
    # Processing info
    processing_time_ms: int
    model_version: str
```

---

## 7. Segmentation vs Detection Tradeoffs

| Approach | Pros | Cons | Use When |
|----------|------|------|----------|
| Detection Only | Fast, robust | No boundary | Quick triage |
| Semantic Seg | Precise boundaries | Slower, may fail | Clinical assessment |
| Instance Seg | Multiple lesions | Complex | Multi-lesion imaging |
| Hybrid | Balanced | More complex | Default approach |

**Recommended:** Hybrid approach - detection for localization, semantic segmentation for boundary extraction.

---

## 8. Pixel-Level Confidence Estimation

```python
class ConfidenceEstimator:
    """
    Estimates per-pixel segmentation confidence.
    """
    
    def estimate(
        self, 
        probability_map: np.ndarray,
        mask: np.ndarray
    ) -> ConfidenceOutput:
        """Estimate per-pixel and overall confidence."""
        # Per-pixel confidence (distance from 0.5)
        pixel_confidence = 2 * np.abs(probability_map - 0.5)
        
        # Boundary uncertainty (lower confidence near boundaries)
        boundary = cv2.Canny(mask, 100, 200)
        boundary_distance = cv2.distanceTransform(255 - boundary, cv2.DIST_L2, 5)
        boundary_factor = np.minimum(boundary_distance / 10, 1.0)
        
        adjusted_confidence = pixel_confidence * boundary_factor
        
        # Overall metrics
        mean_confidence = float(np.mean(adjusted_confidence[mask > 0]))
        boundary_confidence = float(np.mean(pixel_confidence[boundary > 0]))
        
        return ConfidenceOutput(
            pixel_confidence=adjusted_confidence,
            mean_confidence=mean_confidence,
            boundary_confidence=boundary_confidence,
            low_confidence_ratio=float(np.mean(adjusted_confidence < 0.5))
        )
```

---

## 9. Coordinate Systems

All measurements support two coordinate systems:

### Image Coordinates
- Origin: Top-left corner (0, 0)
- Units: Pixels
- X increases right, Y increases down

### Normalized Coordinates
- Origin: Top-left corner (0.0, 0.0)
- Range: (0.0, 0.0) to (1.0, 1.0)
- Device-independent

```python
def to_normalized(coord: tuple, image_shape: tuple) -> tuple:
    """Convert image coordinates to normalized."""
    h, w = image_shape[:2]
    x, y = coord
    return (x / w, y / h)

def from_normalized(coord: tuple, image_shape: tuple) -> tuple:
    """Convert normalized coordinates to image."""
    h, w = image_shape[:2]
    nx, ny = coord
    return (int(nx * w), int(ny * h))
```
