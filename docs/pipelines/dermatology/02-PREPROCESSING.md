# 02 - Preprocessing Stages

## Purpose
Define the image preprocessing pipeline that transforms raw skin lesion images into analysis-ready inputs with normalized quality, removed artifacts, and enhanced lesion visibility.

---

## Preprocessing Pipeline Overview

```
+----------------+     +----------------+     +----------------+
|   RAW IMAGE    |---->| COLOR CONST.   |---->| ILLUMINATION   |
|   (RGB/HEIC)   |     | NORMALIZATION  |     | CORRECTION     |
+----------------+     +----------------+     +----------------+
                                                     |
        +--------------------------------------------+
        |
        v
+----------------+     +----------------+     +----------------+
|   HAIR/ARTIFACT|---->| CONTRAST       |---->| BACKGROUND     |
|   REMOVAL      |     | ENHANCEMENT    |     | SUPPRESSION    |
+----------------+     +----------------+     +----------------+
                                                     |
        +--------------------------------------------+
        |
        v
+----------------+     +----------------+
|   RESIZE &     |---->| PREPROCESSED   |
|   ALIGNMENT    |     | OUTPUT         |
+----------------+     +----------------+
```

---

## Stage 1: Color Constancy Normalization

### Purpose
Remove illuminant color bias to ensure consistent lesion color representation regardless of lighting conditions.

### Algorithm: Shades of Gray

```python
class ColorConstancyNormalizer:
    """
    Implements Shades of Gray color constancy algorithm.
    Generalizes Gray World and Max-RGB methods.
    """
    
    def __init__(self, norm_order: int = 6):
        """
        Args:
            norm_order: Minkowski norm order. 
                       1 = Gray World, inf = Max-RGB, 6 = Shades of Gray
        """
        self.norm_order = norm_order
    
    def normalize(self, image: np.ndarray) -> ColorConstancyResult:
        """Apply color constancy normalization."""
        image_float = image.astype(np.float32) / 255.0
        
        # Compute illuminant estimate using Minkowski norm
        if self.norm_order == np.inf:
            illuminant = np.max(image_float, axis=(0, 1))
        else:
            illuminant = np.power(
                np.mean(np.power(image_float, self.norm_order), axis=(0, 1)),
                1.0 / self.norm_order
            )
        
        # Normalize illuminant
        illuminant = illuminant / np.linalg.norm(illuminant)
        
        # Apply correction
        target_illuminant = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        correction = target_illuminant / (illuminant + 1e-6)
        
        corrected = image_float * correction
        corrected = np.clip(corrected, 0, 1)
        
        output = (corrected * 255).astype(np.uint8)
        
        return ColorConstancyResult(
            image=output,
            illuminant_estimate=illuminant,
            correction_applied=correction,
            confidence=self._estimate_confidence(image_float, illuminant)
        )
    
    def _estimate_confidence(self, image: np.ndarray, illuminant: np.ndarray) -> float:
        """Estimate confidence in illuminant estimation."""
        # Lower confidence if illuminant is far from neutral
        neutral = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
        deviation = np.linalg.norm(illuminant - neutral)
        confidence = np.exp(-deviation * 2)
        return float(confidence)
```

### Input/Output Format
- **Input:** RGB image (H x W x 3), uint8
- **Output:** Color-corrected RGB image (H x W x 3), uint8

### Failure Conditions
| Condition | Detection | Action |
|-----------|-----------|--------|
| Near-black image | Mean intensity < 10 | Skip, log warning |
| Saturated image | >50% pixels at 255 | Reduce correction strength |
| Confidence < 0.3 | Algorithm confidence low | Apply partial correction |

### Quality Threshold
- Minimum confidence: 0.3
- Fallback: Apply 50% correction strength

---

## Stage 2: Illumination Correction

### Purpose
Correct for uneven lighting across the image, especially vignetting and directional shadows.

### Algorithm: Adaptive Histogram Equalization (CLAHE) + Homomorphic Filtering

```python
class IlluminationCorrector:
    """
    Corrects uneven illumination using CLAHE and homomorphic filtering.
    """
    
    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: tuple = (8, 8),
        homomorphic_cutoff: float = 0.3
    ):
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip_limit, 
            tileGridSize=clahe_tile_size
        )
        self.cutoff = homomorphic_cutoff
    
    def correct(self, image: np.ndarray) -> IlluminationResult:
        """Apply illumination correction."""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_clahe = self.clahe.apply(l_channel)
        
        # Homomorphic filtering for low-frequency correction
        l_homo = self._homomorphic_filter(l_clahe)
        
        # Merge channels
        corrected_lab = cv2.merge([l_homo, a_channel, b_channel])
        corrected_rgb = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB)
        
        # Measure improvement
        uniformity_before = self._measure_uniformity(l_channel)
        uniformity_after = self._measure_uniformity(l_homo)
        
        return IlluminationResult(
            image=corrected_rgb,
            uniformity_before=uniformity_before,
            uniformity_after=uniformity_after,
            improvement=uniformity_after - uniformity_before
        )
    
    def _homomorphic_filter(self, l_channel: np.ndarray) -> np.ndarray:
        """Apply homomorphic filtering to reduce low-frequency illumination."""
        # Log transform
        log_img = np.log1p(l_channel.astype(np.float32))
        
        # FFT
        dft = np.fft.fft2(log_img)
        dft_shift = np.fft.fftshift(dft)
        
        # High-pass filter
        rows, cols = l_channel.shape
        crow, ccol = rows // 2, cols // 2
        
        # Gaussian high-pass
        y, x = np.ogrid[:rows, :cols]
        d = np.sqrt((x - ccol)**2 + (y - crow)**2)
        h = 1 - np.exp(-(d**2) / (2 * (self.cutoff * min(rows, cols))**2))
        
        # Apply filter
        filtered = dft_shift * (0.5 + 0.5 * h)
        
        # Inverse FFT
        idft_shift = np.fft.ifftshift(filtered)
        idft = np.fft.ifft2(idft_shift)
        result = np.expm1(np.abs(idft))
        
        # Normalize
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
        return result.astype(np.uint8)
    
    def _measure_uniformity(self, image: np.ndarray) -> float:
        """Measure illumination uniformity (0-1)."""
        blocks = self._divide_into_blocks(image, 4, 4)
        means = [np.mean(b) for b in blocks]
        std = np.std(means)
        uniformity = 1.0 - min(1.0, std / 50.0)
        return uniformity
```

### Input/Output Format
- **Input:** Color-corrected RGB image
- **Output:** Illumination-corrected RGB image

### Failure Conditions
| Condition | Detection | Action |
|-----------|-----------|--------|
| Already uniform | uniformity > 0.9 | Skip processing |
| FFT failure | Exception in filtering | Fall back to CLAHE only |
| No improvement | improvement < 0.05 | Use original L channel |

---

## Stage 3: Hair and Artifact Removal

### Purpose
Remove hair, ruler markings, air bubbles, and other artifacts that obscure the lesion.

### Algorithm: DullRazor + Inpainting

```python
class HairArtifactRemover:
    """
    Removes hair and artifacts using morphological operations and inpainting.
    Based on DullRazor algorithm with enhancements.
    """
    
    def __init__(
        self,
        hair_kernel_size: int = 17,
        inpaint_radius: int = 5
    ):
        self.hair_kernel_size = hair_kernel_size
        self.inpaint_radius = inpaint_radius
    
    def remove(self, image: np.ndarray) -> ArtifactRemovalResult:
        """Remove hair and artifacts from image."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Step 1: Detect hair using morphological black-hat
        hair_mask = self._detect_hair(gray)
        
        # Step 2: Detect other artifacts (rulers, bubbles)
        artifact_mask = self._detect_artifacts(image)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(hair_mask, artifact_mask)
        
        # Step 3: Inpaint to remove detected artifacts
        inpainted = cv2.inpaint(
            image, 
            combined_mask, 
            self.inpaint_radius, 
            cv2.INPAINT_TELEA
        )
        
        # Calculate removal statistics
        hair_coverage = np.sum(hair_mask > 0) / hair_mask.size
        artifact_coverage = np.sum(artifact_mask > 0) / artifact_mask.size
        
        return ArtifactRemovalResult(
            image=inpainted,
            hair_mask=hair_mask,
            artifact_mask=artifact_mask,
            hair_coverage=hair_coverage,
            artifact_coverage=artifact_coverage,
            total_inpainted_ratio=np.sum(combined_mask > 0) / combined_mask.size
        )
    
    def _detect_hair(self, gray: np.ndarray) -> np.ndarray:
        """Detect hair using morphological black-hat transform."""
        # Create line-shaped kernels at multiple orientations
        masks = []
        for angle in range(0, 180, 15):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, 
                (self.hair_kernel_size, 1)
            )
            kernel = self._rotate_kernel(kernel, angle)
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            masks.append(blackhat)
        
        # Combine all orientations
        combined = np.max(masks, axis=0)
        
        # Threshold
        _, hair_mask = cv2.threshold(combined, 10, 255, cv2.THRESH_BINARY)
        
        # Clean up with morphological operations
        hair_mask = cv2.morphologyEx(
            hair_mask, 
            cv2.MORPH_CLOSE, 
            np.ones((3, 3), np.uint8)
        )
        
        return hair_mask
    
    def _detect_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Detect non-hair artifacts like rulers, bubbles, ink."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Detect very bright spots (bubbles, reflections)
        _, bright_mask = cv2.threshold(hsv[:, :, 2], 250, 255, cv2.THRESH_BINARY)
        
        # Detect rulers (high contrast linear structures)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100)
        
        ruler_mask = np.zeros(gray.shape, dtype=np.uint8)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(ruler_mask, (x1, y1), (x2, y2), 255, 3)
        
        # Detect ink markings (highly saturated non-skin colors)
        ink_mask = self._detect_ink(hsv)
        
        # Combine all artifact masks
        artifact_mask = cv2.bitwise_or(bright_mask, ruler_mask)
        artifact_mask = cv2.bitwise_or(artifact_mask, ink_mask)
        
        return artifact_mask
    
    def _detect_ink(self, hsv: np.ndarray) -> np.ndarray:
        """Detect ink markings (blue, purple, black marker)."""
        # Blue ink
        blue_lower = np.array([100, 100, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Purple ink
        purple_lower = np.array([130, 50, 50])
        purple_upper = np.array([160, 255, 255])
        purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
        
        return cv2.bitwise_or(blue_mask, purple_mask)
```

### Input/Output Format
- **Input:** Illumination-corrected RGB image
- **Output:** Artifact-free RGB image + artifact masks

### Failure Conditions
| Condition | Detection | Action |
|-----------|-----------|--------|
| Excessive hair | hair_coverage > 0.4 | Warn user, proceed with caution |
| Large artifact area | total_inpainted > 0.3 | Warn, may affect accuracy |
| Inpainting failure | Exception | Return original with mask |

---

## Stage 4: Lesion Contrast Enhancement

### Purpose
Enhance the visual contrast between the lesion and surrounding skin for better segmentation.

### Algorithm: Color-Based Enhancement

```python
class LesionContrastEnhancer:
    """
    Enhances lesion visibility through adaptive color space manipulation.
    """
    
    def enhance(self, image: np.ndarray) -> ContrastEnhanceResult:
        """Enhance lesion-skin contrast."""
        # Convert to multiple color spaces
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Enhance L channel in LAB (lightness contrast)
        l_enhanced = self._enhance_channel(lab[:, :, 0], "lightness")
        
        # Enhance A channel (red-green, captures pigmentation)
        a_enhanced = self._enhance_channel(lab[:, :, 1], "color")
        
        # Enhance B channel (blue-yellow)
        b_enhanced = self._enhance_channel(lab[:, :, 2], "color")
        
        # Reconstruct
        enhanced_lab = cv2.merge([l_enhanced, a_enhanced, b_enhanced])
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        
        # Measure contrast improvement
        contrast_before = self._measure_lesion_contrast(image)
        contrast_after = self._measure_lesion_contrast(enhanced_rgb)
        
        return ContrastEnhanceResult(
            image=enhanced_rgb,
            contrast_before=contrast_before,
            contrast_after=contrast_after,
            improvement_ratio=contrast_after / max(contrast_before, 1e-6)
        )
    
    def _enhance_channel(self, channel: np.ndarray, mode: str) -> np.ndarray:
        """Apply adaptive enhancement to a single channel."""
        if mode == "lightness":
            # Sigmoid-based contrast stretching
            normalized = channel.astype(np.float32) / 255.0
            mean = np.mean(normalized)
            enhanced = 1 / (1 + np.exp(-10 * (normalized - mean)))
            return (enhanced * 255).astype(np.uint8)
        else:
            # Linear stretch for color channels
            p2, p98 = np.percentile(channel, (2, 98))
            stretched = np.clip((channel - p2) * 255.0 / (p98 - p2), 0, 255)
            return stretched.astype(np.uint8)
    
    def _measure_lesion_contrast(self, image: np.ndarray) -> float:
        """Estimate lesion-background contrast."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Simple contrast measure using standard deviation
        return float(np.std(gray))
```

---

## Stage 5: Background Skin Suppression

### Purpose
Suppress the surrounding normal skin to isolate the lesion region for analysis.

### Algorithm: Skin Color Modeling + Attenuation

```python
class BackgroundSuppressor:
    """
    Suppresses normal skin background to highlight lesion.
    """
    
    def suppress(self, image: np.ndarray, lesion_mask: np.ndarray = None) -> SuppressResult:
        """Suppress background skin while preserving lesion."""
        # Estimate skin color from periphery
        skin_color = self._estimate_skin_color(image, lesion_mask)
        
        # Compute pixel-wise skin probability
        skin_prob = self._compute_skin_probability(image, skin_color)
        
        # Create soft suppression mask
        suppression = 1.0 - skin_prob * 0.5  # Reduce skin brightness by up to 50%
        
        # Apply suppression
        suppressed = image.astype(np.float32) * suppression[:, :, np.newaxis]
        suppressed = np.clip(suppressed, 0, 255).astype(np.uint8)
        
        return SuppressResult(
            image=suppressed,
            skin_color_estimate=skin_color,
            suppression_map=suppression
        )
    
    def _estimate_skin_color(self, image: np.ndarray, lesion_mask: np.ndarray) -> np.ndarray:
        """Estimate normal skin color from image periphery."""
        h, w = image.shape[:2]
        
        # Create periphery mask (outer 20%)
        margin = int(min(h, w) * 0.2)
        periphery_mask = np.ones((h, w), dtype=np.uint8)
        periphery_mask[margin:h-margin, margin:w-margin] = 0
        
        # Exclude lesion if mask provided
        if lesion_mask is not None:
            periphery_mask = cv2.bitwise_and(periphery_mask, cv2.bitwise_not(lesion_mask))
        
        # Compute mean color in periphery
        skin_pixels = image[periphery_mask > 0]
        skin_color = np.mean(skin_pixels, axis=0) if len(skin_pixels) > 0 else np.array([180, 140, 120])
        
        return skin_color
    
    def _compute_skin_probability(self, image: np.ndarray, skin_color: np.ndarray) -> np.ndarray:
        """Compute probability of each pixel being normal skin."""
        # Color distance from estimated skin
        diff = image.astype(np.float32) - skin_color
        distance = np.sqrt(np.sum(diff**2, axis=2))
        
        # Convert to probability (closer = higher prob)
        max_dist = np.sqrt(3 * 255**2)
        prob = 1.0 - (distance / max_dist)
        
        return prob
```

---

## Stage 6: Image Resizing and Alignment

### Purpose
Standardize image dimensions for model input while preserving aspect ratio and lesion positioning.

### Algorithm: Aspect-Preserving Resize with Padding

```python
class ImageResizer:
    """
    Resizes images to model input dimensions while preserving aspect ratio.
    """
    
    def __init__(
        self,
        target_size: tuple = (512, 512),
        padding_color: tuple = (0, 0, 0),
        preserve_aspect: bool = True
    ):
        self.target_size = target_size
        self.padding_color = padding_color
        self.preserve_aspect = preserve_aspect
    
    def resize(self, image: np.ndarray) -> ResizeResult:
        """Resize image to target dimensions."""
        h, w = image.shape[:2]
        target_w, target_h = self.target_size
        
        if self.preserve_aspect:
            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Pad to target size
            pad_left = (target_w - new_w) // 2
            pad_top = (target_h - new_h) // 2
            
            output = np.full((target_h, target_w, 3), self.padding_color, dtype=np.uint8)
            output[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized
            
            # Store transform info for coordinate mapping
            transform = {
                "scale": scale,
                "offset": (pad_left, pad_top),
                "original_size": (w, h),
                "resized_size": (new_w, new_h)
            }
        else:
            output = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
            transform = {
                "scale_x": target_w / w,
                "scale_y": target_h / h
            }
        
        return ResizeResult(
            image=output,
            transform=transform
        )
    
    def inverse_transform(self, coords: np.ndarray, transform: dict) -> np.ndarray:
        """Map coordinates from resized space back to original space."""
        scale = transform["scale"]
        offset_x, offset_y = transform["offset"]
        
        # Remove offset and scale
        original_coords = (coords - np.array([offset_x, offset_y])) / scale
        
        return original_coords
```

---

## Preprocessing Quality Thresholds

| Stage | Metric | Minimum | Optimal | Action if Below |
|-------|--------|---------|---------|-----------------|
| Color Constancy | Confidence | 0.30 | 0.70 | Partial correction |
| Illumination | Uniformity | 0.40 | 0.80 | Log warning |
| Hair Removal | Coverage | - | < 0.20 | User retake if > 0.40 |
| Contrast | Improvement | 1.0x | 1.5x | Skip if no improvement |
| Final Quality | Combined | 0.50 | 0.80 | Proceed with caution |

---

## Output Schema

```python
@dataclass
class PreprocessingResult:
    """Complete preprocessing output."""
    
    # Processed image
    image: np.ndarray                    # Final preprocessed image
    
    # Stage outputs
    color_constancy: ColorConstancyResult
    illumination: IlluminationResult
    artifact_removal: ArtifactRemovalResult
    contrast: ContrastEnhanceResult
    background: SuppressResult
    resize: ResizeResult
    
    # Quality metrics
    overall_quality: float               # 0-1 score
    processing_time_ms: int
    warnings: List[str]
    
    # Transform info (for mapping back to original)
    coordinate_transform: dict
    
    # Confidence
    confidence: float                    # Overall preprocessing confidence
```

---

## Fallback Behavior Matrix

| Stage | Failure Type | Fallback Action | Impact |
|-------|--------------|-----------------|--------|
| Color Constancy | Low confidence | Apply 50% correction | Minor |
| Illumination | FFT failure | CLAHE only | Minor |
| Hair Removal | Exception | Skip, use original | Moderate |
| Contrast | No improvement | Skip stage | Minor |
| Background | No skin detected | Skip suppression | Minor |
| Resize | Invalid dimensions | Maintain original | Blocks analysis |
