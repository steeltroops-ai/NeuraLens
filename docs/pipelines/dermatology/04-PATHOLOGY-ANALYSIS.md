# 04 - Pathology and Risk Analysis Modules

## Purpose
Define the disease detection and clinical risk assessment pipelines for skin lesion classification.

---

## Module Architecture

```
+------------------+     +------------------+     +------------------+
|  SEGMENTED       |---->| FEATURE          |---->| ABCDE            |
|  LESION          |     | EXTRACTION       |     | ANALYSIS         |
+------------------+     +------------------+     +------------------+
                                                          |
        +-------------------------------------------------+
        |
        v
+------------------+     +------------------+     +------------------+
|  MELANOMA        |---->| MALIGNANCY       |---->| SUBTYPE          |
|  CLASSIFIER      |     | ASSESSMENT       |     | CLASSIFICATION   |
+------------------+     +------------------+     +------------------+
                                                          |
        +-------------------------------------------------+
        |
        v
+------------------+     +------------------+
|  RISK TIER       |---->| PATHOLOGY        |
|  STRATIFICATION  |     | OUTPUT           |
+------------------+     +------------------+
```

---

## 1. Visual Feature Extraction (ABCDE Criteria)

### Purpose
Extract dermatologically-relevant features based on the ABCDE criteria for melanoma detection.

### A - Asymmetry Analysis

```python
class AsymmetryAnalyzer:
    """
    Analyzes lesion asymmetry along multiple axes.
    Asymmetry is a key indicator of malignancy.
    """
    
    def analyze(
        self, 
        mask: np.ndarray, 
        image: np.ndarray
    ) -> AsymmetryResult:
        """Compute asymmetry scores."""
        # Find centroid
        M = cv2.moments(mask)
        if M["m00"] == 0:
            return AsymmetryResult(valid=False)
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Shape asymmetry (binary mask)
        shape_horizontal = self._compute_axis_asymmetry(mask, cx, axis='horizontal')
        shape_vertical = self._compute_axis_asymmetry(mask, cy, axis='vertical')
        shape_asymmetry = (shape_horizontal + shape_vertical) / 2
        
        # Color asymmetry (using LAB color space)
        color_asymmetry = self._compute_color_asymmetry(image, mask, cx, cy)
        
        # Texture asymmetry
        texture_asymmetry = self._compute_texture_asymmetry(image, mask, cx, cy)
        
        # Combined asymmetry score (0-1, higher = more asymmetric)
        combined_score = (
            shape_asymmetry * 0.5 +
            color_asymmetry * 0.3 +
            texture_asymmetry * 0.2
        )
        
        # Classification
        if combined_score < 0.2:
            classification = "symmetric"
        elif combined_score < 0.4:
            classification = "mildly_asymmetric"
        elif combined_score < 0.6:
            classification = "moderately_asymmetric"
        else:
            classification = "highly_asymmetric"
        
        return AsymmetryResult(
            valid=True,
            shape_asymmetry=shape_asymmetry,
            color_asymmetry=color_asymmetry,
            texture_asymmetry=texture_asymmetry,
            combined_score=combined_score,
            classification=classification,
            is_concerning=combined_score > 0.4
        )
    
    def _compute_axis_asymmetry(
        self, 
        mask: np.ndarray, 
        center: int, 
        axis: str
    ) -> float:
        """Compute asymmetry along specified axis."""
        if axis == 'horizontal':
            half1 = mask[:, :center]
            half2 = cv2.flip(mask[:, center:], 1)
        else:
            half1 = mask[:center, :]
            half2 = cv2.flip(mask[center:, :], 0)
        
        # Align sizes
        min_size = min(half1.shape[0 if axis == 'vertical' else 1], 
                       half2.shape[0 if axis == 'vertical' else 1])
        
        if axis == 'horizontal':
            half1 = half1[:, -min_size:]
            half2 = half2[:, :min_size]
        else:
            half1 = half1[-min_size:, :]
            half2 = half2[:min_size, :]
        
        # Compute non-overlapping area
        intersection = np.sum((half1 > 0) & (half2 > 0))
        union = np.sum((half1 > 0) | (half2 > 0))
        
        if union == 0:
            return 0.0
        
        iou = intersection / union
        return 1.0 - iou
    
    def _compute_color_asymmetry(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        cx: int, 
        cy: int
    ) -> float:
        """Compute color distribution asymmetry."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Left vs right color histograms
        left_region = lab[:, :cx][mask[:, :cx] > 0]
        right_region = lab[:, cx:][mask[:, cx:] > 0]
        
        if len(left_region) == 0 or len(right_region) == 0:
            return 0.0
        
        # Compare mean colors
        left_mean = np.mean(left_region, axis=0)
        right_mean = np.mean(right_region, axis=0)
        
        color_diff = np.linalg.norm(left_mean - right_mean) / (255 * np.sqrt(3))
        
        return min(color_diff * 2, 1.0)  # Scale to 0-1
    
    def _compute_texture_asymmetry(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        cx: int, 
        cy: int
    ) -> float:
        """Compute texture pattern asymmetry using LBP."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute LBP
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        
        # Compare histograms of left/right
        left_lbp = lbp[:, :cx][mask[:, :cx] > 0]
        right_lbp = lbp[:, cx:][mask[:, cx:] > 0]
        
        if len(left_lbp) == 0 or len(right_lbp) == 0:
            return 0.0
        
        left_hist, _ = np.histogram(left_lbp, bins=10, range=(0, 10))
        right_hist, _ = np.histogram(right_lbp, bins=10, range=(0, 10))
        
        # Normalize
        left_hist = left_hist / (np.sum(left_hist) + 1e-6)
        right_hist = right_hist / (np.sum(right_hist) + 1e-6)
        
        # Chi-square distance
        chi2 = np.sum((left_hist - right_hist)**2 / (left_hist + right_hist + 1e-6))
        
        return min(chi2, 1.0)
```

### B - Border Irregularity Analysis

```python
class BorderAnalyzer:
    """
    Analyzes lesion border characteristics.
    Irregular, notched, or blurred borders indicate malignancy.
    """
    
    def analyze(self, mask: np.ndarray, image: np.ndarray) -> BorderResult:
        """Analyze border characteristics."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        
        if len(contours) == 0:
            return BorderResult(valid=False)
        
        contour = max(contours, key=cv2.contourArea)
        
        # Fractal dimension of border (higher = more irregular)
        fractal_dim = self._compute_fractal_dimension(contour)
        
        # Curvature analysis
        curvature_stats = self._analyze_curvature(contour)
        
        # Notch/lobule detection
        notches = self._detect_notches(contour)
        
        # Border sharpness
        sharpness = self._analyze_border_sharpness(image, mask)
        
        # Compactness (perimeter^2 / area)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
        
        # Border irregularity score
        irregularity_score = self._compute_irregularity_score(
            fractal_dim, curvature_stats, notches, compactness
        )
        
        return BorderResult(
            valid=True,
            fractal_dimension=fractal_dim,
            curvature_mean=curvature_stats['mean'],
            curvature_std=curvature_stats['std'],
            notch_count=len(notches),
            compactness=compactness,
            sharpness=sharpness,
            irregularity_score=irregularity_score,
            classification=self._classify_border(irregularity_score),
            is_concerning=irregularity_score > 0.5
        )
    
    def _compute_fractal_dimension(self, contour: np.ndarray) -> float:
        """Compute fractal dimension using box-counting method."""
        # Create binary image from contour
        points = contour.squeeze()
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        size = max(x_max - x_min, y_max - y_min) + 1
        img = np.zeros((size, size), dtype=np.uint8)
        
        shifted = points - [x_min, y_min]
        for p in shifted:
            img[min(p[1], size-1), min(p[0], size-1)] = 1
        
        # Box counting
        scales = [2, 4, 8, 16, 32]
        counts = []
        
        for scale in scales:
            scaled = cv2.resize(img, None, fx=1/scale, fy=1/scale, 
                               interpolation=cv2.INTER_MAX)
            counts.append(np.sum(scaled > 0))
        
        # Linear regression in log-log space
        log_scales = np.log(1 / np.array(scales))
        log_counts = np.log(np.array(counts) + 1)
        
        coeffs = np.polyfit(log_scales, log_counts, 1)
        fractal_dim = coeffs[0]
        
        return min(max(fractal_dim, 1.0), 2.0)  # Bounded 1-2
    
    def _analyze_curvature(self, contour: np.ndarray) -> dict:
        """Analyze curvature along the border."""
        points = contour.squeeze()
        n = len(points)
        
        curvatures = []
        for i in range(n):
            p1 = points[(i - 5) % n]
            p2 = points[i]
            p3 = points[(i + 5) % n]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            cross = np.cross(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 > 0 and norm2 > 0:
                curvature = cross / (norm1 * norm2)
                curvatures.append(abs(curvature))
        
        return {
            'mean': float(np.mean(curvatures)) if curvatures else 0,
            'std': float(np.std(curvatures)) if curvatures else 0,
            'max': float(np.max(curvatures)) if curvatures else 0
        }
    
    def _detect_notches(self, contour: np.ndarray) -> List[tuple]:
        """Detect notches (concave regions) in the border."""
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        
        notches = []
        if defects is not None:
            for defect in defects:
                start_idx, end_idx, far_idx, depth = defect[0]
                if depth > 500:  # Significant depth threshold
                    far_point = tuple(contour[far_idx][0])
                    notches.append(far_point)
        
        return notches
    
    def _classify_border(self, score: float) -> str:
        if score < 0.25:
            return "regular"
        elif score < 0.50:
            return "slightly_irregular"
        elif score < 0.75:
            return "moderately_irregular"
        else:
            return "highly_irregular"
```

### C - Color Analysis

```python
class ColorAnalyzer:
    """
    Analyzes color distribution within the lesion.
    Multiple colors and specific color patterns indicate malignancy.
    """
    
    # Dermoscopic colors of concern
    COLORS_OF_CONCERN = {
        'white': ([200, 200, 200], [255, 255, 255]),     # Regression
        'red': ([150, 50, 50], [255, 100, 100]),          # Vascularization
        'light_brown': ([139, 90, 43], [180, 140, 80]),
        'dark_brown': ([65, 40, 20], [120, 80, 50]),
        'blue_gray': ([80, 80, 100], [140, 140, 160]),    # Melanin deep
        'black': ([0, 0, 0], [40, 40, 40])                # Very concerning
    }
    
    def analyze(self, image: np.ndarray, mask: np.ndarray) -> ColorResult:
        """Analyze lesion color characteristics."""
        # Extract lesion pixels
        lesion_pixels = image[mask > 0]
        
        if len(lesion_pixels) == 0:
            return ColorResult(valid=False)
        
        # Color clustering
        color_clusters = self._cluster_colors(lesion_pixels)
        
        # Color variety score
        color_variety = self._compute_color_variety(color_clusters)
        
        # Detect concerning colors
        concerning_colors = self._detect_concerning_colors(image, mask)
        
        # Color homogeneity
        homogeneity = self._compute_homogeneity(lesion_pixels)
        
        # Blue-white veil detection
        blue_white_veil = self._detect_blue_white_veil(image, mask)
        
        # Color distribution analysis
        distribution = self._analyze_color_distribution(image, mask)
        
        # Combined color score
        color_score = self._compute_color_score(
            color_variety, concerning_colors, blue_white_veil
        )
        
        return ColorResult(
            valid=True,
            num_colors=len(color_clusters),
            color_clusters=color_clusters,
            color_variety_score=color_variety,
            concerning_colors=concerning_colors,
            has_blue_white_veil=blue_white_veil,
            homogeneity=homogeneity,
            distribution=distribution,
            color_score=color_score,
            is_concerning=color_score > 0.5 or len(concerning_colors) >= 3
        )
    
    def _cluster_colors(self, pixels: np.ndarray, n_clusters: int = 6) -> List[dict]:
        """Cluster colors using K-means."""
        from sklearn.cluster import KMeans
        
        # Subsample for efficiency
        if len(pixels) > 5000:
            indices = np.random.choice(len(pixels), 5000, replace=False)
            sample = pixels[indices]
        else:
            sample = pixels
        
        # Cluster
        kmeans = KMeans(n_clusters=min(n_clusters, len(sample) // 10 + 1), 
                        random_state=42, n_init=10)
        labels = kmeans.fit_predict(sample)
        
        clusters = []
        for i in range(kmeans.n_clusters):
            cluster_pixels = sample[labels == i]
            proportion = len(cluster_pixels) / len(sample)
            
            if proportion > 0.05:  # At least 5% of lesion
                clusters.append({
                    'center': kmeans.cluster_centers_[i].tolist(),
                    'proportion': proportion,
                    'name': self._name_color(kmeans.cluster_centers_[i])
                })
        
        return sorted(clusters, key=lambda x: x['proportion'], reverse=True)
    
    def _name_color(self, rgb: np.ndarray) -> str:
        """Assign a name to an RGB color."""
        r, g, b = rgb
        
        # Simple color naming based on HSV
        hsv = cv2.cvtColor(
            np.array([[rgb]], dtype=np.uint8), 
            cv2.COLOR_RGB2HSV
        )[0][0]
        
        h, s, v = hsv
        
        if v < 50:
            return "black"
        elif s < 30:
            if v > 200:
                return "white"
            else:
                return "gray"
        elif h < 10 or h > 170:
            return "red"
        elif 10 <= h < 30:
            return "brown"
        elif 30 <= h < 60:
            return "tan"
        elif 100 <= h < 130:
            return "blue"
        else:
            return "other"
    
    def _detect_concerning_colors(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> List[str]:
        """Detect dermoscopic colors of concern."""
        detected = []
        
        for color_name, (lower, upper) in self.COLORS_OF_CONCERN.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            
            color_mask = cv2.inRange(image, lower, upper)
            color_mask = cv2.bitwise_and(color_mask, mask)
            
            proportion = np.sum(color_mask > 0) / np.sum(mask > 0)
            
            if proportion > 0.03:  # At least 3% of lesion
                detected.append(color_name)
        
        return detected
    
    def _detect_blue_white_veil(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> bool:
        """Detect blue-white veil (strong melanoma indicator)."""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Blue-white veil: low saturation, moderate-high value, blue-ish hue
        bwv_mask = (
            (hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 130) &  # Blue hue
            (hsv[:, :, 1] < 80) &                          # Low saturation
            (hsv[:, :, 2] > 100) & (hsv[:, :, 2] < 200)   # Moderate value
        )
        
        bwv_in_lesion = bwv_mask & (mask > 0)
        proportion = np.sum(bwv_in_lesion) / np.sum(mask > 0)
        
        return proportion > 0.1  # At least 10% of lesion
```

### D - Diameter Analysis

```python
class DiameterAnalyzer:
    """
    Analyzes lesion diameter.
    Diameter > 6mm is concerning (but not definitive).
    """
    
    THRESHOLD_MM = 6.0
    
    def analyze(self, geometry: GeometryResult) -> DiameterResult:
        """Analyze lesion diameter."""
        diameter_mm = geometry.diameter_mm
        major_axis_mm = geometry.major_axis_mm
        minor_axis_mm = geometry.minor_axis_mm
        
        # Use maximum dimension
        max_dimension = max(diameter_mm, major_axis_mm)
        
        # Classification
        if max_dimension < 3.0:
            classification = "small"
            risk_contribution = 0.0
        elif max_dimension < 6.0:
            classification = "medium"
            risk_contribution = 0.2
        elif max_dimension < 10.0:
            classification = "concerning"
            risk_contribution = 0.5
        else:
            classification = "large"
            risk_contribution = 0.8
        
        return DiameterResult(
            diameter_mm=diameter_mm,
            major_axis_mm=major_axis_mm,
            minor_axis_mm=minor_axis_mm,
            max_dimension_mm=max_dimension,
            exceeds_threshold=max_dimension > self.THRESHOLD_MM,
            classification=classification,
            risk_contribution=risk_contribution,
            is_concerning=max_dimension >= self.THRESHOLD_MM
        )
```

### E - Evolution Analysis (Proxy)

```python
class EvolutionAnalyzer:
    """
    Analyzes evolution indicators (changes over time).
    Uses texture patterns as proxy for rapid growth.
    """
    
    def analyze(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        prior_image: Optional[np.ndarray] = None,
        prior_mask: Optional[np.ndarray] = None
    ) -> EvolutionResult:
        """Analyze evolution indicators."""
        # Static indicators (from single image)
        texture_heterogeneity = self._compute_texture_heterogeneity(image, mask)
        growth_pattern = self._analyze_growth_pattern(mask)
        
        # Dynamic indicators (if prior image available)
        if prior_image is not None and prior_mask is not None:
            change_analysis = self._compare_to_prior(
                image, mask, prior_image, prior_mask
            )
        else:
            change_analysis = None
        
        # Evolution score based on available data
        if change_analysis is not None:
            evolution_score = change_analysis['overall_change']
        else:
            # Use texture heterogeneity as proxy
            evolution_score = min(texture_heterogeneity * 0.5, 1.0)
        
        return EvolutionResult(
            texture_heterogeneity=texture_heterogeneity,
            growth_pattern=growth_pattern,
            prior_comparison=change_analysis,
            evolution_score=evolution_score,
            has_prior=prior_image is not None,
            classification=self._classify_evolution(evolution_score),
            is_concerning=evolution_score > 0.4
        )
    
    def _compute_texture_heterogeneity(
        self, 
        image: np.ndarray, 
        mask: np.ndarray
    ) -> float:
        """Compute texture heterogeneity as proxy for irregular growth."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # GLCM-based texture features
        from skimage.feature import graycomatrix, graycoprops
        
        lesion_gray = gray.copy()
        lesion_gray[mask == 0] = 0
        
        # Compute GLCM
        glcm = graycomatrix(
            lesion_gray, 
            distances=[1], 
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
            levels=256,
            symmetric=True,
            normed=True
        )
        
        # Extract features
        contrast = graycoprops(glcm, 'contrast').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        
        # Heterogeneity = high contrast, low homogeneity
        heterogeneity = (contrast / 100) * (1 - homogeneity)
        
        return min(heterogeneity, 1.0)
    
    def _compare_to_prior(
        self,
        current_image: np.ndarray,
        current_mask: np.ndarray,
        prior_image: np.ndarray,
        prior_mask: np.ndarray
    ) -> dict:
        """Compare current lesion to prior image."""
        # Size change
        current_area = np.sum(current_mask > 0)
        prior_area = np.sum(prior_mask > 0)
        size_change = (current_area - prior_area) / prior_area if prior_area > 0 else 0
        
        # Color change
        current_colors = current_image[current_mask > 0].mean(axis=0)
        prior_colors = prior_image[prior_mask > 0].mean(axis=0)
        color_change = np.linalg.norm(current_colors - prior_colors) / 255
        
        # Shape change (IoU)
        intersection = np.sum((current_mask > 0) & (prior_mask > 0))
        union = np.sum((current_mask > 0) | (prior_mask > 0))
        shape_similarity = intersection / union if union > 0 else 0
        shape_change = 1 - shape_similarity
        
        overall_change = (
            abs(size_change) * 0.4 +
            color_change * 0.3 +
            shape_change * 0.3
        )
        
        return {
            'size_change_ratio': size_change,
            'color_change': color_change,
            'shape_change': shape_change,
            'overall_change': min(overall_change, 1.0),
            'grew_larger': size_change > 0.1,
            'changed_color': color_change > 0.1,
            'changed_shape': shape_change > 0.2
        }
```

---

## 2. Melanoma Probability Estimation

```python
class MelanomaClassifier:
    """
    Estimates melanoma probability using deep learning and ABCDE features.
    """
    
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.abcde_weight = 0.3
        self.dl_weight = 0.7
    
    def classify(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        abcde_features: ABCDEFeatures
    ) -> MelanomaResult:
        """Classify melanoma probability."""
        # Deep learning prediction
        dl_prob = self._dl_predict(image, mask)
        
        # ABCDE-based scoring
        abcde_score = self._abcde_score(abcde_features)
        
        # Combined probability (weighted ensemble)
        combined_prob = (
            self.dl_weight * dl_prob + 
            self.abcde_weight * abcde_score
        )
        
        # Uncertainty estimation
        uncertainty = self._estimate_uncertainty(dl_prob, abcde_score)
        
        # Classification
        if combined_prob < 0.15:
            classification = "unlikely"
        elif combined_prob < 0.40:
            classification = "low_suspicion"
        elif combined_prob < 0.70:
            classification = "moderate_suspicion"
        else:
            classification = "high_suspicion"
        
        return MelanomaResult(
            probability=combined_prob,
            dl_probability=dl_prob,
            abcde_probability=abcde_score,
            uncertainty=uncertainty,
            classification=classification,
            concerning_features=self._get_concerning_features(abcde_features),
            recommendation=self._get_recommendation(combined_prob, uncertainty)
        )
    
    def _abcde_score(self, features: ABCDEFeatures) -> float:
        """Compute melanoma risk from ABCDE features."""
        score = 0.0
        
        # A: Asymmetry (weight: 0.25)
        if features.asymmetry.is_concerning:
            score += 0.25 * features.asymmetry.combined_score
        
        # B: Border (weight: 0.20)
        if features.border.is_concerning:
            score += 0.20 * features.border.irregularity_score
        
        # C: Color (weight: 0.25)
        if features.color.is_concerning:
            score += 0.25 * features.color.color_score
        
        # D: Diameter (weight: 0.15)
        if features.diameter.is_concerning:
            score += 0.15 * features.diameter.risk_contribution
        
        # E: Evolution (weight: 0.15)
        if features.evolution.is_concerning:
            score += 0.15 * features.evolution.evolution_score
        
        return min(score, 1.0)
    
    def _estimate_uncertainty(self, dl_prob: float, abcde_score: float) -> float:
        """Estimate prediction uncertainty."""
        # Higher uncertainty when DL and ABCDE disagree
        disagreement = abs(dl_prob - abcde_score)
        
        # Higher uncertainty when probability is near decision boundary
        boundary_distance = abs(0.5 - (dl_prob + abcde_score) / 2)
        boundary_uncertainty = 1 - 2 * boundary_distance
        
        return (disagreement * 0.6 + boundary_uncertainty * 0.4)
```

---

## 3. Benign vs Malignant Classification

```python
class MalignancyClassifier:
    """
    Binary classification: benign vs malignant (any type).
    """
    
    CLASSES = ["benign", "malignant"]
    
    def classify(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        features: ExtractedFeatures
    ) -> MalignancyResult:
        """Classify as benign or malignant."""
        # Model inference
        with torch.no_grad():
            input_tensor = self._prepare_input(image, mask)
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        benign_prob = probs[0]
        malignant_prob = probs[1]
        
        # Feature-based adjustment
        if features.abcde.total_concerning >= 3:
            malignant_prob = min(malignant_prob * 1.2, 0.95)
            benign_prob = 1 - malignant_prob
        
        return MalignancyResult(
            classification="malignant" if malignant_prob > 0.5 else "benign",
            benign_probability=benign_prob,
            malignant_probability=malignant_prob,
            confidence=abs(malignant_prob - 0.5) * 2,
            needs_biopsy=malignant_prob > 0.3
        )
```

---

## 4. Subtype Classification

```python
class SubtypeClassifier:
    """
    Multi-class classification for specific lesion types.
    """
    
    CLASSES = [
        "melanoma",
        "basal_cell_carcinoma",
        "squamous_cell_carcinoma",
        "actinic_keratosis",
        "benign_keratosis",
        "dermatofibroma",
        "nevus",
        "vascular_lesion"
    ]
    
    def classify(self, image: np.ndarray, mask: np.ndarray) -> SubtypeResult:
        """Classify lesion subtype."""
        with torch.no_grad():
            input_tensor = self._prepare_input(image, mask)
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        
        # Get top predictions
        sorted_indices = np.argsort(probs)[::-1]
        
        predictions = []
        for idx in sorted_indices[:3]:
            predictions.append({
                "subtype": self.CLASSES[idx],
                "probability": float(probs[idx]),
                "is_malignant": self.CLASSES[idx] in [
                    "melanoma", "basal_cell_carcinoma", "squamous_cell_carcinoma"
                ]
            })
        
        top_prediction = predictions[0]
        
        return SubtypeResult(
            primary_subtype=top_prediction["subtype"],
            primary_probability=top_prediction["probability"],
            is_malignant=top_prediction["is_malignant"],
            all_predictions=predictions,
            confidence=top_prediction["probability"]
        )
```

---

## 5. Risk Tier Stratification

```python
class RiskStratifier:
    """
    Assigns clinical risk tier based on all analysis outputs.
    """
    
    TIERS = {
        1: {"name": "CRITICAL", "action": "Immediate referral", "urgency": "24-48 hours"},
        2: {"name": "HIGH", "action": "Urgent referral", "urgency": "1-2 weeks"},
        3: {"name": "MODERATE", "action": "Scheduled appointment", "urgency": "1-3 months"},
        4: {"name": "LOW", "action": "Routine monitoring", "urgency": "Annual check"},
        5: {"name": "BENIGN", "action": "No action required", "urgency": "None"}
    }
    
    def stratify(
        self,
        melanoma: MelanomaResult,
        malignancy: MalignancyResult,
        subtype: SubtypeResult,
        features: ExtractedFeatures
    ) -> RiskTierResult:
        """Assign risk tier."""
        # Calculate composite risk score
        risk_score = self._calculate_risk_score(
            melanoma, malignancy, subtype, features
        )
        
        # Determine tier
        tier = self._score_to_tier(risk_score)
        
        # Get tier details
        tier_info = self.TIERS[tier]
        
        # Generate clinical reasoning
        reasoning = self._generate_reasoning(
            tier, melanoma, malignancy, subtype, features
        )
        
        return RiskTierResult(
            tier=tier,
            tier_name=tier_info["name"],
            risk_score=risk_score,
            action=tier_info["action"],
            urgency=tier_info["urgency"],
            reasoning=reasoning,
            contributing_factors=self._get_contributing_factors(features)
        )
    
    def _calculate_risk_score(
        self,
        melanoma: MelanomaResult,
        malignancy: MalignancyResult,
        subtype: SubtypeResult,
        features: ExtractedFeatures
    ) -> float:
        """Calculate composite risk score (0-100)."""
        score = 0.0
        
        # Melanoma probability (weight: 40%)
        score += melanoma.probability * 40
        
        # Malignancy probability (weight: 25%)
        score += malignancy.malignant_probability * 25
        
        # Subtype risk (weight: 20%)
        if subtype.is_malignant:
            score += subtype.primary_probability * 20
        
        # ABCDE concerning features (weight: 15%)
        abcde_count = sum([
            features.abcde.asymmetry.is_concerning,
            features.abcde.border.is_concerning,
            features.abcde.color.is_concerning,
            features.abcde.diameter.is_concerning,
            features.abcde.evolution.is_concerning
        ])
        score += (abcde_count / 5) * 15
        
        return min(score, 100)
    
    def _score_to_tier(self, score: float) -> int:
        """Convert risk score to tier."""
        if score >= 70:
            return 1  # Critical
        elif score >= 50:
            return 2  # High
        elif score >= 30:
            return 3  # Moderate
        elif score >= 15:
            return 4  # Low
        else:
            return 5  # Benign
```

---

## 6. Output Schemas

```python
@dataclass
class PathologyOutput:
    """Complete pathology analysis output."""
    
    # ABCDE Features
    abcde: ABCDEFeatures
    
    # Classification results
    melanoma: MelanomaResult
    malignancy: MalignancyResult
    subtype: SubtypeResult
    
    # Risk stratification
    risk_tier: RiskTierResult
    
    # Uncertainty
    overall_confidence: float
    uncertainty_flags: List[str]
    
    # Processing info
    model_versions: dict
    processing_time_ms: int
```
