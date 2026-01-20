"""
Dermatology Pipeline Configuration

Central configuration for the skin lesion analysis pipeline.
"""

from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


# =============================================================================
# FILE AND INPUT CONFIGURATION
# =============================================================================

ACCEPTED_MIME_TYPES = [
    "image/jpeg",
    "image/png", 
    "image/heic",
    "image/heif",
    "application/dicom"
]

ACCEPTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".heic", ".heif", ".dcm"]

MAX_FILE_SIZE_MB = 50
MIN_FILE_SIZE_KB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_FILE_SIZE_BYTES = MIN_FILE_SIZE_KB * 1024

# =============================================================================
# IMAGE QUALITY THRESHOLDS
# =============================================================================

@dataclass
class QualityThresholds:
    """Image quality validation thresholds."""
    # Resolution - relaxed for broader compatibility
    min_resolution_mp: float = 0.1  # Minimum megapixels (was 0.3)
    optimal_resolution_mp: float = 2.0
    min_width: int = 200  # Was 640
    min_height: int = 200  # Was 480
    
    # Focus/Sharpness (Laplacian variance) - relaxed
    min_focus_score: float = 10.0  # Was 30.0
    optimal_focus_score: float = 100.0
    
    # Brightness (0-255 scale)
    min_brightness: int = 20  # Was 40
    max_brightness: int = 240  # Was 220
    optimal_brightness_range: Tuple[int, int] = (80, 180)
    
    # Exposure - relaxed
    max_overexposed_ratio: float = 0.25  # Was 0.15
    max_underexposed_ratio: float = 0.30  # Was 0.20
    
    # Illumination uniformity (0-1)
    min_uniformity: float = 0.20  # Was 0.40
    optimal_uniformity: float = 0.80
    
    # Color cast (max channel difference)
    max_color_cast: float = 80.0  # Was 50.0
    warning_color_cast: float = 40.0  # Was 30.0
    
    # Skin detection - relaxed
    min_skin_ratio: float = 0.05  # Was 0.10
    optimal_skin_ratio: float = 0.30


QUALITY_THRESHOLDS = QualityThresholds()

# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================

@dataclass
class PreprocessingConfig:
    """Preprocessing stage configuration."""
    # Color constancy
    color_constancy_norm_order: int = 6  # Shades of Gray
    min_color_constancy_confidence: float = 0.30
    
    # CLAHE
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)
    
    # Hair removal
    hair_kernel_size: int = 17
    inpaint_radius: int = 5
    max_hair_coverage: float = 0.40
    
    # Resize
    target_size: Tuple[int, int] = (512, 512)
    preserve_aspect: bool = True
    padding_color: Tuple[int, int, int] = (0, 0, 0)


PREPROCESSING_CONFIG = PreprocessingConfig()

# =============================================================================
# SEGMENTATION CONFIGURATION
# =============================================================================

@dataclass
class SegmentationConfig:
    """Segmentation module configuration."""
    # Detection
    detection_confidence_threshold: float = 0.50
    detection_nms_threshold: float = 0.40
    
    # Segmentation
    segmentation_threshold: float = 0.50
    min_lesion_ratio: float = 0.001
    max_lesion_ratio: float = 0.80
    
    # Geometry
    min_diameter_mm: float = 1.0
    max_diameter_mm: float = 100.0
    min_circularity: float = 0.10
    
    # CRF refinement
    crf_iterations: int = 5
    
    # Calibration (pixels per mm, adjustable)
    default_pixels_per_mm: float = 10.0


SEGMENTATION_CONFIG = SegmentationConfig()

# =============================================================================
# CLASSIFICATION CONFIGURATION
# =============================================================================

@dataclass
class ClassificationConfig:
    """Classification module configuration."""
    # Model settings
    model_input_size: Tuple[int, int] = (512, 512)
    
    # Melanoma thresholds (conservative for safety)
    melanoma_critical_threshold: float = 0.70
    melanoma_high_threshold: float = 0.40
    melanoma_moderate_threshold: float = 0.20
    melanoma_low_threshold: float = 0.10
    
    # Malignancy threshold
    malignancy_threshold: float = 0.50
    
    # Confidence requirements
    min_classification_confidence: float = 0.50
    min_melanoma_positive_confidence: float = 0.80
    
    # Ensemble
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "efficientnet": 0.40,
        "vit": 0.35,
        "resnet": 0.25
    })


CLASSIFICATION_CONFIG = ClassificationConfig()

# =============================================================================
# ABCDE ANALYSIS CONFIGURATION
# =============================================================================

@dataclass
class ABCDEConfig:
    """ABCDE criteria analysis configuration."""
    # Asymmetry
    asymmetry_concerning_threshold: float = 0.40
    
    # Border
    border_irregularity_threshold: float = 0.50
    min_fractal_dimension: float = 1.0
    max_fractal_dimension: float = 2.0
    
    # Color
    max_expected_colors: int = 6
    concerning_color_count: int = 3
    
    # Diameter (mm)
    diameter_threshold_mm: float = 6.0
    
    # Evolution
    evolution_concerning_threshold: float = 0.40
    
    # Weights for combined scoring
    weights: Dict[str, float] = field(default_factory=lambda: {
        "asymmetry": 1.3,
        "border": 1.0,
        "color": 1.3,
        "diameter": 0.8,
        "evolution": 1.5
    })


ABCDE_CONFIG = ABCDEConfig()

# =============================================================================
# RISK STRATIFICATION
# =============================================================================

@dataclass
class RiskTierConfig:
    """Risk tier configuration."""
    tier_1_threshold: float = 70.0  # Critical
    tier_2_threshold: float = 50.0  # High
    tier_3_threshold: float = 30.0  # Moderate
    tier_4_threshold: float = 15.0  # Low
    # Below tier_4 = Benign (Tier 5)


RISK_CONFIG = RiskTierConfig()

RISK_TIERS = {
    1: {"name": "CRITICAL", "action": "Immediate referral", "urgency": "24-48 hours"},
    2: {"name": "HIGH", "action": "Urgent referral", "urgency": "1-2 weeks"},
    3: {"name": "MODERATE", "action": "Scheduled appointment", "urgency": "1-3 months"},
    4: {"name": "LOW", "action": "Routine monitoring", "urgency": "Annual check"},
    5: {"name": "BENIGN", "action": "No action required", "urgency": "None"}
}

# =============================================================================
# LESION SUBTYPES
# =============================================================================

LESION_SUBTYPES = [
    "melanoma",
    "basal_cell_carcinoma",
    "squamous_cell_carcinoma",
    "actinic_keratosis",
    "benign_keratosis",
    "dermatofibroma",
    "nevus",
    "vascular_lesion"
]

MALIGNANT_SUBTYPES = [
    "melanoma",
    "basal_cell_carcinoma",
    "squamous_cell_carcinoma"
]

# =============================================================================
# CONCERNING COLORS (Dermoscopy)
# =============================================================================

CONCERNING_COLORS = {
    "white": {"lower": [200, 200, 200], "upper": [255, 255, 255], "significance": "regression"},
    "red": {"lower": [150, 50, 50], "upper": [255, 100, 100], "significance": "vascularization"},
    "light_brown": {"lower": [139, 90, 43], "upper": [180, 140, 80], "significance": "melanin_superficial"},
    "dark_brown": {"lower": [65, 40, 20], "upper": [120, 80, 50], "significance": "melanin_deep"},
    "blue_gray": {"lower": [80, 80, 100], "upper": [140, 140, 160], "significance": "melanin_very_deep"},
    "black": {"lower": [0, 0, 0], "upper": [40, 40, 40], "significance": "melanin_aggregation"}
}

# =============================================================================
# PIPELINE TIMEOUTS
# =============================================================================

STAGE_TIMEOUTS_MS = {
    "validation": 500,
    "preprocessing": 1000,
    "segmentation": 2000,
    "feature_extraction": 1000,
    "classification": 1500,
    "scoring": 500,
    "explanation": 5000,
    "formatting": 500
}

TOTAL_TIMEOUT_MS = 30000
MAX_RETRIES = 2

# =============================================================================
# MODEL PATHS
# =============================================================================

MODELS_DIR = Path(__file__).parent / "models"

MODEL_PATHS = {
    "segmentation": MODELS_DIR / "unet_effb4_lesion.pth",
    "classification_efficientnet": MODELS_DIR / "effnet_b4_isic.pth",
    "classification_vit": MODELS_DIR / "vit_derm.pth",
    "detection": MODELS_DIR / "yolo_lesion.pth"
}

# =============================================================================
# DISCLAIMERS
# =============================================================================

MANDATORY_DISCLAIMER = """
IMPORTANT NOTICE: This AI skin analysis is for informational screening 
purposes only and is NOT a medical diagnosis. This tool cannot and does 
not provide a definitive diagnosis of any skin condition, including 
skin cancer.

Only a qualified healthcare provider (dermatologist or physician) can 
diagnose skin conditions after proper clinical examination, which may 
include dermoscopy, biopsy, and histopathological analysis.

If you have concerns about any skin lesion, especially if it is 
changing, bleeding, itching, or has any unusual characteristics, 
please consult a healthcare provider promptly.
"""

CRITICAL_DISCLAIMER = """
URGENT: This analysis has identified features that require prompt 
medical attention. This is NOT a diagnosis, but we strongly recommend 
you contact a dermatologist or healthcare provider within the next 
24-48 hours for professional evaluation.

Do not delay seeking medical care based on any AI analysis.
"""
