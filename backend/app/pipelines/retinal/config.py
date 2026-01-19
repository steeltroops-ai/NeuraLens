"""
Retinal Analysis Pipeline - Configuration
Clinical constants and processing parameters matching speech pipeline structure.

References:
- ETDRS Research Group (1991) - Image quality standards
- Wilkinson et al. (2003) - ICDR grading scale
- Wong TY et al. (2004) - Vessel biomarker reference values
- Jonas JB et al. (2003) - Optic disc parameters

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

from typing import Dict, Tuple, List
from enum import Enum


# =============================================================================
# INPUT CONSTRAINTS
# =============================================================================

INPUT_CONSTRAINTS = {
    "max_file_size_mb": 15,
    "min_resolution": 512,
    "recommended_resolution": 1024,
    "max_resolution": 8192,
    "target_channels": 3,  # RGB
    "model_input_size": 512,
    "supported_formats": ["jpeg", "jpg", "png", "tiff", "tif"]
}

# Supported MIME types
SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg", 
    "image/png",
    "image/tiff",
    "image/x-tiff"
}


# =============================================================================
# QUALITY THRESHOLDS (ETDRS Standards)
# =============================================================================

QUALITY_THRESHOLDS = {
    "excellent": 0.80,
    "good": 0.60,
    "fair": 0.40,
    "poor": 0.20,
    "ungradable": 0.0
}

# Minimum quality for analysis
MIN_QUALITY_SCORE = 0.30
RECOMMENDED_QUALITY_SCORE = 0.60


# =============================================================================
# BIOMARKER NORMAL RANGES (Peer-reviewed literature)
# =============================================================================

BIOMARKER_NORMAL_RANGES: Dict[str, Tuple[float, float]] = {
    # Optic Disc Parameters
    "cup_disc_ratio": (0.1, 0.4),          # Normal CDR range
    "disc_area_mm2": (1.5, 3.0),           # Normal disc area
    "rim_area_mm2": (1.0, 2.0),            # Normal neuroretinal rim
    "rnfl_thickness_um": (80, 120),        # Retinal nerve fiber layer
    
    # Vessel Parameters  
    "av_ratio": (0.65, 0.75),              # Arteriole-to-venule ratio
    "crae_um": (140, 180),                 # Central retinal artery equivalent
    "crve_um": (200, 250),                 # Central retinal vein equivalent
    "tortuosity_index": (1.0, 1.15),       # Vessel straightness
    "vessel_density": (0.60, 0.85),        # % of vessel pixels
    "fractal_dimension": (1.35, 1.45),     # Vascular complexity
    "branching_angle_deg": (70, 85),       # Optimal bifurcation angle
    
    # Lesion Counts (healthy = 0)
    "microaneurysm_count": (0, 0),
    "hemorrhage_count": (0, 0),
    "exudate_area_percent": (0.0, 1.0),
    "cotton_wool_spots": (0, 0)
}


# =============================================================================
# ABNORMAL THRESHOLDS (Trigger clinical concern)
# =============================================================================

BIOMARKER_ABNORMAL_THRESHOLDS: Dict[str, float] = {
    # Optic Disc - Glaucoma indicators
    "cup_disc_ratio_high": 0.6,            # Suspect glaucoma
    "cup_disc_ratio_critical": 0.8,        # High glaucoma risk
    "rnfl_thickness_low": 80,              # Below normal
    "rnfl_thickness_critical": 60,         # Severe loss
    
    # Vessel - Hypertensive indicators
    "av_ratio_low": 0.60,                  # Arterial narrowing
    "av_ratio_critical": 0.50,             # Severe narrowing
    "tortuosity_high": 1.2,                # Abnormal tortuosity
    
    # Lesions - DR indicators
    "microaneurysm_mild": 1,               # Mild NPDR threshold
    "hemorrhage_moderate": 5,              # Moderate NPDR
    "exudate_area_severe": 3.0,            # Severe exudation %
}


# =============================================================================
# DIABETIC RETINOPATHY GRADING (ICDR Scale)
# =============================================================================

class DRGrade(int, Enum):
    """ICDR Diabetic Retinopathy Grades"""
    NO_DR = 0
    MILD_NPDR = 1
    MODERATE_NPDR = 2
    SEVERE_NPDR = 3
    PROLIFERATIVE_DR = 4


DR_GRADE_CRITERIA = {
    DRGrade.NO_DR: {
        "description": "No apparent retinopathy",
        "criteria": "No visible signs of DR",
        "referral": "Routine 12 months"
    },
    DRGrade.MILD_NPDR: {
        "description": "Mild nonproliferative DR",
        "criteria": "Microaneurysms only",
        "referral": "Routine 12 months"
    },
    DRGrade.MODERATE_NPDR: {
        "description": "Moderate nonproliferative DR",
        "criteria": "More than just MA, less than severe NPDR",
        "referral": "Follow-up 6 months"
    },
    DRGrade.SEVERE_NPDR: {
        "description": "Severe nonproliferative DR",
        "criteria": "4-2-1 rule: hemorrhages 4 quadrants, VB 2 quadrants, or IRMA 1 quadrant",
        "referral": "Refer within 2 weeks"
    },
    DRGrade.PROLIFERATIVE_DR: {
        "description": "Proliferative DR",
        "criteria": "Neovascularization or vitreous/preretinal hemorrhage",
        "referral": "Urgent referral"
    }
}


# =============================================================================
# RISK WEIGHTS (Derived from clinical literature)
# =============================================================================

RISK_WEIGHTS: Dict[str, float] = {
    "dr_grade": 0.40,
    "cup_disc_ratio": 0.25,
    "vessel_abnormality": 0.20,
    "lesion_burden": 0.15
}

# Risk categories (score ranges)
RISK_CATEGORIES = {
    (0, 25): "low",
    (25, 50): "moderate",
    (50, 75): "high",
    (75, 100): "critical"
}


# =============================================================================
# CONDITION PATTERNS (For multi-condition scoring)
# =============================================================================

CONDITION_PATTERNS = {
    "diabetic_retinopathy": {
        "microaneurysms": 0.30,
        "hemorrhages": 0.25,
        "exudates": 0.20,
        "vessel_changes": 0.15,
        "neovascularization": 0.10
    },
    "glaucoma": {
        "cup_disc_ratio": 0.40,
        "rnfl_thickness": 0.25,
        "rim_notching": 0.20,
        "disc_asymmetry": 0.15
    },
    "hypertensive_retinopathy": {
        "av_ratio": 0.35,
        "av_nicking": 0.25,
        "arterial_narrowing": 0.25,
        "copper_wiring": 0.15
    },
    "amd": {
        "drusen": 0.40,
        "rpe_changes": 0.30,
        "cnv": 0.20,
        "geographic_atrophy": 0.10
    }
}


# =============================================================================
# ICD-10 DIAGNOSTIC CODES
# =============================================================================

ICD10_CODES = {
    # Diabetic Retinopathy
    "E11.319": "Type 2 diabetes mellitus with unspecified diabetic retinopathy without macular edema",
    "E11.329": "Type 2 diabetes mellitus with mild nonproliferative diabetic retinopathy without macular edema",
    "E11.339": "Type 2 diabetes mellitus with moderate nonproliferative diabetic retinopathy without macular edema",
    "E11.349": "Type 2 diabetes mellitus with severe nonproliferative diabetic retinopathy without macular edema",
    "E11.359": "Type 2 diabetes mellitus with proliferative diabetic retinopathy without macular edema",
    "E11.321": "Type 2 diabetes mellitus with mild nonproliferative diabetic retinopathy with macular edema",
    
    # Glaucoma
    "H40.10X0": "Unspecified open-angle glaucoma, stage unspecified",
    "H40.11X1": "Primary open-angle glaucoma, mild stage",
    "H40.11X2": "Primary open-angle glaucoma, moderate stage",
    "H40.11X3": "Primary open-angle glaucoma, severe stage",
    
    # AMD
    "H35.30": "Unspecified macular degeneration",
    "H35.31": "Nonexudative age-related macular degeneration",
    "H35.32": "Exudative age-related macular degeneration",
    
    # Hypertensive Retinopathy  
    "H35.031": "Hypertensive retinopathy, right eye",
    "H35.032": "Hypertensive retinopathy, left eye",
    "H35.033": "Hypertensive retinopathy, bilateral",
}


# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

PROCESSING_CONFIG = {
    # CLAHE parameters
    "clahe_clip_limit": 2.0,
    "clahe_tile_size": (8, 8),
    
    # Color normalization (LAB space)
    "target_l_mean": 128.0,
    "target_l_std": 50.0,
    
    # Fundus detection
    "min_fundus_coverage": 0.40,
    "min_vessel_density": 0.02,
    "red_channel_dominance": 0.80,
    
    # Model settings
    "model_backbone": "efficientnet_b4",
    "model_pretrained": True,
    "model_num_classes": 5,  # DR grades 0-4
    
    # Inference
    "batch_size": 1,
    "use_gpu": True,
    "confidence_threshold": 0.5
}


# =============================================================================
# RECOMMENDATION TEMPLATES
# =============================================================================

RECOMMENDATIONS = {
    "low_risk": [
        "No sight-threatening abnormalities detected",
        "Continue routine annual screening",
        "Maintain diabetes control if applicable",
        "No immediate referral required"
    ],
    "moderate_risk": [
        "Mild to moderate changes observed",
        "Follow-up examination recommended in 6 months",
        "Optimize blood sugar and blood pressure control",
        "Consult ophthalmologist if symptoms develop"
    ],
    "high_risk": [
        "Significant abnormalities detected",
        "Ophthalmology referral recommended within 2 weeks",
        "Additional imaging may be required",
        "Monitor for vision changes"
    ],
    "critical_risk": [
        "Sight-threatening findings detected",
        "Urgent ophthalmology referral required",
        "Seek specialist care within 24-48 hours",
        "Do not delay due to schedule availability"
    ]
}


# =============================================================================
# SAFETY DISCLAIMERS
# =============================================================================

DISCLAIMERS = {
    "screening": (
        "This AI system is intended for screening purposes only and does not "
        "provide a clinical diagnosis. All findings should be reviewed by a "
        "qualified ophthalmologist or optometrist before clinical decisions are made."
    ),
    "false_negative": (
        "Negative results do not rule out the presence of disease. Patients with "
        "symptoms or risk factors should receive comprehensive eye examination "
        "regardless of AI screening results."
    ),
    "emergency": (
        "In case of sudden vision loss, flashes, floaters, or other acute symptoms, "
        "seek immediate medical attention. Do not rely on AI screening for emergencies."
    ),
    "not_fda_cleared": (
        "This system has not been cleared or approved by the FDA for autonomous "
        "diagnostic use. It is intended as a clinical decision support tool only."
    )
}
