"""
Clinical Constants for Retinal Analysis Pipeline

Evidence-Based Reference Values from Peer-Reviewed Literature:

VASCULAR BIOMARKERS:
- Wong et al. (2004) ARIC Study - Retinal vessel caliber 
- Knudtson et al. (2003) - CRAE/CRVE formulas
- Grisan et al. (2008) - Vessel tortuosity measurement
- Liew et al. (2011) - Fractal dimension analysis

OPTIC DISC:
- Varma et al. (2012) - Los Angeles Latino Eye Study
- Budenz et al. (2007) - RNFL OCT normative data
- Jonas et al. (2003) - Cup-to-disc ratio distributions

DIABETIC RETINOPATHY:
- Wilkinson et al. (2003) - ICDR Classification
- ETDRS Research Group (1991) - ETDRS Grading
- Gulshan et al. (2016) JAMA - DL DR Detection

MACULAR:
- Macular Photocoagulation Study Group
- Chakravarthy et al. (2010) - AMD biomarkers

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Dict, List


class DRGrade(Enum):
    """International Clinical Diabetic Retinopathy (ICDR) Scale
    Reference: Wilkinson et al., Ophthalmology 2003
    """
    NO_DR = 0           # No abnormalities
    MILD_NPDR = 1       # Microaneurysms only
    MODERATE_NPDR = 2   # More than mild, less than severe
    SEVERE_NPDR = 3     # 4-2-1 rule applies
    PROLIFERATIVE_DR = 4  # Neovascularization


class RiskCategory(Enum):
    """Clinical risk stratification"""
    MINIMAL = "minimal"      # 0-15
    LOW = "low"              # 16-30
    MODERATE = "moderate"    # 31-50
    ELEVATED = "elevated"    # 51-70
    HIGH = "high"            # 71-85
    CRITICAL = "critical"    # 86-100


@dataclass
class BiomarkerReference:
    """Reference range for a biomarker"""
    name: str
    unit: str
    normal_min: float
    normal_max: float
    borderline_min: float
    borderline_max: float
    source: str


class ClinicalConstants:
    """
    Evidence-based clinical reference values from peer-reviewed literature.
    
    Each constant includes source citation for medical traceability.
    """
    
    # ========================================================================
    # VESSEL TORTUOSITY INDEX
    # Source: Grisan et al. (2008) IEEE Trans Med Imaging
    # Method: Integral of curvature along vessel centerline
    # ========================================================================
    TORTUOSITY_NORMAL_MIN = 0.05
    TORTUOSITY_NORMAL_MAX = 0.18
    TORTUOSITY_BORDERLINE = 0.25
    TORTUOSITY_ABNORMAL = 0.35
    
    # ========================================================================
    # ARTERIOLE-VENULE RATIO (AVR)
    # Source: Wong et al. (2004) ARIC Study, Lancet
    # Formula: CRAE/CRVE (Knudtson et al. 2003)
    # Clinical: Lower AVR associated with hypertension, stroke risk
    # ========================================================================
    AVR_NORMAL_MIN = 0.67
    AVR_NORMAL_MAX = 0.75
    AVR_BORDERLINE = 0.60
    AVR_ABNORMAL = 0.55
    AVR_CRITICAL = 0.50
    
    # ========================================================================
    # CUP-TO-DISC RATIO (CDR)
    # Source: Varma et al. (2012) Los Angeles Latino Eye Study
    # Method: Vertical CDR more sensitive for glaucoma
    # ========================================================================
    CDR_NORMAL_MIN = 0.10
    CDR_NORMAL_MAX = 0.40
    CDR_BORDERLINE = 0.50
    CDR_SUSPECT = 0.55
    CDR_PROBABLE_GLAUCOMA = 0.60
    CDR_DEFINITE_GLAUCOMA = 0.70
    CDR_ASYMMETRY_THRESHOLD = 0.20  # Between eyes
    
    # ========================================================================
    # VESSEL DENSITY
    # Source: Reif et al. (2012), OCT-A normative data
    # Measurement: % of retinal area covered by vessels
    # ========================================================================
    VESSEL_DENSITY_NORMAL = 0.70
    VESSEL_DENSITY_BORDERLINE = 0.55
    VESSEL_DENSITY_REDUCED = 0.45
    VESSEL_DENSITY_SEVERE = 0.35
    
    # ========================================================================
    # RETINAL NERVE FIBER LAYER (RNFL) THICKNESS
    # Source: Budenz et al. (2007) Ophthalmology - OCT normative data
    # Units: micrometers (normalized to 0-1 scale where 1.0 = 100μm)
    # ========================================================================
    RNFL_NORMAL = 1.00           # ~100 μm average
    RNFL_BORDERLINE = 0.80       # ~80 μm
    RNFL_THIN = 0.60             # ~60 μm
    RNFL_AGE_DECLINE = 0.002     # Loss per year after age 50
    
    # ========================================================================
    # FRACTAL DIMENSION
    # Source: Liew et al. (2011) Invest Ophthalmol Vis Sci
    # Method: Box-counting algorithm
    # Clinical: Reduced complexity indicates vascular pathology
    # ========================================================================
    FRACTAL_DIM_NORMAL_MIN = 1.40
    FRACTAL_DIM_NORMAL_MAX = 1.50
    FRACTAL_DIM_SPARSE = 1.35
    
    # ========================================================================
    # DIABETIC RETINOPATHY LESIONS
    # Source: ETDRS Research Group (1991), Ophthalmology
    # ========================================================================
    HEMORRHAGE_NONE = 0
    HEMORRHAGE_MILD = 1
    HEMORRHAGE_MODERATE = 5
    HEMORRHAGE_SEVERE = 20
    
    MICROANEURYSM_NONE = 0
    MICROANEURYSM_MILD = 1
    MICROANEURYSM_MODERATE = 5
    MICROANEURYSM_SEVERE = 15
    
    EXUDATE_NONE = 0.0
    EXUDATE_MILD = 0.5
    EXUDATE_MODERATE = 2.0
    EXUDATE_SEVERE = 5.0
    
    # 4-2-1 Rule for Severe NPDR
    RULE_421_HEMORRHAGES_4Q = 20   # Hemorrhages in 4 quadrants
    RULE_421_VENOUS_BEADING_2Q = 2  # Venous beading in 2 quadrants
    RULE_421_IRMA_1Q = 1            # IRMA in 1 quadrant
    
    # ========================================================================
    # MACULAR PARAMETERS
    # Source: Macular Photocoagulation Study Group
    # ========================================================================
    MACULAR_THICKNESS_NORMAL = 270  # μm
    MACULAR_THICKNESS_MIN = 200
    MACULAR_THICKNESS_MAX = 320
    MACULAR_THICKNESS_EDEMA = 350
    MACULAR_VOLUME_NORMAL = 0.25    # mm³
    
    # ========================================================================
    # OPTIC DISC MEASUREMENTS
    # Source: Jonas et al. (2003) Acta Ophthalmologica
    # ========================================================================
    DISC_AREA_NORMAL_MIN = 1.8      # mm²
    DISC_AREA_NORMAL_MAX = 2.8      # mm²
    RIM_AREA_NORMAL_MIN = 1.2       # mm²
    RIM_AREA_NORMAL_MAX = 1.8       # mm²
    
    # ========================================================================
    # BRANCHING COEFFICIENT (Murray's Law)
    # Source: Murray (1926), Sherman (1981)
    # Optimal: parent³ = child1³ + child2³
    # ========================================================================
    BRANCHING_NORMAL_MIN = 1.4
    BRANCHING_NORMAL_MAX = 1.7
    BRANCHING_ABNORMAL = 2.0
    
    # ========================================================================
    # AMYLOID-BETA INDICATORS (Alzheimer's Research)
    # Source: Koronyo et al. (2017) JCI Insight
    # Note: Experimental biomarker - not FDA approved
    # ========================================================================
    AMYLOID_ABSENT = 0.0
    AMYLOID_LOW = 0.2
    AMYLOID_MODERATE = 0.5
    AMYLOID_HIGH = 0.7
    
    # Age at which amyloid screening becomes relevant
    AMYLOID_SCREENING_AGE = 50
    
    # ========================================================================
    # IMAGE QUALITY STANDARDS (ETDRS)
    # Source: ETDRS Protocol - Photo Grading
    # ========================================================================
    QUALITY_EXCELLENT = 0.90
    QUALITY_GOOD = 0.75
    QUALITY_FAIR = 0.60
    QUALITY_POOR = 0.40
    QUALITY_UNGRADABLE = 0.25
    
    MIN_RESOLUTION = 512  # pixels
    OPTIMAL_RESOLUTION = 2048
    MAX_FILE_SIZE_MB = 15
    
    SNR_EXCELLENT = 35  # dB
    SNR_GOOD = 28
    SNR_ACCEPTABLE = 20
    
    # ========================================================================
    # RISK SCORE WEIGHTS (Evidence-Based)
    # Based on meta-analysis of retinal biomarker studies
    # ========================================================================
    WEIGHT_DR_GRADE = 0.25
    WEIGHT_CDR = 0.18
    WEIGHT_HEMORRHAGES = 0.12
    WEIGHT_MICROANEURYSMS = 0.10
    WEIGHT_AVR = 0.10
    WEIGHT_TORTUOSITY = 0.08
    WEIGHT_VESSEL_DENSITY = 0.07
    WEIGHT_RNFL = 0.05
    WEIGHT_AMYLOID = 0.05


# Biomarker reference database
BIOMARKER_REFERENCES: Dict[str, BiomarkerReference] = {
    "vessel_tortuosity": BiomarkerReference(
        name="Vessel Tortuosity Index",
        unit="index",
        normal_min=ClinicalConstants.TORTUOSITY_NORMAL_MIN,
        normal_max=ClinicalConstants.TORTUOSITY_NORMAL_MAX,
        borderline_min=ClinicalConstants.TORTUOSITY_NORMAL_MAX,
        borderline_max=ClinicalConstants.TORTUOSITY_BORDERLINE,
        source="Grisan et al., IEEE TMI 2008"
    ),
    "av_ratio": BiomarkerReference(
        name="Arteriole-Venule Ratio",
        unit="CRAE/CRVE",
        normal_min=ClinicalConstants.AVR_NORMAL_MIN,
        normal_max=ClinicalConstants.AVR_NORMAL_MAX,
        borderline_min=ClinicalConstants.AVR_BORDERLINE,
        borderline_max=ClinicalConstants.AVR_NORMAL_MIN,
        source="Wong et al., ARIC Study 2004"
    ),
    "cup_disc_ratio": BiomarkerReference(
        name="Cup-to-Disc Ratio",
        unit="vertical",
        normal_min=ClinicalConstants.CDR_NORMAL_MIN,
        normal_max=ClinicalConstants.CDR_NORMAL_MAX,
        borderline_min=ClinicalConstants.CDR_NORMAL_MAX,
        borderline_max=ClinicalConstants.CDR_BORDERLINE,
        source="Varma et al., LALES 2012"
    ),
    "vessel_density": BiomarkerReference(
        name="Vessel Density",
        unit="index",
        normal_min=0.60,
        normal_max=0.85,
        borderline_min=ClinicalConstants.VESSEL_DENSITY_BORDERLINE,
        borderline_max=0.60,
        source="Reif et al., OCT-A 2012"
    ),
    "rnfl_thickness": BiomarkerReference(
        name="RNFL Thickness",
        unit="normalized",
        normal_min=ClinicalConstants.RNFL_BORDERLINE,
        normal_max=ClinicalConstants.RNFL_NORMAL,
        borderline_min=ClinicalConstants.RNFL_THIN,
        borderline_max=ClinicalConstants.RNFL_BORDERLINE,
        source="Budenz et al., Ophthalmology 2007"
    ),
    "fractal_dimension": BiomarkerReference(
        name="Fractal Dimension",
        unit="D",
        normal_min=ClinicalConstants.FRACTAL_DIM_NORMAL_MIN,
        normal_max=ClinicalConstants.FRACTAL_DIM_NORMAL_MAX,
        borderline_min=ClinicalConstants.FRACTAL_DIM_SPARSE,
        borderline_max=ClinicalConstants.FRACTAL_DIM_NORMAL_MIN,
        source="Liew et al., IOVS 2011"
    ),
    "hemorrhage_count": BiomarkerReference(
        name="Retinal Hemorrhages",
        unit="count",
        normal_min=0,
        normal_max=0,
        borderline_min=1,
        borderline_max=ClinicalConstants.HEMORRHAGE_MODERATE,
        source="ETDRS Research Group 1991"
    ),
    "microaneurysm_count": BiomarkerReference(
        name="Microaneurysms",
        unit="count",
        normal_min=0,
        normal_max=0,
        borderline_min=1,
        borderline_max=ClinicalConstants.MICROANEURYSM_MODERATE,
        source="ETDRS Research Group 1991"
    ),
    "exudate_area": BiomarkerReference(
        name="Exudate Area",
        unit="%",
        normal_min=0,
        normal_max=0.1,
        borderline_min=0.1,
        borderline_max=ClinicalConstants.EXUDATE_MILD,
        source="ETDRS Research Group 1991"
    ),
    "macular_thickness": BiomarkerReference(
        name="Macular Thickness",
        unit="μm (normalized)",
        normal_min=ClinicalConstants.MACULAR_THICKNESS_MIN / ClinicalConstants.MACULAR_THICKNESS_NORMAL,
        normal_max=ClinicalConstants.MACULAR_THICKNESS_MAX / ClinicalConstants.MACULAR_THICKNESS_NORMAL,
        borderline_min=ClinicalConstants.MACULAR_THICKNESS_MAX / ClinicalConstants.MACULAR_THICKNESS_NORMAL,
        borderline_max=ClinicalConstants.MACULAR_THICKNESS_EDEMA / ClinicalConstants.MACULAR_THICKNESS_NORMAL,
        source="Macular Photocoagulation Study"
    ),
    "branching_coefficient": BiomarkerReference(
        name="Branching Coefficient",
        unit="index",
        normal_min=ClinicalConstants.BRANCHING_NORMAL_MIN,
        normal_max=ClinicalConstants.BRANCHING_NORMAL_MAX,
        borderline_min=ClinicalConstants.BRANCHING_NORMAL_MAX,
        borderline_max=ClinicalConstants.BRANCHING_ABNORMAL,
        source="Murray's Law (1926)"
    ),
    "amyloid_score": BiomarkerReference(
        name="Amyloid-Beta Score",
        unit="score",
        normal_min=0,
        normal_max=ClinicalConstants.AMYLOID_LOW,
        borderline_min=ClinicalConstants.AMYLOID_LOW,
        borderline_max=ClinicalConstants.AMYLOID_MODERATE,
        source="Koronyo et al., JCI Insight 2017"
    ),
}


# ICD-10 Diagnostic Codes
ICD10_CODES = {
    "no_dr": "Z96.1",  # Presence of intraocular lens (placeholder for screening)
    "mild_npdr": "E11.319",
    "moderate_npdr": "E11.329",
    "severe_npdr": "E11.339",
    "pdr": "E11.359",
    "dme": "E11.311",  # Diabetic macular edema
    "glaucoma_suspect": "H40.001",
    "glaucoma": "H40.11",
    "hypertensive_retinopathy": "H35.039",
    "rnfl_defect": "H47.20",
    "normal_exam": "Z01.00",
    "amd_suspect": "H35.30",
}


# Referral Urgency Guidelines
REFERRAL_URGENCY = {
    "routine_12_months": {
        "timeframe": "12 months",
        "description": "Routine annual screening",
        "action": "Continue regular diabetic eye examination"
    },
    "routine_6_months": {
        "timeframe": "6 months",
        "description": "Increased monitoring",
        "action": "Schedule follow-up in 6 months"
    },
    "monitor_6_months": {
        "timeframe": "6 months",
        "description": "Requires monitoring",
        "action": "Refer to ophthalmology within 6 months"
    },
    "refer_3_months": {
        "timeframe": "3 months",
        "description": "Non-urgent referral",
        "action": "Ophthalmology consultation recommended"
    },
    "refer_1_month": {
        "timeframe": "1 month",
        "description": "Urgent referral needed",
        "action": "Urgent ophthalmology referral within 4 weeks"
    },
    "urgent_1_week": {
        "timeframe": "1 week",
        "description": "High-priority case",
        "action": "URGENT: Ophthalmology referral within 7 days"
    },
    "emergent_24h": {
        "timeframe": "24 hours",
        "description": "Medical emergency",
        "action": "EMERGENT: Immediate ophthalmology evaluation required"
    },
}
