"""
Cardiology Pipeline - Constants
Static reference values and enumerations.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ==============================================================================
# ENUMERATIONS
# ==============================================================================

class RiskCategory(str, Enum):
    """Risk level categories."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class RhythmType(str, Enum):
    """Cardiac rhythm classifications."""
    NORMAL_SINUS = "Normal Sinus Rhythm"
    SINUS_BRADY = "Sinus Bradycardia"
    SINUS_TACHY = "Sinus Tachycardia"
    AFIB = "Atrial Fibrillation"
    AFLUTTER = "Atrial Flutter"
    PVC = "Premature Ventricular Contraction"
    PAC = "Premature Atrial Contraction"
    UNKNOWN = "Unknown Rhythm"


class EchoView(str, Enum):
    """Echocardiographic views."""
    PLAX = "PLAX"  # Parasternal Long Axis
    PSAX = "PSAX"  # Parasternal Short Axis
    A4C = "A4C"    # Apical 4-Chamber
    A2C = "A2C"    # Apical 2-Chamber
    A3C = "A3C"    # Apical 3-Chamber
    SUBCOSTAL = "SUBCOSTAL"
    SUPRASTERNAL = "SUPRASTERNAL"
    UNKNOWN = "UNKNOWN"


class Severity(str, Enum):
    """Clinical severity levels."""
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class PipelineStage(str, Enum):
    """Pipeline execution stages."""
    RECEIPT = "RECEIPT"
    VALIDATION = "VALIDATION"
    PREPROCESSING = "PREPROCESSING"
    DETECTION = "DETECTION"
    ANALYSIS = "ANALYSIS"
    FUSION = "FUSION"
    SCORING = "SCORING"
    FORMATTING = "FORMATTING"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class Modality(str, Enum):
    """Input modalities."""
    ECG_SIGNAL = "ecg_signal"
    ECHO_IMAGE = "echo_image"
    ECHO_VIDEO = "echo_video"
    CLINICAL_METADATA = "clinical_metadata"


class AutonomicState(str, Enum):
    """Autonomic nervous system state."""
    BALANCED = "balanced"
    PARASYMPATHETIC_DOMINANT = "parasympathetic_dominant"
    SYMPATHETIC_DOMINANT = "sympathetic_dominant"
    LOW_OVERALL = "low_overall"


# ==============================================================================
# CLINICAL REFERENCE VALUES
# ==============================================================================

@dataclass(frozen=True)
class ClinicalConstants:
    """Clinical reference constants."""
    
    # Heart rate boundaries
    BRADYCARDIA_THRESHOLD: int = 60
    TACHYCARDIA_THRESHOLD: int = 100
    SEVERE_BRADY_THRESHOLD: int = 50
    SEVERE_TACHY_THRESHOLD: int = 120
    CRITICAL_BRADY_THRESHOLD: int = 40
    CRITICAL_TACHY_THRESHOLD: int = 150
    
    # HRV thresholds
    RMSSD_LOW_THRESHOLD: float = 20.0
    RMSSD_HIGH_THRESHOLD: float = 60.0
    SDNN_LOW_THRESHOLD: float = 40.0
    
    # Arrhythmia detection
    AFIB_RR_CV_THRESHOLD: float = 0.15
    IRREGULAR_RR_CV_THRESHOLD: float = 0.10
    
    # EF thresholds
    EF_NORMAL_MIN: float = 55.0
    EF_MILD_REDUCTION_MIN: float = 45.0
    EF_MOD_REDUCTION_MIN: float = 30.0
    
    # Interval thresholds (ms)
    PR_PROLONGED_THRESHOLD: int = 200
    QRS_WIDE_THRESHOLD: int = 120
    QTC_PROLONGED_THRESHOLD: int = 460
    QTC_PROLONGED_FEMALE_THRESHOLD: int = 470
    
    # Quality thresholds
    MIN_R_PEAKS_FOR_HRV: int = 5
    MIN_CARDIAC_CYCLES: int = 3
    MIN_SNR_DB: float = 5.0


# Singleton instance
CLINICAL = ClinicalConstants()


# ==============================================================================
# BIOMARKER DEFINITIONS
# ==============================================================================

@dataclass
class BiomarkerDefinition:
    """Definition of a cardiac biomarker."""
    name: str
    friendly_name: str
    unit: str
    normal_range: Tuple[float, float]
    abnormal_low: Optional[float]
    abnormal_high: Optional[float]
    clinical_meaning: str
    source: str  # "ecg" or "echo"


BIOMARKER_DEFINITIONS: Dict[str, BiomarkerDefinition] = {
    # ECG-derived biomarkers
    "heart_rate_bpm": BiomarkerDefinition(
        name="heart_rate_bpm",
        friendly_name="Heart Rate",
        unit="bpm",
        normal_range=(60, 100),
        abnormal_low=50,
        abnormal_high=110,
        clinical_meaning="Resting cardiac rate",
        source="ecg",
    ),
    "rmssd_ms": BiomarkerDefinition(
        name="rmssd_ms",
        friendly_name="RMSSD",
        unit="ms",
        normal_range=(25, 60),
        abnormal_low=20,
        abnormal_high=100,
        clinical_meaning="Parasympathetic nervous system activity",
        source="ecg",
    ),
    "sdnn_ms": BiomarkerDefinition(
        name="sdnn_ms",
        friendly_name="SDNN",
        unit="ms",
        normal_range=(50, 120),
        abnormal_low=40,
        abnormal_high=None,
        clinical_meaning="Overall heart rate variability",
        source="ecg",
    ),
    "pnn50_percent": BiomarkerDefinition(
        name="pnn50_percent",
        friendly_name="pNN50",
        unit="%",
        normal_range=(10, 30),
        abnormal_low=5,
        abnormal_high=40,
        clinical_meaning="High-frequency HRV component",
        source="ecg",
    ),
    "pr_interval_ms": BiomarkerDefinition(
        name="pr_interval_ms",
        friendly_name="PR Interval",
        unit="ms",
        normal_range=(120, 200),
        abnormal_low=None,
        abnormal_high=200,
        clinical_meaning="AV conduction time",
        source="ecg",
    ),
    "qrs_duration_ms": BiomarkerDefinition(
        name="qrs_duration_ms",
        friendly_name="QRS Duration",
        unit="ms",
        normal_range=(80, 120),
        abnormal_low=None,
        abnormal_high=120,
        clinical_meaning="Ventricular depolarization time",
        source="ecg",
    ),
    "qtc_ms": BiomarkerDefinition(
        name="qtc_ms",
        friendly_name="Corrected QT",
        unit="ms",
        normal_range=(350, 450),
        abnormal_low=None,
        abnormal_high=460,
        clinical_meaning="Rate-corrected QT interval",
        source="ecg",
    ),
    
    # Echo-derived biomarkers
    "ejection_fraction_percent": BiomarkerDefinition(
        name="ejection_fraction_percent",
        friendly_name="Ejection Fraction",
        unit="%",
        normal_range=(55, 70),
        abnormal_low=40,
        abnormal_high=None,
        clinical_meaning="LV systolic function",
        source="echo",
    ),
    "lv_end_diastolic_volume_ml": BiomarkerDefinition(
        name="lv_end_diastolic_volume_ml",
        friendly_name="LV End-Diastolic Volume",
        unit="mL",
        normal_range=(70, 150),
        abnormal_low=None,
        abnormal_high=200,
        clinical_meaning="LV volume at end of filling",
        source="echo",
    ),
    "lv_end_systolic_volume_ml": BiomarkerDefinition(
        name="lv_end_systolic_volume_ml",
        friendly_name="LV End-Systolic Volume",
        unit="mL",
        normal_range=(25, 60),
        abnormal_low=None,
        abnormal_high=90,
        clinical_meaning="LV volume at end of contraction",
        source="echo",
    ),
}


# ==============================================================================
# WALL SEGMENT DEFINITIONS
# ==============================================================================

WALL_SEGMENTS = {
    # 17-segment model
    "basal": ["basal_anterior", "basal_anteroseptal", "basal_inferoseptal",
              "basal_inferior", "basal_inferolateral", "basal_anterolateral"],
    "mid": ["mid_anterior", "mid_anteroseptal", "mid_inferoseptal",
            "mid_inferior", "mid_inferolateral", "mid_anterolateral"],
    "apical": ["apical_anterior", "apical_septal", 
               "apical_inferior", "apical_lateral"],
    "apex": ["apex"],
}

CORONARY_TERRITORIES = {
    "LAD": ["basal_anterior", "basal_anteroseptal", "mid_anterior", 
            "mid_anteroseptal", "apical_anterior", "apical_septal", "apex"],
    "RCA": ["basal_inferior", "basal_inferoseptal", "mid_inferior", 
            "mid_inferoseptal", "apical_inferior"],
    "LCx": ["basal_anterolateral", "basal_inferolateral", 
            "mid_anterolateral", "mid_inferolateral", "apical_lateral"],
}


# ==============================================================================
# FINDING TEMPLATES
# ==============================================================================

FINDING_TEMPLATES = {
    "normal_sinus_rhythm": {
        "title": "Normal Sinus Rhythm",
        "severity": Severity.NORMAL,
        "description": "Regular rhythm with heart rate 60-100 bpm",
        "source": "ecg",
    },
    "sinus_bradycardia": {
        "title": "Sinus Bradycardia",
        "severity": Severity.MILD,
        "description": "Heart rate below 60 bpm with regular rhythm",
        "source": "ecg",
    },
    "sinus_tachycardia": {
        "title": "Sinus Tachycardia",
        "severity": Severity.MILD,
        "description": "Heart rate above 100 bpm with regular rhythm",
        "source": "ecg",
    },
    "atrial_fibrillation_suspected": {
        "title": "Atrial Fibrillation Suspected",
        "severity": Severity.MODERATE,
        "description": "Irregularly irregular rhythm detected",
        "source": "ecg",
    },
    "low_hrv": {
        "title": "Low Heart Rate Variability",
        "severity": Severity.MILD,
        "description": "Reduced HRV may indicate stress or reduced vagal tone",
        "source": "ecg",
    },
    "normal_ef": {
        "title": "Normal Ejection Fraction",
        "severity": Severity.NORMAL,
        "description": "EF >= 55% indicates normal systolic function",
        "source": "echo",
    },
    "reduced_ef": {
        "title": "Reduced Ejection Fraction",
        "severity": Severity.MODERATE,
        "description": "EF < 55% indicates reduced systolic function",
        "source": "echo",
    },
}


# ==============================================================================
# SYMPTOMS DICTIONARY
# ==============================================================================

VALID_SYMPTOMS = {
    "chest_pain": "Chest pain or discomfort",
    "dyspnea": "Shortness of breath",
    "palpitations": "Heart palpitations",
    "fatigue": "Unusual fatigue",
    "syncope": "Fainting or near-fainting",
    "edema": "Swelling in legs/ankles",
    "dizziness": "Dizziness or lightheadedness",
    "orthopnea": "Difficulty breathing when lying down",
}


# ==============================================================================
# TIME CONSTANTS
# ==============================================================================

TIMEOUT_SECONDS = {
    "validation": 30,
    "preprocessing": 60,
    "detection": 120,
    "analysis": 90,
    "fusion": 30,
    "total_pipeline": 300,
}

RETRY_CONFIG = {
    "max_attempts": 3,
    "backoff_factor": 2.0,
    "initial_delay_sec": 1.0,
}
