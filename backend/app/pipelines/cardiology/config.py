"""
Cardiology Pipeline - Configuration
Clinical thresholds, input constraints, and reference values.
"""

from typing import Dict, Tuple, Any
from dataclasses import dataclass


# ==============================================================================
# INPUT CONSTRAINTS
# ==============================================================================

INPUT_CONSTRAINTS = {
    # ECG Signal Constraints
    "ecg": {
        "min_sample_rate_hz": 100,
        "max_sample_rate_hz": 1000,
        "optimal_sample_rate_hz": 500,
        "min_duration_sec": 5,
        "max_duration_sec": 300,
        "min_r_peaks": 5,
        "amplitude_range_mv": (-5.0, 5.0),
        "max_file_size_mb": 5,
        "allowed_formats": [".csv", ".json", ".txt"],
    },
    
    # Echo Image Constraints
    "echo_image": {
        "min_resolution": 256,
        "max_resolution": 4096,
        "max_file_size_mb": 10,
        "max_images_per_request": 20,
        "allowed_formats": [".jpg", ".jpeg", ".png"],
        "color_modes": ["RGB", "L"],  # RGB or Grayscale
    },
    
    # Echo Video Constraints
    "echo_video": {
        "min_resolution": 256,
        "max_resolution": 1920,
        "min_frame_rate": 15,
        "max_frame_rate": 120,
        "min_duration_sec": 1,
        "max_duration_sec": 60,
        "max_file_size_mb": 100,
        "allowed_formats": [".mp4", ".avi", ".dcm"],
        "allowed_codecs": ["h264", "h265", "mpeg4"],
    },
    
    # Metadata Constraints
    "metadata": {
        "max_file_size_kb": 100,
        "age_range": (0, 120),
        "weight_range_kg": (1, 500),
        "height_range_cm": (30, 250),
        "systolic_bp_range": (60, 300),
        "diastolic_bp_range": (30, 200),
        "valid_sex": ["male", "female", "other"],
    },
}


# ==============================================================================
# QUALITY THRESHOLDS
# ==============================================================================

QUALITY_THRESHOLDS = {
    # ECG Quality
    "ecg": {
        "min_snr_db": 5.0,
        "good_snr_db": 10.0,
        "excellent_snr_db": 15.0,
        "max_flatline_sec": 2.0,
        "max_missing_ratio": 0.05,
        "max_clipping_ratio": 0.01,
        "min_usable_ratio": 0.50,
        "good_usable_ratio": 0.70,
        "excellent_usable_ratio": 0.90,
    },
    
    # Echo Quality
    "echo": {
        "min_frame_quality": 0.30,
        "good_frame_quality": 0.60,
        "excellent_frame_quality": 0.85,
        "min_view_confidence": 0.40,
        "good_view_confidence": 0.60,
        "excellent_view_confidence": 0.85,
        "min_frames_usable_ratio": 0.30,
        "good_frames_usable_ratio": 0.50,
        "min_cardiac_cycles": 1,
        "recommended_cardiac_cycles": 3,
    },
}


# ==============================================================================
# HRV NORMAL RANGES (Time Domain)
# ==============================================================================

HRV_NORMAL_RANGES = {
    "heart_rate_bpm": {
        "low": 60,
        "high": 100,
        "abnormal_low": 50,
        "abnormal_high": 110,
        "unit": "bpm",
        "clinical_meaning": "Resting heart rate",
    },
    "rmssd_ms": {
        "low": 25,
        "high": 60,
        "abnormal_low": 20,
        "abnormal_high": 100,
        "unit": "ms",
        "clinical_meaning": "Parasympathetic (vagal) activity",
    },
    "sdnn_ms": {
        "low": 50,
        "high": 120,
        "abnormal_low": 40,
        "abnormal_high": 150,
        "unit": "ms",
        "clinical_meaning": "Overall HRV / total variability",
    },
    "pnn50_percent": {
        "low": 10,
        "high": 30,
        "abnormal_low": 5,
        "abnormal_high": 40,
        "unit": "%",
        "clinical_meaning": "High-frequency HRV component",
    },
    "mean_rr_ms": {
        "low": 600,
        "high": 1000,
        "abnormal_low": 500,
        "abnormal_high": 1200,
        "unit": "ms",
        "clinical_meaning": "Average RR interval",
    },
    "sdsd_ms": {
        "low": 20,
        "high": 50,
        "abnormal_low": 15,
        "abnormal_high": 70,
        "unit": "ms",
        "clinical_meaning": "Short-term HRV variation",
    },
    "cv_rr_percent": {
        "low": 3,
        "high": 8,
        "abnormal_low": 2,
        "abnormal_high": 15,
        "unit": "%",
        "clinical_meaning": "Coefficient of variation of RR",
    },
}


# ==============================================================================
# ECG INTERVAL NORMAL RANGES
# ==============================================================================

INTERVAL_NORMAL_RANGES = {
    "pr_interval_ms": {
        "low": 120,
        "high": 200,
        "abnormal_high": 200,  # 1st degree AV block
        "unit": "ms",
        "clinical_meaning": "AV conduction time",
    },
    "qrs_duration_ms": {
        "low": 80,
        "high": 120,
        "abnormal_high": 120,  # Bundle branch block
        "unit": "ms",
        "clinical_meaning": "Ventricular depolarization",
    },
    "qt_interval_ms": {
        "low": 350,
        "high": 450,
        "abnormal_high": 460,  # Long QT
        "unit": "ms",
        "clinical_meaning": "Total ventricular activity",
    },
    "qtc_ms": {
        "low": 350,
        "high": 450,
        "abnormal_high": 460,  # Long QT syndrome
        "abnormal_high_female": 470,
        "unit": "ms",
        "clinical_meaning": "Rate-corrected QT interval",
    },
}


# ==============================================================================
# EJECTION FRACTION GRADING
# ==============================================================================

EF_GRADES = {
    "normal": {"min": 55, "max": 100, "label": "Normal", "severity": "normal"},
    "mildly_reduced": {"min": 45, "max": 54, "label": "Mildly Reduced", "severity": "mild"},
    "moderately_reduced": {"min": 30, "max": 44, "label": "Moderately Reduced", "severity": "moderate"},
    "severely_reduced": {"min": 0, "max": 29, "label": "Severely Reduced", "severity": "severe"},
}


# ==============================================================================
# RHYTHM CLASSIFICATION CRITERIA
# ==============================================================================

RHYTHM_CRITERIA = {
    "normal_sinus_rhythm": {
        "hr_range": (60, 100),
        "regularity": "regular",
        "rr_cv_max": 0.10,
        "description": "Normal heart rhythm with rate 60-100 bpm",
    },
    "sinus_bradycardia": {
        "hr_max": 60,
        "regularity": "regular",
        "description": "Slow but regular rhythm",
    },
    "sinus_tachycardia": {
        "hr_min": 100,
        "regularity": "regular",
        "description": "Fast but regular rhythm",
    },
    "atrial_fibrillation": {
        "rr_cv_min": 0.15,
        "regularity": "irregular",
        "description": "Irregularly irregular rhythm",
    },
}


# ==============================================================================
# RISK WEIGHTS
# ==============================================================================

RISK_WEIGHTS = {
    # EF contribution (max 30 points)
    "ef_severely_reduced": 30,
    "ef_moderately_reduced": 20,
    "ef_mildly_reduced": 10,
    
    # Heart rate contribution (max 25 points)
    "hr_critical": 25,  # <50 or >120
    "hr_borderline": 10,  # <60 or >100
    
    # Arrhythmia contribution (max 35 points)
    "afib_detected": 35,
    "pvcs_frequent": 15,
    "bradycardia_severe": 25,
    
    # HRV contribution (max 20 points)
    "rmssd_very_low": 20,
    "sdnn_very_low": 15,
    
    # Age contribution (max 10 points)
    "age_over_75": 10,
    "age_over_65": 5,
    
    # Clinical history (max 15 points each)
    "prior_mi": 15,
    "hypertension": 10,
    "diabetes": 10,
    "afib_history": 10,
}


# ==============================================================================
# RISK CATEGORIES
# ==============================================================================

RISK_CATEGORIES = {
    "low": {"min": 0, "max": 19, "color": "green", "urgency": "routine"},
    "moderate": {"min": 20, "max": 44, "color": "yellow", "urgency": "monitor"},
    "high": {"min": 45, "max": 69, "color": "orange", "urgency": "review"},
    "critical": {"min": 70, "max": 100, "color": "red", "urgency": "urgent"},
}


# ==============================================================================
# ECHO VIEW CONFIGURATION
# ==============================================================================

ECHO_VIEWS = {
    "PLAX": {
        "name": "Parasternal Long Axis",
        "priority": 1,
        "structures": ["LV", "LA", "Aortic Valve", "Mitral Valve"],
        "ef_suitable": True,
    },
    "PSAX": {
        "name": "Parasternal Short Axis",
        "priority": 2,
        "structures": ["LV", "RV", "Wall Segments"],
        "ef_suitable": False,
    },
    "A4C": {
        "name": "Apical 4-Chamber",
        "priority": 1,
        "structures": ["LV", "RV", "LA", "RA", "Mitral Valve", "Tricuspid Valve"],
        "ef_suitable": True,
    },
    "A2C": {
        "name": "Apical 2-Chamber",
        "priority": 2,
        "structures": ["LV", "LA", "Mitral Valve"],
        "ef_suitable": True,
    },
    "A3C": {
        "name": "Apical 3-Chamber",
        "priority": 3,
        "structures": ["LV", "LA", "Aortic Valve"],
        "ef_suitable": True,
    },
    "SUBCOSTAL": {
        "name": "Subcostal View",
        "priority": 3,
        "structures": ["LV", "RV", "LA", "RA"],
        "ef_suitable": False,
    },
}


# ==============================================================================
# PROCESSING CONFIGURATION
# ==============================================================================

PROCESSING_CONFIG = {
    "ecg": {
        "target_sample_rate": 500,
        "bandpass_low_hz": 0.5,
        "bandpass_high_hz": 45,
        "powerline_freq_hz": 50,  # or 60 for US
        "filter_order": 4,
        "notch_q": 30,
    },
    "echo": {
        "target_fps": 30,
        "target_resolution": 224,
        "speckle_filter": "bilateral",
        "normalization": "clahe",
    },
}


# ==============================================================================
# DETECTABLE CONDITIONS
# ==============================================================================

DETECTABLE_CONDITIONS = [
    # Rhythm conditions
    "Normal Sinus Rhythm",
    "Sinus Bradycardia",
    "Sinus Tachycardia",
    "Atrial Fibrillation",
    "Premature Ventricular Contractions (PVC)",
    "Premature Atrial Contractions (PAC)",
    
    # Conduction conditions
    "1st Degree AV Block",
    "Long QT Syndrome",
    
    # Functional conditions (with echo)
    "Reduced Ejection Fraction",
    "Wall Motion Abnormality",
    "LV Dilation",
    "LA Dilation",
    
    # HRV conditions
    "Low Heart Rate Variability",
    "Autonomic Imbalance",
]


# ==============================================================================
# DISCLAIMER CONFIGURATION
# ==============================================================================

DISCLAIMER_CONFIG = {
    "mandatory_disclaimer": True,
    "classification": "SCREENING",  # SCREENING, SUPPORTIVE, DIAGNOSTIC
    "requires_physician_review": True,
    "emergency_conditions": [
        "Atrial Fibrillation",
        "Severe Bradycardia",
        "Long QT Syndrome",
        "Severely Reduced EF",
    ],
}
