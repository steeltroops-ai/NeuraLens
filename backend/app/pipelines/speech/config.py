"""
Speech Analysis Pipeline - Configuration
Clinical constants and processing parameters
"""

from typing import Dict, Tuple


# Input constraints from PRD
INPUT_CONSTRAINTS = {
    "max_file_size_mb": 10,
    "min_duration_sec": 3,
    "max_duration_sec": 60,
    "target_sample_rate": 16000,
    "target_channels": 1,  # Mono
    "min_signal_db": -40,  # Reject if too quiet
    "supported_formats": ["wav", "mp3", "webm", "ogg", "m4a"]
}

# Supported MIME types
SUPPORTED_MIME_TYPES = {
    "audio/wav", "audio/wave", "audio/x-wav",
    "audio/mpeg", "audio/mp3",
    "audio/webm", "audio/webm;codecs=opus",
    "audio/ogg", "audio/ogg;codecs=opus",
    "audio/mp4", "audio/m4a", "audio/x-m4a"
}

# Clinical normal ranges based on published research
# Reference: Tsanas et al. (2012), Konig et al. (2015)
BIOMARKER_NORMAL_RANGES: Dict[str, Tuple[float, float]] = {
    "jitter": (0.00, 1.04), # Percent (Local)
    "shimmer": (0.00, 3.81), # Percent (Local)
    "hnr": (20.0, 30.0), # dB
    "cpps": (14.0, 30.0), # dB (Smoothed) - New Gold Standard
    "speech_rate": (3.5, 6.5),
    "pause_ratio": (0.10, 0.25),
    "fluency_score": (0.75, 1.0),
    "voice_tremor": (0.0, 0.15),
    "articulation_clarity": (0.9, 1.1), # FCR ratio near 1.0 is healthy
    "prosody_variation": (20.0, 100.0), # Hz (F0 Std Dev)
}

# Abnormal thresholds that trigger clinical alerts
BIOMARKER_ABNORMAL_THRESHOLDS: Dict[str, float] = {
    "jitter": 2.0,           
    "shimmer": 5.0,         
    "hnr": 12.0, # Below this is pathological
    "cpps": 11.0, # Below this is Dysphonia
    "speech_rate_low": 2.5,  
    "speech_rate_high": 7.5, 
    "pause_ratio": 0.40,    
    "fluency_score": 0.50,   
    "voice_tremor": 0.25,    
    "articulation_clarity": 1.2,  # >1.2 implies centralized vowels (Dysarthria)
    "prosody_variation": 10.0,    # Monotone
}

# Clinical weights for risk calculation (based on published research)
RISK_WEIGHTS: Dict[str, float] = {
    "jitter": 0.10,
    "shimmer": 0.08,
    "hnr": 0.07,
    "cpps": 0.15, # Strong predictor of dysphonia
    "speech_rate": 0.10,
    "pause_ratio": 0.15,
    "fluency_score": 0.05,
    "voice_tremor": 0.15, 
    "articulation_clarity": 0.05,
    "prosody_variation": 0.10,
}

# Risk categories
RISK_CATEGORIES = {
    (0, 25): "low",
    (25, 50): "moderate",
    (50, 75): "high",
    (75, 100): "critical"
}

# Condition patterns for probability estimation
CONDITION_PATTERNS = {
    "parkinsons": {
        "voice_tremor": 0.30,
        "jitter": 0.25,
        "speech_rate_slow": 0.20,
        "articulation_clarity": 0.15,
        "prosody_variation": 0.10
    },
    "alzheimers": {
        "pause_ratio": 0.35,
        "fluency_score": 0.25,
        "speech_rate_slow": 0.20,
        "prosody_variation": 0.20
    },
    "depression": {
        "prosody_variation": 0.40,
        "speech_rate_slow": 0.30,
        "energy_low": 0.30
    },
    "dysarthria": {
        "articulation_clarity": 0.40,
        "shimmer": 0.30,
        "hnr": 0.30
    }
}

# Tremor frequency bands (Hz)
TREMOR_BANDS = {
    "parkinsonian_resting": (4.0, 6.0),
    "essential_postural": (4.0, 12.0),
    "physiological_normal": (8.0, 12.0)
}

# Processing parameters
PROCESSING_CONFIG = {
    "f0_min_hz": 75,
    "f0_max_hz": 500,
    "frame_length_ms": 25,
    "hop_length_ms": 10,
    "silence_threshold_db": -40,
    "whisper_model": "tiny",  # Use tiny for speed
}

# Recommendation templates
RECOMMENDATIONS = {
    "low_risk": [
        "Voice biomarkers within normal range",
        "Continue annual voice monitoring",
        "No immediate clinical action required"
    ],
    "moderate_risk": [
        "Some biomarkers slightly outside normal range",
        "Consider follow-up assessment in 3-6 months",
        "Consult healthcare provider if symptoms develop"
    ],
    "high_risk": [
        "Multiple biomarkers indicate potential concern",
        "Recommend neurological evaluation",
        "Schedule appointment with specialist"
    ],
    "critical_risk": [
        "Significant abnormalities detected",
        "Urgent neurological evaluation recommended",
        "Seek medical attention promptly"
    ]
}
