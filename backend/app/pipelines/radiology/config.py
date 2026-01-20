"""
Radiology Pipeline - Configuration

Centralized configuration for the radiology/X-ray analysis pipeline.
"""

from typing import List, Dict, Set
from dataclasses import dataclass


@dataclass
class RadiologyConfig:
    """Configuration values for radiology pipeline."""
    
    # Version info
    VERSION = "4.0.0"
    PIPELINE_NAME = "radiology"
    
    # File constraints
    MAX_FILE_SIZE_MB = 10
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
    
    # Image constraints
    MIN_RESOLUTION = 224
    MAX_RESOLUTION = 4096
    RECOMMENDED_RESOLUTION = 512
    MODEL_INPUT_SIZE = 224
    
    # DICOM constraints
    MAX_DICOM_SIZE_MB = 50
    MAX_VOLUME_SIZE_MB = 500
    MIN_VOLUME_SLICES = 10
    MAX_VOLUME_SLICES = 1000
    
    # Quality thresholds
    MIN_CONTRAST = 0.5
    MIN_INTENSITY_VARIANCE = 100
    MIN_QUALITY_SCORE = 0.5
    
    # Model configuration
    MODEL_NAME = "densenet121-res224-all"
    PATHOLOGY_COUNT = 18
    CONFIDENCE_THRESHOLD = 0.15  # 15% minimum to report finding
    
    # Risk scoring weights
    CRITICAL_WEIGHT = 0.5
    HIGH_WEIGHT = 0.3
    MODERATE_WEIGHT = 0.15
    LOW_WEIGHT = 0.05
    
    # Timeouts
    ANALYSIS_TIMEOUT_SECONDS = 30
    MODEL_LOAD_TIMEOUT_SECONDS = 60
    
    # Processing settings
    ENABLE_HEATMAP = True
    ENABLE_SEGMENTATION = True
    HEATMAP_OPACITY = 0.4


# Pathology definitions with clinical metadata
PATHOLOGIES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum"
]

# Priority conditions for risk assessment
CRITICAL_CONDITIONS: Set[str] = {"Pneumothorax", "Mass"}
HIGH_RISK_CONDITIONS: Set[str] = {"Pneumonia", "Consolidation", "Edema", "Effusion"}
MODERATE_CONDITIONS: Set[str] = {"Cardiomegaly", "Atelectasis", "Nodule", "Infiltration"}
LOW_RISK_CONDITIONS: Set[str] = {"Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"}

# Pathology clinical information
PATHOLOGY_INFO: Dict[str, Dict] = {
    "Atelectasis": {
        "description": "Partial or complete lung collapse",
        "category": "Collapse",
        "urgency": "moderate",
        "accuracy": 85
    },
    "Consolidation": {
        "description": "Lung tissue filled with fluid/pus",
        "category": "Opacity",
        "urgency": "high",
        "accuracy": 86
    },
    "Infiltration": {
        "description": "Substance denser than air in lung",
        "category": "Opacity",
        "urgency": "moderate",
        "accuracy": 81
    },
    "Pneumothorax": {
        "description": "Air in pleural space (collapsed lung)",
        "category": "Emergency",
        "urgency": "critical",
        "accuracy": 88
    },
    "Edema": {
        "description": "Fluid in lung tissue (pulmonary edema)",
        "category": "Fluid",
        "urgency": "high",
        "accuracy": 84
    },
    "Emphysema": {
        "description": "Destruction of alveoli (COPD)",
        "category": "COPD",
        "urgency": "moderate",
        "accuracy": 82
    },
    "Fibrosis": {
        "description": "Scarring/thickening of lung tissue",
        "category": "Scarring",
        "urgency": "moderate",
        "accuracy": 80
    },
    "Effusion": {
        "description": "Fluid in pleural space",
        "category": "Fluid",
        "urgency": "moderate",
        "accuracy": 89
    },
    "Pneumonia": {
        "description": "Lung infection (bacterial, viral)",
        "category": "Infection",
        "urgency": "high",
        "accuracy": 92
    },
    "Pleural_Thickening": {
        "description": "Thickened pleural membrane",
        "category": "Pleural",
        "urgency": "low",
        "accuracy": 80
    },
    "Cardiomegaly": {
        "description": "Enlarged heart",
        "category": "Cardiac",
        "urgency": "moderate",
        "accuracy": 90
    },
    "Nodule": {
        "description": "Small rounded opacity in lung",
        "category": "Mass",
        "urgency": "moderate",
        "accuracy": 78
    },
    "Mass": {
        "description": "Large opacity (>3cm), possible tumor",
        "category": "Mass",
        "urgency": "high",
        "accuracy": 82
    },
    "Hernia": {
        "description": "Hiatal hernia visible on chest X-ray",
        "category": "Structural",
        "urgency": "low",
        "accuracy": 75
    },
    "Lung Lesion": {
        "description": "Abnormal tissue in lung",
        "category": "Mass",
        "urgency": "moderate",
        "accuracy": 78
    },
    "Fracture": {
        "description": "Rib fracture",
        "category": "Trauma",
        "urgency": "moderate",
        "accuracy": 82
    },
    "Lung Opacity": {
        "description": "Any opacity in lung field",
        "category": "Opacity",
        "urgency": "varies",
        "accuracy": 85
    },
    "Enlarged Cardiomediastinum": {
        "description": "Widened mediastinum",
        "category": "Cardiac",
        "urgency": "moderate",
        "accuracy": 84
    }
}

# Supported DICOM modalities
SUPPORTED_MODALITIES: Set[str] = {"CR", "DX", "CT", "MR", "PT", "NM", "XA", "RF", "MG"}

# Model training datasets
TRAINING_DATASETS: List[Dict[str, str]] = [
    {"name": "NIH ChestX-ray14", "size": "112,120 images"},
    {"name": "CheXpert", "size": "224,316 images"},
    {"name": "MIMIC-CXR", "size": "377,110 images"},
    {"name": "PadChest", "size": "160,000+ images"},
    {"name": "COVID-Chestxray", "size": "1,000+ images"},
    {"name": "RSNA Pneumonia", "size": "26,684 images"},
    {"name": "VinDr-CXR", "size": "18,000 images"},
    {"name": "Google NIH", "size": "112,120 images"}
]
