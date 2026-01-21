"""
Cognitive Pipeline Configuration - Production Grade
"""

from pydantic_settings import BaseSettings


class CognitiveConfig(BaseSettings):
    """Configuration for cognitive assessment pipeline."""
    
    # Pipeline Identity
    PIPELINE_NAME: str = "cognitive"
    VERSION: str = "2.0.0"
    
    # Thresholds
    MIN_REACTION_TIME_MS: float = 100.0  # Faster is physiologically impossible
    MAX_REACTION_TIME_MS: float = 2000.0  # Slower indicates distraction/lapse
    MIN_TRIALS_FOR_VALIDITY: int = 5
    
    # Risk Thresholds
    RISK_THRESHOLD_LOW: float = 0.3
    RISK_THRESHOLD_MODERATE: float = 0.5
    RISK_THRESHOLD_HIGH: float = 0.7
    
    # Domain Weights (must sum to ~1.0)
    WEIGHT_MEMORY: float = 0.25
    WEIGHT_ATTENTION: float = 0.20
    WEIGHT_EXECUTIVE: float = 0.20
    WEIGHT_SPEED: float = 0.20
    WEIGHT_INHIBITION: float = 0.15
    
    # Confidence
    MIN_CONFIDENCE_THRESHOLD: float = 0.6
    
    # Timeouts
    PROCESSING_TIMEOUT_MS: int = 30000
    
    class Config:
        env_prefix = "COGNITIVE_"


config = CognitiveConfig()
