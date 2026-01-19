"""
Explanation Rule Loader
Dynamically loads pipeline-specific explanation rules
"""

from typing import Dict, Any, Optional, Callable
from importlib import import_module
import logging

logger = logging.getLogger(__name__)


class PipelineRuleLoader:
    """Loads and caches explanation rules for each pipeline."""
    
    _cache: Dict[str, Any] = {}
    
    @classmethod
    def get_rules(cls, pipeline: str) -> Dict[str, Any]:
        """
        Load explanation rules for a pipeline.
        
        Args:
            pipeline: Pipeline name (speech, retinal, etc.)
            
        Returns:
            Dictionary containing explanation rules and templates
        """
        if pipeline in cls._cache:
            return cls._cache[pipeline]
        
        try:
            module = import_module(f'app.pipelines.{pipeline}.explanation_rules')
            
            rules = {
                'biomarker_explanations': getattr(module, 'BIOMARKER_EXPLANATIONS', {}),
                'condition_explanations': getattr(module, 'CONDITION_EXPLANATIONS', {}),
                'risk_level_messages': getattr(module, 'RISK_LEVEL_MESSAGES', {}),
                'mandatory_disclaimer': getattr(module, 'MANDATORY_DISCLAIMER', cls._default_disclaimer()),
                'quality_warnings': getattr(module, 'QUALITY_WARNINGS', {}),
                'generate_explanation': getattr(module, 'generate_speech_explanation', None),
            }
            
            cls._cache[pipeline] = rules
            logger.info(f"Loaded explanation rules for {pipeline}")
            return rules
            
        except ImportError as e:
            logger.warning(f"No explanation rules found for {pipeline}: {e}")
            return cls._get_default_rules(pipeline)
        except Exception as e:
            logger.error(f"Error loading rules for {pipeline}: {e}")
            return cls._get_default_rules(pipeline)
    
    @classmethod
    def get_generator(cls, pipeline: str) -> Optional[Callable]:
        """Get the explanation generator function for a pipeline."""
        rules = cls.get_rules(pipeline)
        return rules.get('generate_explanation')
    
    @classmethod
    def _get_default_rules(cls, pipeline: str) -> Dict[str, Any]:
        """Return default rules if pipeline-specific rules not found."""
        return {
            'biomarker_explanations': {},
            'condition_explanations': {},
            'risk_level_messages': cls._default_risk_messages(),
            'mandatory_disclaimer': cls._default_disclaimer(),
            'quality_warnings': {},
            'generate_explanation': None,
        }
    
    @staticmethod
    def _default_disclaimer() -> str:
        return """
IMPORTANT DISCLAIMER: This analysis is for informational screening 
purposes only and is NOT a medical diagnosis. Always consult a qualified 
healthcare provider for medical advice, diagnosis, or treatment.
"""
    
    @staticmethod
    def _default_risk_messages() -> Dict[str, Dict[str, str]]:
        return {
            'low': {
                'summary': 'Your analysis shows low risk indicators.',
                'action': 'Continue routine monitoring.'
            },
            'moderate': {
                'summary': 'Some indicators are outside typical ranges.',
                'action': 'Consider follow-up assessment.'
            },
            'high': {
                'summary': 'Several indicators warrant clinical attention.',
                'action': 'Consult with a healthcare provider.'
            },
            'critical': {
                'summary': 'Significant abnormalities detected.',
                'action': 'Seek prompt medical evaluation.'
            }
        }
    
    @classmethod
    def clear_cache(cls):
        """Clear the rules cache."""
        cls._cache.clear()


# Convenience function
def load_pipeline_rules(pipeline: str) -> Dict[str, Any]:
    """Load explanation rules for a pipeline."""
    return PipelineRuleLoader.get_rules(pipeline)
