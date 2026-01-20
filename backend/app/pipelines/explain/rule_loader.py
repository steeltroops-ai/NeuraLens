"""
Explanation Rule Loader
Dynamically loads pipeline-specific explanation rules

Supports both architecture patterns:
- Old: app.pipelines.{pipeline}.explanation_rules
- New: app.pipelines.{pipeline}.explanation.rules
"""

from typing import Dict, Any, Optional, Callable
from importlib import import_module
import logging

logger = logging.getLogger(__name__)


class PipelineRuleLoader:
    """Loads and caches explanation rules for each pipeline."""
    
    _cache: Dict[str, Any] = {}
    
    # Import paths to try (in order of preference)
    IMPORT_PATHS = [
        'app.pipelines.{pipeline}.explanation.rules',  # New architecture
        'app.pipelines.{pipeline}.explanation_rules',  # Legacy
    ]
    
    @classmethod
    def get_rules(cls, pipeline: str) -> Dict[str, Any]:
        """
        Load explanation rules for a pipeline.
        
        Tries multiple import paths to support both old and new architecture.
        
        Args:
            pipeline: Pipeline name (speech, retinal, radiology, etc.)
            
        Returns:
            Dictionary containing explanation rules and templates
        """
        if pipeline in cls._cache:
            return cls._cache[pipeline]
        
        # Try each import path
        for path_template in cls.IMPORT_PATHS:
            path = path_template.format(pipeline=pipeline)
            try:
                module = import_module(path)
                
                # Try to get the rules class first (new architecture)
                rules_class = getattr(module, f'{pipeline.title()}ExplanationRules', None)
                if rules_class is None:
                    # Try generic name
                    rules_class = getattr(module, 'ExplanationRules', None)
                
                if rules_class:
                    # New architecture with class
                    rules = cls._extract_rules_from_class(rules_class)
                else:
                    # Legacy architecture with module-level exports
                    rules = cls._extract_rules_from_module(module)
                
                cls._cache[pipeline] = rules
                logger.info(f"Loaded explanation rules for {pipeline} from {path}")
                return rules
                
            except ImportError:
                continue
            except Exception as e:
                logger.warning(f"Error loading rules from {path}: {e}")
                continue
        
        # No rules found in any path
        logger.info(f"No explanation rules found for {pipeline}, using defaults")
        return cls._get_default_rules(pipeline)
    
    @classmethod
    def _extract_rules_from_class(cls, rules_class) -> Dict[str, Any]:
        """Extract rules from a rules class (new architecture)."""
        return {
            'biomarker_explanations': getattr(rules_class, 'BIOMARKER_EXPLANATIONS', 
                                              getattr(rules_class, 'CONDITION_EXPLANATIONS', {})),
            'condition_explanations': getattr(rules_class, 'CONDITION_EXPLANATIONS', {}),
            'risk_level_messages': getattr(rules_class, 'RISK_TEMPLATES', 
                                          getattr(rules_class, 'RISK_LEVEL_MESSAGES', {})),
            'mandatory_disclaimer': getattr(rules_class, 'MANDATORY_DISCLAIMER', cls._default_disclaimer()),
            'quality_warnings': getattr(rules_class, 'QUALITY_WARNINGS', {}),
            'generate_explanation': getattr(rules_class, 'generate_explanation', None),
        }
    
    @classmethod
    def _extract_rules_from_module(cls, module) -> Dict[str, Any]:
        """Extract rules from module-level exports (legacy architecture)."""
        return {
            'biomarker_explanations': getattr(module, 'BIOMARKER_EXPLANATIONS', {}),
            'condition_explanations': getattr(module, 'CONDITION_EXPLANATIONS', {}),
            'risk_level_messages': getattr(module, 'RISK_LEVEL_MESSAGES', {}),
            'mandatory_disclaimer': getattr(module, 'MANDATORY_DISCLAIMER', cls._default_disclaimer()),
            'quality_warnings': getattr(module, 'QUALITY_WARNINGS', {}),
            'generate_explanation': getattr(module, 'generate_speech_explanation', 
                                           getattr(module, 'generate_explanation', None)),
        }
    
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
