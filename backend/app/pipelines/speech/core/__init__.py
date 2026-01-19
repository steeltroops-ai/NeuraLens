"""
Speech Pipeline - Core Module

Main orchestration and service entry for speech analysis.
"""

from .service import ResearchGradeSpeechService, PipelineConfig

# Alias for backward compatibility
SpeechPipelineService = ResearchGradeSpeechService
SpeechAnalysisService = ResearchGradeSpeechService

__all__ = [
    'ResearchGradeSpeechService',
    'SpeechPipelineService',
    'SpeechAnalysisService',
    'PipelineConfig',
]
