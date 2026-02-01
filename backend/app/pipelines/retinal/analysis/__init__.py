"""
Retinal Pipeline - Analysis Module

Contains ML analysis and inference components:
- analyzer.py: Main ML analysis layer (model inference)
- enhanced_analyzer.py: v5.1 Enhanced with real deep learning (v5.1)

Author: NeuraLens Medical AI Team
Version: 5.1.0
"""

import logging
_logger = logging.getLogger(__name__)

# v5.1 Enhanced Analyzer (priority)
try:
    from .enhanced_analyzer import (
        EnhancedRetinalAnalyzer,
        enhanced_retinal_analyzer,
        AnalyzerConfig,
    )
    ENHANCED_AVAILABLE = True
    _logger.info("v5.1 Enhanced analyzer loaded successfully")
except ImportError as e:
    _logger.warning(f"Enhanced analyzer not available: {e}")
    ENHANCED_AVAILABLE = False
    EnhancedRetinalAnalyzer = None
    enhanced_retinal_analyzer = None
    AnalyzerConfig = None

# Legacy analyzer (may have missing dependencies)
try:
    from .analyzer import RealtimeRetinalProcessor, realtime_retinal_processor
except ImportError as e:
    _logger.warning(f"Legacy analyzer not available: {e}")
    RealtimeRetinalProcessor = None
    realtime_retinal_processor = None

__all__ = [
    "RealtimeRetinalProcessor",
    "realtime_retinal_processor",
    
    # v5.1
    "EnhancedRetinalAnalyzer",
    "enhanced_retinal_analyzer",
    "AnalyzerConfig",
    "ENHANCED_AVAILABLE",
]


