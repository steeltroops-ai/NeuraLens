"""
Retinal Pipeline - Output Module

Contains output generation components:
- report_generator.py: Clinical PDF report generation
- visualization.py: Heatmaps, overlays, charts

Author: NeuraLens Medical AI Team
Version: 4.0.0
"""

from .report_generator import ReportGenerator, report_generator
from .visualization import (
    RetinalVisualizationService,
    visualization_service,
    ColorPalette,
    VisualizationConfig,
)

__all__ = [
    "ReportGenerator",
    "report_generator",
    "RetinalVisualizationService",
    "visualization_service",
    "ColorPalette",
    "VisualizationConfig",
]
