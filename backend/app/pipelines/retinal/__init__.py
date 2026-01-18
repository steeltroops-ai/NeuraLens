"""
Retinal Analysis Pipeline v4.0

Modular, scientifically accurate retinal image analysis.

Modules:
- constants: Clinical reference values from peer-reviewed literature  
- schemas: Pydantic data models for pipeline
- biomarker_extractor: Biomarker extraction algorithms
- clinical_assessment: DR grading, risk, findings
- validator: Image quality assessment (ETDRS)
- visualization: Heatmap and overlay generation
- router: Main API endpoints and pipeline orchestration

Author: NeuraLens Medical AI Team
"""

from .router import router

__all__ = ["router"]
