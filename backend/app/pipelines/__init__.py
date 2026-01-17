"""
MediLens Diagnostic Pipelines
Exports routers for all diagnostic modules
"""

from app.api.v1.endpoints import speech

# Export routers for API integration
__all__ = ['speech']
