"""
Retinal Pipeline - Core Module

Main orchestration and service entry for retinal analysis.
"""

from .service import (
    ResearchGradeRetinalService,
    RetinalPipelineService,
    ServiceConfig,
)
from .orchestrator import (
    ExecutionContext,
    ReceiptConfirmation,
    execute_with_retry,
    handle_hard_stop,
    AuditLogger,
    SAFETY_DISCLAIMERS,
    get_disclaimers,
    create_execution_context,
    STAGE_CONFIGS,
)

# Backward compatibility alias
RetinalAnalysisService = ResearchGradeRetinalService

__all__ = [
    'ResearchGradeRetinalService',
    'RetinalPipelineService',
    'RetinalAnalysisService',
    'ServiceConfig',
    'ExecutionContext',
    'ReceiptConfirmation',
    'execute_with_retry',
    'handle_hard_stop',
    'AuditLogger',
    'SAFETY_DISCLAIMERS',
    'get_disclaimers',
    'create_execution_context',
    'STAGE_CONFIGS',
]
