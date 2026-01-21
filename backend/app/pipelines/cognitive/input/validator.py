"""
Cognitive Input Validator - Production Grade
Comprehensive validation with detailed error reporting.
"""

from typing import List, Tuple
from datetime import datetime
import logging

from ..schemas import CognitiveSessionInput, TaskSession, TaskEvent, TaskCompletionStatus
from ..errors.codes import ErrorCode, PipelineError
from ..config import config

logger = logging.getLogger(__name__)


class ValidationResult:
    """Structured validation result"""
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.task_validity: dict = {}
    
    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


class CognitiveValidator:
    """
    Production-grade input validator.
    
    Validates:
    - Session structure
    - Task integrity
    - Event timing consistency
    - Anti-cheat heuristics
    """
    
    def validate(self, data: CognitiveSessionInput) -> List[str]:
        """
        Validate session input.
        Returns list of error strings (empty if valid).
        """
        result = ValidationResult()
        
        # Session-level validation
        self._validate_session(data, result)
        
        # Task-level validation
        for task in data.tasks:
            self._validate_task(task, result)
        
        if result.warnings:
            logger.warning(f"[VALIDATOR] Warnings: {result.warnings}")
        
        return result.errors
    
    def validate_detailed(self, data: CognitiveSessionInput) -> ValidationResult:
        """Return full validation result with warnings"""
        result = ValidationResult()
        self._validate_session(data, result)
        for task in data.tasks:
            self._validate_task(task, result)
        return result
    
    def _validate_session(self, data: CognitiveSessionInput, result: ValidationResult):
        """Session-level validations"""
        
        # Empty tasks
        if not data.tasks:
            result.errors.append(ErrorCode.E_INP_001.message)
            return
        
        # Session ID format (already validated by Pydantic, but double-check)
        if not data.session_id.startswith('sess_'):
            result.errors.append(ErrorCode.E_INP_005.message)
        
        # Too many tasks (anti-abuse)
        if len(data.tasks) > 10:
            result.warnings.append("Unusually high number of tasks submitted")
    
    def _validate_task(self, task: TaskSession, result: ValidationResult):
        """Task-level validations"""
        task_id = task.task_id
        
        # Duration check
        duration = (task.end_time - task.start_time).total_seconds()
        if duration < 0:
            result.errors.append(f"{ErrorCode.E_INP_003.message} (task: {task_id})")
            result.task_validity[task_id] = False
            return
        
        if duration < 5:  # Less than 5 seconds is suspicious
            result.warnings.append(f"Task {task_id} completed unusually fast ({duration:.1f}s)")
        
        if duration > 1800:  # More than 30 minutes
            result.warnings.append(f"Task {task_id} took unusually long ({duration:.1f}s)")
        
        # Event validation
        if not task.events:
            result.errors.append(f"{ErrorCode.E_INP_007.message} (task: {task_id})")
            result.task_validity[task_id] = False
            return
        
        # Monotonic timestamps
        is_monotonic = self._check_monotonic_timestamps(task.events)
        if not is_monotonic:
            result.errors.append(f"{ErrorCode.E_INP_006.message} (task: {task_id})")
            result.task_validity[task_id] = False
            return
        
        # Anti-cheat: Check for impossible reaction times
        suspicious_timings = self._check_suspicious_timings(task)
        if suspicious_timings:
            result.warnings.append(f"Task {task_id}: {suspicious_timings}")
        
        result.task_validity[task_id] = True
    
    def _check_monotonic_timestamps(self, events: List[TaskEvent]) -> bool:
        """Ensure event timestamps are monotonically increasing"""
        last_ts = -1.0
        for event in events:
            if event.timestamp < last_ts:
                return False
            last_ts = event.timestamp
        return True
    
    def _check_suspicious_timings(self, task: TaskSession) -> str:
        """Detect impossible or suspicious timing patterns"""
        
        # For reaction time tasks
        if "reaction" in task.task_id:
            reaction_times = []
            last_stim = None
            
            for event in task.events:
                if event.event_type == "stimulus_shown":
                    last_stim = event.timestamp
                elif event.event_type == "response_received" and last_stim is not None:
                    rt = event.timestamp - last_stim
                    reaction_times.append(rt)
                    last_stim = None
            
            if reaction_times:
                # Physically impossible reaction time
                too_fast = [rt for rt in reaction_times if rt < config.MIN_REACTION_TIME_MS]
                if len(too_fast) > len(reaction_times) * 0.5:
                    return f"Majority of reaction times below {config.MIN_REACTION_TIME_MS}ms (potential automation)"
                
                # Perfect consistency is suspicious
                if len(reaction_times) > 3:
                    import numpy as np
                    std = np.std(reaction_times)
                    if std < 10:  # Less than 10ms variance across trials
                        return "Suspiciously consistent reaction times"
        
        return ""
